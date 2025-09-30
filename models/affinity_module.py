import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import numpy as np
from collections import defaultdict
import clip
import sonata
try:
    import flash_attn
except ImportError:
    flash_attn = None
import open3d as o3d
from xdecoder.modeling.BaseModel import BaseModel
from xdecoder.modeling import build_model
from xdecoder.modeling.architectures.xdecoder_model import GeneralizedXdecoder
from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from sklearn.neighbors import KDTree
from models.utils.visualization import (
    get_pca_color,
)
import torch_scatter
import faiss
from tqdm import tqdm
import glob
from collections import abc
from detectron2.config import LazyConfig
from detectron2.utils.logger import setup_logger

class MinkowskiResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ME.MinkowskiConvolution(channels, channels, kernel_size=3, dimension=3)
        self.norm1 = ME.MinkowskiBatchNorm(channels)
        self.conv2 = ME.MinkowskiConvolution(channels, channels, kernel_size=3, dimension=3)
        self.norm2 = ME.MinkowskiBatchNorm(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = MEF.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += identity
        return MEF.relu(out)

class AffinityPredictor(nn.Module):
    def __init__(self, input_dim, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_layer = nn.Sequential(
            ME.MinkowskiConvolution(input_dim, hidden_dim, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(hidden_dim),
            ME.MinkowskiReLU(),
        )
        self.res_blocks = nn.Sequential(
            MinkowskiResBlock(hidden_dim),
            MinkowskiResBlock(hidden_dim),
            MinkowskiResBlock(hidden_dim),
            MinkowskiResBlock(hidden_dim),
        )
        self.output_layer = ME.MinkowskiConvolution(hidden_dim, embed_dim, kernel_size=1, dimension=3)

    def forward(self, x_s: ME.SparseTensor) -> ME.SparseTensor:
        out = self.input_layer(x_s)
        out = self.res_blocks(out)
        out = self.output_layer(out)
        return out

    def get_param_groups(self):
        """
        Divide the model's parameters into three groups to apply differential learning rates.
        - ‘input’: Adapter layers, which should have a smaller learning rate.
        - ‘middle’: Core processing layers, using the baseline learning rate.
        - ‘output’: Projection heads, which can have a larger learning rate.
        """
        return {
            "input": list(self.input_layer.parameters()),
            "middle": list(self.res_blocks.parameters()),
            "output": list(self.output_layer.parameters())
        }

# from models.modeling.meta_arch.mink_unet import mink_unet
# class MinkUNetWrapper(nn.Module):
#     def __init__(self, in_channels, out_channels, D=3, arch='MinkUNet18A'):
#         super().__init__()
#         self.mink_unet = mink_unet(in_channels, out_channels, D, arch)

#     def forward(self, x_s):
#         temp_out, final_out = self.mink_unet(x_s)
#         return final_out

#     def get_param_groups(self):
#         input_params = []
#         middle_params = []
#         output_params = []

#         for name, param in self.mink_unet.named_parameters():
#             if self._is_input_layer(name):
#                 input_params.append(param)
#             elif self._is_output_layer(name):
#                 output_params.append(param)
#             else:
#                 middle_params.append(param)

#         return {
#             "input": input_params,
#             "middle": middle_params,
#             "output": output_params
#         }

#     def _is_input_layer(self, layer_name):
#         input_patterns = [
#             'conv0p1s1',
#             'bn0',
#         ]
#         return any(pattern in layer_name for pattern in input_patterns)

#     def _is_output_layer(self, layer_name):
#         output_patterns = [
#             'final',
#         ]
#         return any(pattern in layer_name for pattern in output_patterns)

class SonataXAffinityTrainer(nn.Module):
    def __init__(self, cfg, xdecoder_cfg, scene_config, device='cuda', use_lseg=True):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.use_lseg = use_lseg
        self.scene_config = scene_config
        self.use_ape = cfg.use_ape

        if self.use_lseg:
            from models.Lseg.lseg_module import LSegModule
            from models.Lseg.lseg_multievalmodule import LSeg_MultiEvalModule
            import torchvision.transforms as transforms
            from encoding.models.sseg import BaseNet

            module = LSegModule.load_from_checkpoint(
                checkpoint_path=cfg.lseg_model_path,
                data_path='../datasets/',
                dataset='ade20k',
                backbone='clip_vitl16_384',
                aux=False,
                num_features=256,
                aux_weight=0,
                se_loss=False,
                se_weight=0,
                base_lr=0,
                batch_size=1,
                max_epochs=0,
                ignore_index=255,
                dropout=0.0,
                scale_inv=False,
                augment=False,
                no_batchnorm=False,
                widehead=True,
                widehead_hr=False,
                map_locatin="cuda",
                arch_option=0,
                block_depth=0,
                activation='lrelu',
            )

            if isinstance(module.net, BaseNet):
                model = module.net
            else:
                model = module
            model = model.eval().cuda()
            model.mean = [0.5, 0.5, 0.5]
            model.std = [0.5, 0.5, 0.5]
            model.crop_size = 640
            model.base_size = 640

            self.lseg_evaluator = LSeg_MultiEvalModule(
                model, scales=([1]), flip=True
            ).cuda()
            self.lseg_evaluator.eval()

            for param in self.lseg_evaluator.parameters():
                param.requires_grad = False

            self.lseg_normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            self.text_model = module.net.clip_pretrained.to(torch.float32).cuda()

            if hasattr(cfg, 'prompt_eng') and cfg.prompt_eng:
                print('Use prompt engineering: a XX in a scene')
                labelset = [ "a " + label + " in a scene" for label in cfg.all_label]
                labelset.append('background and other objects')
            print(f"Final labelset: {labelset}")
            self.text_features = self.extract_text_feature(labelset).float()
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
            print("LSeg model loaded and frozen.")
        elif self.use_ape:
            from xdecoder_test.models.predictor_lazy import VisualizationDemo

            self.config_file = "third_party/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k.py"
            self.confidence_threshold = 0.1
            print('Use prompt engineering: a XX in a scene')
            labelset = [ "a " + label + " in a scene" for label in cfg.all_label]
            labelset.append('background')
            labelset = ','.join(labelset)
            self.text_prompt = labelset
            self.with_sseg = True
            self.with_box = False
            self.with_mask = False

            self.opts = [
                "train.init_checkpoint=third_party/APE/checkpoint/APE-L_D.pth",
                "model.model_language.cache_dir=",
                "model.model_vision.select_box_nums_for_evaluation=500",
                "model.model_vision.text_feature_bank_reset=True"
            ]
            setup_logger(name="fvcore")
            setup_logger(name="ape")
            setup_logger(name="timm")
            self.ape_cfg = self._setup_cfg()
            self.demo = VisualizationDemo(self.ape_cfg, args=self._create_args_namespace())
        else:
            self.xdecoder_cfg = xdecoder_cfg
            self.xdecoder_cfg['device'] = self.device
            print("Loading teacher models...")
            self.xdecoder_teacher = BaseModel(
                self.xdecoder_cfg, build_model(self.xdecoder_cfg)
                ).from_pretrained(os.path.join(self.xdecoder_cfg['RESUME_FROM'])).eval().cuda()
            self.xdecoder_teacher.eval()
            for param in self.xdecoder_teacher.parameters():
                param.requires_grad = False
            stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int32).tolist() for _ in range(len(cfg.all_label))]
            stuff_dataset_id_to_contiguous_id = {x: x for x in range(len(cfg.all_label))}
            MetadataCatalog.get("demo").set(
                stuff_colors=stuff_colors,
                stuff_classes=cfg.all_label,
                stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
            )
            print('Use prompt engineering: a XX in a scene')
            labelset = [ "a " + label + " in a scene" for label in cfg.all_label]
            labelset.append('background')
            print(f"Final labelset: {labelset}")
            self.xdecoder_teacher.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(labelset, is_eval=True)
            metadata = MetadataCatalog.get('demo')
            self.xdecoder_teacher.model.metadata = metadata
            self.xdecoder_teacher.model.sem_seg_head.num_classes = len(cfg.all_label)
            print("X-Decoder teacher loaded and frozen.")

        if flash_attn is not None:
            self.sonata_teacher = sonata.load("sonata", repo_id="facebook/sonata").cuda()
        else:
            custom_config = dict(
                enc_patch_size=[1024 for _ in range(5)],
                enable_flash=False,
            )
            self.sonata_teacher = sonata.load(
                "sonata", repo_id="facebook/sonata", custom_config=custom_config
            ).cuda()
        self.sonata_teacher.eval()
        for param in self.sonata_teacher.parameters():
            param.requires_grad = False
        print("Sonata teacher loaded and frozen.")

        xdecoder_feature_dim = 512+6
        self.affinity_student = AffinityPredictor(
            input_dim=xdecoder_feature_dim,
            embed_dim=128,
            hidden_dim=512,

        ).cuda()
        print("AffinityPredictor student created.")

        self.criterion = nn.CrossEntropyLoss()

        self.num_anchors_per_scene = 4096 # How many anchor points are sampled per scene for learning?
        self.num_negatives_per_anchor = 63  # How many negative samples should be assigned to each anchor?
        self.info_nce_temperature = 0.07   # InfoNCE Loss Temperature Coefficient

    def _create_args_namespace(self):
        class Args:
            def __init__(self):
                self.config_file = None
                self.confidence_threshold = 0.1
                self.text_prompt = None
                self.with_box = False
                self.with_mask = False
                self.with_sseg = True
                self.opts = []

        args = Args()
        args.config_file = self.config_file
        args.confidence_threshold = self.confidence_threshold
        args.text_prompt = self.text_prompt
        args.with_box = self.with_box
        args.with_mask = self.with_mask
        args.with_sseg = self.with_sseg
        args.opts = self.opts

        return args

    def _setup_cfg(self):
        cfg = LazyConfig.load(self.config_file)
        cfg = LazyConfig.apply_overrides(cfg, self.opts)


        if "output_dir" in cfg.model:
            cfg.model.output_dir = cfg.train.output_dir
        if "model_vision" in cfg.model and "output_dir" in cfg.model.model_vision:
            cfg.model.model_vision.output_dir = cfg.train.output_dir
        if "train" in cfg.dataloader:
            if isinstance(cfg.dataloader.train, abc.MutableSequence):
                for i in range(len(cfg.dataloader.train)):
                    if "output_dir" in cfg.dataloader.train[i].mapper:
                        cfg.dataloader.train[i].mapper.output_dir = cfg.train.output_dir
            else:
                if "output_dir" in cfg.dataloader.train.mapper:
                    cfg.dataloader.train.mapper.output_dir = cfg.train.output_dir

        if "model_vision" in cfg.model:
            cfg.model.model_vision.test_score_thresh = self.confidence_threshold
        else:
            cfg.model.test_score_thresh = self.confidence_threshold

        return cfg

    def extract_text_feature(self, labelset):

        if isinstance(labelset, str):
            lines = labelset.split(",")
        elif isinstance(labelset, list):
            lines = labelset
        else:
            raise NotImplementedError

        labels = []
        for line in lines:
            label = line
            labels.append(label)
        text = clip.tokenize(labels)
        text = text.cuda()
        text_features = self.text_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def lift_lseg_features(self, batch_data, max_batch_size=24):
        """
        Extracts features from 2D images using LSeg and lifts them to a 3D point cloud.
        Enhanced version with aggressive memory management to prevent accumulation.
        """
        (
            scene_coords, scene_coords_3d, scene_inds_reconstruct, scene_label,
            ori_coords_3ds, coords_3ds, feat_3ds, gauss_featuress, labels_3ds,
            binary_label_3ds, label_2ds, imgs, x_labels, y_labels, mask_2ds,
            inds_reconstructs, unique_maps, mappings, captions, scene_gauss_features
        ) = batch_data

        num_points = scene_coords.shape[0]
        feature_dim = 512
        num_views = len(imgs)

        print(f"Processing {num_views} views with LSeg using batch size {max_batch_size}...")
        sum_features = torch.zeros((num_points, feature_dim), dtype=torch.float32, device='cpu')
        counter = torch.zeros((num_points, 1), dtype=torch.float32, device='cpu')
        num_batches = (num_views + max_batch_size - 1) // max_batch_size

        for batch_idx in tqdm(range(num_batches), desc="Processing Lseg feature:"):
            start_idx = batch_idx * max_batch_size
            end_idx = min(start_idx + max_batch_size, num_views)
            current_batch_size = end_idx - start_idx

            batch_images = []
            original_shapes = []

            for i in range(start_idx, end_idx):
                img_tensor = imgs[i].float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)
                original_shapes.append(img_tensor.shape[1:])


                image_tensor_processed = self.lseg_normalize(img_tensor)
                image_tensor_processed = F.interpolate(
                    image_tensor_processed.unsqueeze(0),
                    size=(240, 320),
                    mode='bilinear',
                    align_corners=True
                ).squeeze(0)

                batch_images.append(image_tensor_processed)

            batch_tensor = torch.stack(batch_images, dim=0).to(self.device)
            with torch.no_grad():

                self.lseg_evaluator.eval()
                if hasattr(self.lseg_evaluator, 'forward'):
                    batch_feat_2d = self.lseg_evaluator.forward(batch_tensor, label_set='')
                else:
                    batch_inputs = [batch_tensor[i:i+1] for i in range(current_batch_size)]
                    batch_outputs = self.lseg_evaluator.parallel_forward(batch_inputs)
                    batch_feat_2d = torch.cat([output[0] for output in batch_outputs], dim=0)

                for i, feat_2d in enumerate(batch_feat_2d):
                    view_idx = start_idx + i
                    original_shape = original_shapes[i]


                    feat_2d_interpolated = F.interpolate(
                        feat_2d.unsqueeze(0),
                        size=original_shape,
                        mode='bilinear',
                        align_corners=True
                    ).squeeze(0)

                    mask_this_view = (ori_coords_3ds[:, 0] == view_idx)
                    all_mask_this_view = (mask_2ds[:, 0] == view_idx)

                    if mask_this_view.any():
                        mask_2d = mask_2ds[all_mask_this_view][:, 1].bool()
                        point_indices = torch.where(mask_2d)[0]

                        x_coords = x_labels[mask_this_view]
                        y_coords = y_labels[mask_this_view]

                        lifted_features = feat_2d_interpolated[:, x_coords, y_coords].permute(1, 0).cpu()

                        sum_features.index_add_(0, point_indices.cpu(), lifted_features)
                        counter.index_add_(0, point_indices.cpu(),
                                        torch.ones((len(point_indices), 1), dtype=torch.float32))
                    del feat_2d_interpolated

            del batch_images, batch_tensor, batch_feat_2d
            torch.cuda.empty_cache()
        counter[counter == 0] = 1e-6
        final_scene_features = sum_features / counter

        seen_mask = (counter.squeeze() > 1e-5)
        unseen_mask = ~seen_mask

        if unseen_mask.any() and seen_mask.any():
            seen_coords = scene_coords[seen_mask].cpu().numpy()
            unseen_coords = scene_coords[unseen_mask].cpu().numpy()

            from sklearn.neighbors import KDTree
            kdtree = KDTree(seen_coords)
            distances, indices = kdtree.query(unseen_coords, k=1)
            neighbor_features = final_scene_features[seen_mask][indices.flatten()]
            final_scene_features[unseen_mask] = neighbor_features


        final_scene_features = final_scene_features.to(self.device)
        return final_scene_features, self.text_features, self.logit_scale

    def lift_xdecoder_features(self, batch_data):
        (
            scene_coords,
            scene_coords_3d,
            scene_inds_reconstruct,
            scene_label,
            ori_coords_3ds,
            coords_3ds,
            feat_3ds,
            gauss_featuress,
            labels_3ds,
            binary_label_3ds,
            label_2ds,
            imgs,
            x_labels,
            y_labels,
            mask_2ds,
            inds_reconstructs,
            unique_maps,
            mappings,
            captions,
            scene_gauss_features
        ) = batch_data

        point_info = defaultdict(list)
        num_points = scene_coords.shape[0]
        num_classes = len(self.cfg.all_label)
        feature_dim = 512
        counter = torch.zeros((scene_coords.shape[0]), dtype=scene_label.dtype)

        view_counts = torch.zeros(num_points, dtype=torch.long, device=self.device)
        for view_idx in range(len(imgs)):
            all_mask_this_view = (mask_2ds[:, 0] == view_idx)
            mask_2d = mask_2ds[all_mask_this_view][:, 1].bool().to(view_counts.device)
            view_counts.index_add_(0, torch.where(mask_2d)[0], torch.ones_like(view_counts[0:1]).expand(mask_2d.sum()))

        M = view_counts.max().item()
        text_features, logit_scale = None, None

        print(f"Procressing {len(imgs)} views")
        for view_idx in range(len(imgs)):
            img = imgs[view_idx].unsqueeze(0).permute(0, 3, 1, 2).contiguous().to(self.device)
            mask_this_view = (ori_coords_3ds[:, 0] == view_idx)
            all_mask_this_view = (mask_2ds[:, 0] == view_idx)
            mask_2d = mask_2ds[all_mask_this_view][:, 1].bool()

            label_2d = label_2ds[view_idx].unsqueeze(0)

            batch_input = {}
            batch_input["img"] = img
            batch_input["x_label"] = x_labels[mask_this_view]
            batch_input["y_label"] = y_labels[mask_this_view]
            batch_input["label_2d"] = label_2ds[view_idx]
            batch_input["inds_reconstruct"] = inds_reconstructs[mask_this_view]
            batch_input["unique_map"] = unique_maps[all_mask_this_view]
            batch_input["mapping"] = mappings[mask_this_view]
            ori_coords_3d = ori_coords_3ds[mask_this_view]
            ori_coords_3d[:, 0] *= 0
            batch_input["ori_coords"] = ori_coords_3d

            batch_input["use_pure_3d"] = False

            B = batch_input["img"].shape[0]
            xdecoder_batch_inputs = [{'image': batch_input["img"], 'height': self.cfg.mask_shape[0], 'width': self.cfg.mask_shape[1]}]
            _, outputs = self.xdecoder_teacher.model.forward_seg_all(xdecoder_batch_inputs)

            output_2d = []
            scores_keep_thresh = 0.0

            num_classes = outputs["pred_logits"].shape[-1] - 1

            mask_pred_results = outputs["pred_masks"]
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=tuple(self.cfg.mask_shape),
                mode="bicubic",
                align_corners=False,
                antialias=True
            )

            scene_idx = 0
            mask_pred_result = mask_pred_results[scene_idx]
            mask_cls_results = outputs["pred_logits"][scene_idx]
            pred_embeddings = outputs["mask_embed"][scene_idx]
            ori_coords = batch_input["ori_coords"]
            x_label = batch_input["x_label"][ori_coords[:, 0] == scene_idx]
            y_label = batch_input["y_label"][ori_coords[:, 0] == scene_idx]
            points_in_scene = ori_coords[ori_coords[:, 0] == scene_idx]

            scores, labels = F.softmax(mask_cls_results, dim=-1)[..., :-1].max(-1)

            keep = scores > scores_keep_thresh
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred_result[keep].sigmoid()
            cur_embedding = pred_embeddings[keep]

            if cur_masks.shape[0] == 0:
                num_points = points_in_scene.shape[0]
                feat_dim = pred_embeddings.shape[-1]
                single_2d_feature = torch.zeros((num_points, feat_dim), device=pred_embeddings.device)
            else:
                cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
                cur_mask_ids = cur_prob_masks.argmax(0)

                final_keep_indices = []
                final_masks = []
                for k in range(cur_classes.shape[0]):

                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        final_keep_indices.append(k)
                        final_masks.append(mask)

                if not final_keep_indices:
                    num_points = points_in_scene.shape[0]
                    feat_dim = pred_embeddings.shape[-1]
                    single_2d_feature = torch.zeros((num_points, feat_dim), device=pred_embeddings.device)
                else:
                    final_embedding = cur_embedding[final_keep_indices]
                    final_mask_stack = torch.stack(final_masks, dim=0)
                    num_points = points_in_scene.shape[0]
                    feat_dim = final_embedding.shape[-1]
                    mask_3d_feature = torch.zeros((num_points, feat_dim), device=final_embedding.device)
                    counter2d = torch.zeros((num_points, 1), device=final_embedding.device)

                    mask_3d = final_mask_stack[:, x_label, y_label]
                    mask_3d = mask_3d >= 0.5

                    for single_mask_3d, mask_emb in zip(mask_3d, final_embedding):
                        mask_3d_feature[single_mask_3d] += mask_emb
                        counter2d[single_mask_3d] += 1

                    counter2d[counter2d == 0] = 1e-5
                    single_2d_feature = mask_3d_feature / counter2d
                    del mask_3d_feature, counter2d, mask_3d

            output_2d.append(single_2d_feature)
            outputs["2d_pred_feature"] = output_2d
            text_features = outputs["text_embed"]
            logit_scale = outputs["logit_scale"]
            feature_2d = outputs["2d_pred_feature"]
            del outputs

            mask = batch_input["inds_reconstruct"]

            for feature_2d_single in feature_2d:
                if (
                    len(feature_2d_single[torch.sum(feature_2d_single, dim=1) == 0])
                    == 0
                ):
                    continue
                coord_3d = ori_coords_3d[:, 1:].clone()

                coord_3d = coord_3d.to(feature_2d_single.device)

                scene_true = coord_3d[torch.sum(feature_2d_single, dim=1) != 0]
                scene_false = coord_3d[torch.sum(feature_2d_single, dim=1) == 0]
                flase_idx = torch.where(torch.sum(feature_2d_single, dim=1) == 0)[0]
                true_idx = torch.where(torch.sum(feature_2d_single, dim=1) != 0)[0]

                kdtree = KDTree(scene_true.cpu())

                distances, indices = kdtree.query(scene_false.cpu(), k=1)

                match = true_idx[indices.flatten()]

                feature_2d_single[flase_idx] = feature_2d_single[match]

            feature_2d = torch.cat(feature_2d)
            feature_2d = F.normalize(feature_2d, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            logits_pred_2d = logit_scale * (feature_2d @ text_features.t())
            point_indices_in_this_view = torch.where(mask_2d)[0]

            for i, global_idx in enumerate(point_indices_in_this_view):

                point_info[global_idx.item()].append({
                    "feature": feature_2d[i],
                    "logits": logits_pred_2d[i],
                })
            counter[mask_2d] += 1
            del feature_2d
            del logits_pred_2d
            del single_2d_feature
            del img

            torch.cuda.empty_cache()

        num_points = scene_coords.shape[0]
        num_classes = len(self.cfg.all_label)

        final_scene_features = torch.zeros((num_points, feature_dim), dtype=torch.float32, device=self.device)

        points_with_views_indices = torch.tensor(list(point_info.keys()), device=self.device)
        num_points_with_views = len(points_with_views_indices)

        chunk_size = 50000
        for i in range(0, num_points_with_views, chunk_size):
            chunk_indices = points_with_views_indices[i:i+chunk_size]
            current_chunk_size = len(chunk_indices)
            M_chunk = max(len(point_info[idx.item()]) for idx in chunk_indices)
            padded_features_chunk = torch.zeros(current_chunk_size, M_chunk, feature_dim, device=self.device)
            padded_logits_chunk = torch.zeros(current_chunk_size, M_chunk, num_classes, device=self.device)
            validity_mask_chunk = torch.zeros(current_chunk_size, M_chunk, dtype=torch.bool, device=self.device)

            for j, global_idx in enumerate(chunk_indices):
                views_data = point_info[global_idx.item()]
                num_views_for_point = len(views_data)
                validity_mask_chunk[j, :num_views_for_point] = True
                for k, view_data in enumerate(views_data):
                    padded_features_chunk[j, k] = view_data["feature"]
                    padded_logits_chunk[j, k] = view_data["logits"]

            K_chunk = min(M_chunk, 3)
            sum_logits_chunk = padded_logits_chunk.sum(dim=1)
            num_valid_views_chunk = validity_mask_chunk.sum(dim=1, keepdim=True).clamp(min=1)
            avg_logits_chunk = sum_logits_chunk / num_valid_views_chunk
            consensus_idx_chunk = torch.argmax(avg_logits_chunk, dim=1)
            agreement_scores_chunk = torch.gather(padded_logits_chunk, 2, consensus_idx_chunk.view(-1, 1, 1).expand(-1, M_chunk, -1)).squeeze(-1)
            agreement_scores_chunk.masked_fill_(~validity_mask_chunk, -torch.inf)
            top_k_scores_chunk, top_k_indices_chunk = torch.topk(agreement_scores_chunk, k=K_chunk, dim=1)

            top_k_features_chunk = torch.gather(padded_features_chunk, 1, top_k_indices_chunk.unsqueeze(-1).expand(-1, -1, feature_dim))
            fusion_weights_chunk = F.softmax(top_k_scores_chunk, dim=1)
            fused_features_chunk = (top_k_features_chunk * fusion_weights_chunk.unsqueeze(-1)).sum(dim=1)

            final_scene_features[chunk_indices] = fused_features_chunk

        scene_coords = scene_coords.to(counter.device)
        scene_true = scene_coords[counter != 0]
        scene_false = scene_coords[counter == 0]
        if scene_true.shape[0] > 0 and scene_false.shape[0] > 0:
            flase_idx = torch.where(counter == 0)[0]
            true_idx = torch.where(counter != 0)[0]
            kdtree = KDTree(scene_true.cpu().numpy())
            distances, indices = kdtree.query(scene_false.cpu().numpy(), k=1)
            match = true_idx[indices.flatten()]
            final_scene_features[flase_idx] = final_scene_features[match]
        elif scene_false.shape[0] > 0:
            print(f"Warning: No points were seen for this scene. KDTree fill-in skipped. All points are treated as unseen.")
        # # PCA
        # pca_color = get_pca_color(final_scene_features, brightness=1.2, center=True)
        # # inverse back to original scale before grid sampling
        # # point.inverse is acquired from the GirdSampling transform
        # # pca_color_for_coords_3d = pca_color[point.inverse]
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(scene_coords)
        # pcd.colors = o3d.utility.Vector3dVector(pca_color.cpu().detach().numpy())
        # o3d.io.write_point_cloud("xdecoder_test/debug_projection_vis/2dpca.ply", pcd)
        # import pdb;pdb.set_trace()

        return (
            final_scene_features,
            text_features,
            logit_scale,
        )

    def convert_processed_img_to_detectron2(self, processed_img):
        img_chw = processed_img.squeeze(0)

        if hasattr(img_chw, 'cpu'):
            img_np = img_chw.cpu().numpy()
        else:
            img_np = img_chw.numpy()

        img_hwc = np.transpose(img_np, (1, 2, 0))
        if img_hwc.dtype != np.uint8:
            if img_hwc.max() <= 1.0:

                img_hwc = (img_hwc * 255.0).astype(np.uint8)
            else:

                img_hwc = img_hwc.astype(np.uint8)

        img_bgr = img_hwc[..., ::-1]
        return img_bgr

    def lift_ape_features(self, batch_data):
        (
            scene_coords,
            scene_coords_3d,
            scene_inds_reconstruct,
            scene_label,
            ori_coords_3ds,
            coords_3ds,
            feat_3ds,
            gauss_featuress,
            labels_3ds,
            binary_label_3ds,
            label_2ds,
            imgs,
            x_labels,
            y_labels,
            mask_2ds,
            inds_reconstructs,
            unique_maps,
            mappings,
            captions,
            scene_gauss_features
        ) = batch_data

        point_info = defaultdict(list)
        num_points = scene_coords.shape[0]
        num_classes = len(self.cfg.all_label)
        feature_dim = 256
        counter = torch.zeros((scene_coords.shape[0]), dtype=scene_label.dtype)
        view_counts = torch.zeros(num_points, dtype=torch.long, device=self.device)

        for view_idx in range(len(imgs)):
            all_mask_this_view = (mask_2ds[:, 0] == view_idx)
            mask_2d = mask_2ds[all_mask_this_view][:, 1].bool().to(view_counts.device)
            view_counts.index_add_(0, torch.where(mask_2d)[0], torch.ones_like(view_counts[0:1]).expand(mask_2d.sum()))

        M = view_counts.max().item()

        text_features, logit_scale = None, None

        print(f"Procressing {len(imgs)} views")
        for view_idx in range(len(imgs)):
            img = imgs[view_idx].unsqueeze(0).permute(0, 3, 1, 2).contiguous().to(self.device)
            mask_this_view = (ori_coords_3ds[:, 0] == view_idx)
            all_mask_this_view = (mask_2ds[:, 0] == view_idx)
            mask_2d = mask_2ds[all_mask_this_view][:, 1].bool()

            label_2d = label_2ds[view_idx].unsqueeze(0)

            batch_input = {}
            batch_input["img"] = img
            batch_input["x_label"] = x_labels[mask_this_view]
            batch_input["y_label"] = y_labels[mask_this_view]
            batch_input["label_2d"] = label_2ds[view_idx]
            batch_input["inds_reconstruct"] = inds_reconstructs[mask_this_view]
            batch_input["unique_map"] = unique_maps[all_mask_this_view]
            batch_input["mapping"] = mappings[mask_this_view]
            ori_coords_3d = ori_coords_3ds[mask_this_view]
            ori_coords_3d[:, 0] *= 0
            batch_input["ori_coords"] = ori_coords_3d

            batch_input["use_pure_3d"] = False

            B = batch_input["img"].shape[0]
            xdecoder_batch_inputs = [{'image': batch_input["img"], 'height': self.cfg.mask_shape[0], 'width': self.cfg.mask_shape[1]}]

            img = self.convert_processed_img_to_detectron2(batch_input["img"])
            outputs, _, _, _ = self.demo.run_on_image(
                img,
                text_prompt=self.text_prompt,
                with_box=self.with_box,
                with_mask=self.with_mask,
                with_sseg=self.with_sseg,
            )

            output_2d = []
            scores_keep_thresh = 0.0

            mask_pred_results = outputs['instances'].pred_masks
            mask_pred_results = F.interpolate(
                mask_pred_results.unsqueeze(0),
                size=tuple(self.cfg.mask_shape),
                mode="bicubic",
                align_corners=False,
                antialias=True
            )

            scene_idx = 0
            mask_pred_result = mask_pred_results[scene_idx]
            pred_embeddings = outputs['scene_features_for_instances']

            ori_coords = batch_input["ori_coords"]
            x_label = batch_input["x_label"][ori_coords[:, 0] == scene_idx]
            y_label = batch_input["y_label"][ori_coords[:, 0] == scene_idx]
            points_in_scene = ori_coords[ori_coords[:, 0] == scene_idx]

            scores = outputs['instances'].scores
            labels = outputs['instances'].pred_classes
            keep = scores > scores_keep_thresh
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred_result[keep].sigmoid()
            cur_embedding = pred_embeddings[keep]

            if cur_masks.shape[0] == 0:
                num_points = points_in_scene.shape[0]
                feat_dim = pred_embeddings.shape[-1]
                single_2d_feature = torch.zeros((num_points, feat_dim), device=pred_embeddings.device)
            else:
                cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
                cur_mask_ids = cur_prob_masks.argmax(0)

                final_keep_indices = []
                final_masks = []
                for k in range(cur_classes.shape[0]):

                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        final_keep_indices.append(k)
                        final_masks.append(mask)

                if not final_keep_indices:
                    num_points = points_in_scene.shape[0]
                    feat_dim = pred_embeddings.shape[-1]
                    single_2d_feature = torch.zeros((num_points, feat_dim), device=pred_embeddings.device)
                else:

                    final_embedding = cur_embedding[final_keep_indices]
                    final_mask_stack = torch.stack(final_masks, dim=0)
                    num_points = points_in_scene.shape[0]
                    feat_dim = final_embedding.shape[-1]
                    mask_3d_feature = torch.zeros((num_points, feat_dim), device=final_embedding.device)
                    counter2d = torch.zeros((num_points, 1), device=final_embedding.device)
                    mask_3d = final_mask_stack[:, x_label, y_label]
                    mask_3d = mask_3d >= 0.5

                    for single_mask_3d, mask_emb in zip(mask_3d, final_embedding):
                        mask_3d_feature[single_mask_3d] += mask_emb
                        counter2d[single_mask_3d] += 1


                    counter2d[counter2d == 0] = 1e-5
                    single_2d_feature = mask_3d_feature / counter2d
                    del mask_3d_feature, counter2d, mask_3d

            output_2d.append(single_2d_feature)
            outputs["2d_pred_feature"] = output_2d

            text_features = outputs["text_features"][:-1]
            logit_scale = outputs["logit_scale"]
            feature_2d = outputs["2d_pred_feature"]
            del outputs
            mask = batch_input["inds_reconstruct"]

            for feature_2d_single in feature_2d:
                if (
                    len(feature_2d_single[torch.sum(feature_2d_single, dim=1) == 0])
                    == 0
                ):
                    continue
                coord_3d = ori_coords_3d[:, 1:].clone()
                coord_3d = coord_3d.to(feature_2d_single.device)

                scene_true = coord_3d[torch.sum(feature_2d_single, dim=1) != 0]
                scene_false = coord_3d[torch.sum(feature_2d_single, dim=1) == 0]
                flase_idx = torch.where(torch.sum(feature_2d_single, dim=1) == 0)[0]
                true_idx = torch.where(torch.sum(feature_2d_single, dim=1) != 0)[0]

                kdtree = KDTree(scene_true.cpu())
                distances, indices = kdtree.query(scene_false.cpu(), k=1)
                match = true_idx[indices.flatten()]
                feature_2d_single[flase_idx] = feature_2d_single[match]

            feature_2d = torch.cat(feature_2d)
            feature_2d = F.normalize(feature_2d, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            logits_pred_2d = logit_scale * (feature_2d @ text_features.t())

            point_indices_in_this_view = torch.where(mask_2d)[0]

            for i, global_idx in enumerate(point_indices_in_this_view):
                point_info[global_idx.item()].append({
                    "feature": feature_2d[i],
                    "logits": logits_pred_2d[i],
                })
            counter[mask_2d] += 1
            del feature_2d
            del logits_pred_2d
            del single_2d_feature
            del img
            torch.cuda.empty_cache()

        num_points = scene_coords.shape[0]
        num_classes = len(self.cfg.all_label)
        final_scene_features = torch.zeros((num_points, feature_dim), dtype=torch.float32, device=self.device)
        points_with_views_indices = torch.tensor(list(point_info.keys()), device=self.device)
        num_points_with_views = len(points_with_views_indices)

        chunk_size = 50000
        for i in range(0, num_points_with_views, chunk_size):
            chunk_indices = points_with_views_indices[i:i+chunk_size]
            current_chunk_size = len(chunk_indices)
            M_chunk = max(len(point_info[idx.item()]) for idx in chunk_indices)
            padded_features_chunk = torch.zeros(current_chunk_size, M_chunk, feature_dim, device=self.device)
            padded_logits_chunk = torch.zeros(current_chunk_size, M_chunk, num_classes, device=self.device)
            validity_mask_chunk = torch.zeros(current_chunk_size, M_chunk, dtype=torch.bool, device=self.device)

            for j, global_idx in enumerate(chunk_indices):
                views_data = point_info[global_idx.item()]
                num_views_for_point = len(views_data)
                validity_mask_chunk[j, :num_views_for_point] = True
                for k, view_data in enumerate(views_data):
                    padded_features_chunk[j, k] = view_data["feature"]
                    padded_logits_chunk[j, k] = view_data["logits"]

            K_chunk = min(M_chunk, 3)
            sum_logits_chunk = padded_logits_chunk.sum(dim=1)
            num_valid_views_chunk = validity_mask_chunk.sum(dim=1, keepdim=True).clamp(min=1)
            avg_logits_chunk = sum_logits_chunk / num_valid_views_chunk
            consensus_idx_chunk = torch.argmax(avg_logits_chunk, dim=1)
            agreement_scores_chunk = torch.gather(padded_logits_chunk, 2, consensus_idx_chunk.view(-1, 1, 1).expand(-1, M_chunk, -1)).squeeze(-1)
            agreement_scores_chunk.masked_fill_(~validity_mask_chunk, -torch.inf)
            top_k_scores_chunk, top_k_indices_chunk = torch.topk(agreement_scores_chunk, k=K_chunk, dim=1)

            top_k_features_chunk = torch.gather(padded_features_chunk, 1, top_k_indices_chunk.unsqueeze(-1).expand(-1, -1, feature_dim))
            fusion_weights_chunk = F.softmax(top_k_scores_chunk, dim=1)
            fused_features_chunk = (top_k_features_chunk * fusion_weights_chunk.unsqueeze(-1)).sum(dim=1)
            final_scene_features[chunk_indices] = fused_features_chunk

        scene_coords = scene_coords.to(counter.device)
        scene_true = scene_coords[counter != 0]
        scene_false = scene_coords[counter == 0]
        if scene_true.shape[0] > 0 and scene_false.shape[0] > 0:
            flase_idx = torch.where(counter == 0)[0]
            true_idx = torch.where(counter != 0)[0]

            kdtree = KDTree(scene_true.cpu().numpy())
            distances, indices = kdtree.query(scene_false.cpu().numpy(), k=1)
            match = true_idx[indices.flatten()]
            final_scene_features[flase_idx] = final_scene_features[match]

        elif scene_false.shape[0] > 0:
            print(f"Warning: No points were seen for this scene. KDTree fill-in skipped. All points are treated as unseen.")

        # pca_color = get_pca_color(final_scene_features, brightness=1.2, center=True)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(scene_coords)
        # pcd.colors = o3d.utility.Vector3dVector(pca_color.cpu().detach().numpy())
        # o3d.io.write_point_cloud("xdecoder_test/debug_projection_vis/2dpcaape.ply", pcd)

        return (
            final_scene_features,
            text_features,
            logit_scale,
        )

    def get_sonata_features(self, batch_data):
        (
            scene_coords,
            scene_coords_3d,
            scene_inds_reconstruct,
            scene_label,
            ori_coords_3ds,
            coords_3ds,
            feat_3ds,
            gauss_featuress,
            labels_3ds,
            binary_label_3ds,
            label_2ds,
            imgs,
            x_labels,
            y_labels,
            mask_2ds,
            inds_reconstructs,
            unique_maps,
            mappings,
            captions,
            scene_gauss_features
        ) = batch_data

        ori_coords_3d = scene_coords
        feats = scene_gauss_features[:, :3]
        normals = scene_gauss_features[:, 3:6]
        device = self.device

        transform = sonata.transform.default()
        point = {
            "coord": ori_coords_3d.cpu().numpy().astype(np.float32),
            "color": feats.cpu().numpy().astype(np.float32),
            "normal": normals.cpu().numpy().astype(np.float32),
        }
        point = transform(point)

        with torch.inference_mode():
            for key in point.keys():
                if isinstance(point[key], torch.Tensor):
                    point[key] = point[key].cuda(non_blocking=True)

            point = self.sonata_teacher(point)
            for _ in range(2):
                assert "pooling_parent" in point.keys()
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = point.feat[inverse]
                point = parent

            F_sonata = point.feat[point.inverse]
        # # PCA
        # pca_color = get_pca_color(F_sonata, brightness=1.2, center=True)
        # # inverse back to original scale before grid sampling
        # # point.inverse is acquired from the GirdSampling transform
        # # pca_color_for_coords_3d = pca_color[point.inverse]
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(scene_coords)
        # pcd.colors = o3d.utility.Vector3dVector(pca_color.cpu().detach().numpy())
        # o3d.io.write_point_cloud("xdecoder_test/debug_projection_vis/3dpca.ply", pcd)
        # import pdb;pdb.set_trace()
        return F_sonata

    @torch.no_grad()
    def sample_contrastive_pairs(self, F_sonata_full):
        device = F_sonata_full.device
        num_points = F_sonata_full.shape[0]

        num_anchors = min(self.num_anchors_per_scene, num_points // 3)
        if num_anchors == 0:
            return torch.empty(0, device=device, dtype=torch.long), \
                torch.empty(0, device=device, dtype=torch.long), \
                torch.empty(0, device=device, dtype=torch.long)

        anchor_indices = torch.randperm(num_points, device=device)[:num_anchors]
        F_anchor = F_sonata_full[anchor_indices]

        sim_matrix = torch.einsum('ad,pd->ap', F.normalize(F_anchor, p=2, dim=1),
                                            F.normalize(F_sonata_full, p=2, dim=1))

        sim_matrix.scatter_(1, anchor_indices.unsqueeze(1), float('-inf'))
        positive_indices = torch.argmax(sim_matrix, dim=1)
        all_indices = torch.arange(num_points, device=device)
        exclude_mask = (all_indices.unsqueeze(0) == anchor_indices.unsqueeze(1)) | \
                    (all_indices.unsqueeze(0) == positive_indices.unsqueeze(1))
        sim_matrix_for_neg = sim_matrix
        sim_matrix_for_neg[exclude_mask] = float('-inf')
        _, negative_indices = torch.topk(
            sim_matrix_for_neg,
            k=self.num_negatives_per_anchor,
            largest=False,
            dim=1
        )
        del sim_matrix_for_neg
        return anchor_indices, positive_indices, negative_indices

    @torch.no_grad()
    def sample_contrastive_pairs_hybrid(self, F_sonata_full, neighbor_indices):
        """
        Args:
            F_sonata_full (torch.Tensor): [N_points, D_feature], 场景中所有点的Sonata特征。
            neighbor_indices (torch.Tensor): [N_points, K], 每个点的K个空间最近邻的索引。
        """
        device = F_sonata_full.device
        num_points, K = neighbor_indices.shape

        num_anchors = min(self.num_anchors_per_scene, num_points // 3)
        if num_anchors == 0:
            return torch.empty(0, 3, device=device, dtype=torch.long).T

        anchor_indices = torch.randperm(num_points, device=device)[:num_anchors]
        F_anchor = F_sonata_full[anchor_indices]
        sim_matrix_global = torch.einsum('ad,pd->ap', F.normalize(F_anchor, p=2, dim=1),
                                        F.normalize(F_sonata_full, p=2, dim=1))

        sim_matrix_pos = sim_matrix_global.clone()
        sim_matrix_pos.scatter_(1, anchor_indices.unsqueeze(1), float('-inf'))
        positive_indices = torch.argmax(sim_matrix_pos, dim=1)
        del sim_matrix_pos

        N_macro = 48
        N_micro = self.num_negatives_per_anchor - N_macro
        sim_matrix_neg = sim_matrix_global
        exclude_mask = (torch.arange(num_points, device=device).unsqueeze(0) == anchor_indices.unsqueeze(1)) | \
                    (torch.arange(num_points, device=device).unsqueeze(0) == positive_indices.unsqueeze(1))
        sim_matrix_neg[exclude_mask] = float('inf')
        _, macro_negative_indices = torch.topk(sim_matrix_neg, k=N_macro, largest=False, dim=1)

        anchor_neighbors_indices = neighbor_indices[anchor_indices]
        sims_local = torch.gather(sim_matrix_global, 1, anchor_neighbors_indices)
        _, hardest_indices_in_K = torch.topk(sims_local, k=N_micro, largest=False, dim=1)
        micro_negative_indices = torch.gather(anchor_neighbors_indices, 1, hardest_indices_in_K)
        negative_indices = torch.cat([macro_negative_indices, micro_negative_indices], dim=1)

        return anchor_indices, positive_indices, negative_indices

    def forward(self, batch_data):
        with torch.no_grad():
            if self.use_lseg:
                try:
                    F_xdecoder_full, _, _ = self.lift_lseg_features(batch_data)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("OOM with batch_size=24, trying batch_size=10...")
                        torch.cuda.empty_cache()
                        F_xdecoder_full, _, _ = self.lift_lseg_features(batch_data, max_batch_size=10)
                    else:
                        raise e
            elif self.use_ape:
                F_xdecoder_full, text_features, logit_scale = self.lift_ape_features(batch_data)
            else:
                F_xdecoder_full, text_features, logit_scale = self.lift_xdecoder_features(batch_data)

            F_sonata_full = self.get_sonata_features(batch_data)
            F_xdecoder_full = F_xdecoder_full.to(F_sonata_full.device)

            (scene_coords, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _) = batch_data
            K = 96
            coords_np = scene_coords.cpu().numpy()
            index = faiss.IndexFlatL2(coords_np.shape[1])
            index.add(coords_np)

            _, neighbor_indices_with_self = index.search(coords_np, K + 1)
            neighbor_indices = torch.from_numpy(neighbor_indices_with_self[:, 1:]).to(self.device)
            anchor_idx, positive_idx, negative_idx = self.sample_contrastive_pairs_hybrid(F_sonata_full,neighbor_indices)
            del F_sonata_full

        (
            scene_coords,
            scene_coords_3d,
            scene_inds_reconstruct,
            scene_label,
            ori_coords_3ds,
            coords_3ds,
            feat_3ds,
            gauss_featuress,
            labels_3ds,
            binary_label_3ds,
            label_2ds,
            imgs,
            x_labels,
            y_labels,
            mask_2ds,
            inds_reconstructs,
            unique_maps,
            mappings,
            captions,
            scene_gauss_features
        ) = batch_data

        all_sample_indices, point_to_batch_map = torch.unique(
            torch.cat([anchor_idx, positive_idx, negative_idx.flatten()]),
            return_inverse=True
        )
        voxel_indices_for_samples = scene_inds_reconstruct.to(all_sample_indices.device)[all_sample_indices]
        unique_voxel_indices, sample_to_voxel_map = torch.unique(
            voxel_indices_for_samples, return_inverse=True
        )

        features_for_samples = F_xdecoder_full[all_sample_indices]

        voxel_features_input = torch_scatter.scatter_mean(
            features_for_samples, sample_to_voxel_map, dim=0
        )

        voxel_coords_input = scene_coords_3d.to(unique_voxel_indices.device)[unique_voxel_indices]
        s_input_student = ME.SparseTensor(
            features=voxel_features_input,
            coordinates=ME.utils.batched_coordinates([voxel_coords_input]),
            device=voxel_features_input.device
        )
        del voxel_features_input, voxel_coords_input
        s_output_student = self.affinity_student(s_input_student)

        F_pred_on_samples = s_output_student.F[sample_to_voxel_map]
        del s_output_student

        F_pred_on_samples_norm = F.normalize(F_pred_on_samples, p=2, dim=1)

        num_anchors = len(anchor_idx)
        F_pred_anchor = F_pred_on_samples_norm[point_to_batch_map[:num_anchors]]
        F_pred_positive = F_pred_on_samples_norm[point_to_batch_map[num_anchors:2*num_anchors]]
        F_pred_negative = F_pred_on_samples_norm[point_to_batch_map[2*num_anchors:]].reshape(num_anchors, self.num_negatives_per_anchor, -1)

        l_pos = torch.einsum('bd,bd->b', F_pred_anchor, F_pred_positive).unsqueeze(-1)
        l_neg = torch.einsum('bd,bnd->bn', F_pred_anchor, F_pred_negative)

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.info_nce_temperature

        labels = torch.zeros(num_anchors, dtype=torch.long, device=logits.device)
        loss = self.criterion(logits, labels)

        del F_xdecoder_full, s_input_student, F_pred_on_samples, F_pred_anchor, F_pred_positive, F_pred_negative, logits, labels
        torch.cuda.empty_cache()
        return loss

    def visualize_and_save_point_cloud(self, coords, features, filename, mode='pca', kmeans_k=20):
        import open3d as o3d
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        import os
        import numpy as np
        import matplotlib.cm as cm

        print(f"Visualizing with mode '{mode}' and saving to {filename}...")
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        if mode == 'pca':
            pca = PCA(n_components=3)
            features_3d = pca.fit_transform(features.astype(np.float64))

            min_vals = features_3d.min(axis=0)
            max_vals = features_3d.max(axis=0)

            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1.0
            colors = (features_3d - min_vals) / range_vals

        elif mode == 'kmeans':
            print(f"Running K-Means with k={kmeans_k}...")
            kmeans = KMeans(n_clusters=kmeans_k, random_state=0, n_init='auto')
            labels = kmeans.fit_predict(features.astype(np.float64))
            cmap = cm.get_cmap('tab20')
            colors = cmap(labels / (kmeans_k - 1))[:, :3] if kmeans_k > 1 else np.tile([0.5, 0.5, 0.5], (len(labels), 1))

        elif mode == 'rgb':
            if features.max() > 1.0:
                colors = features / 255.0
            else:
                colors = features
            assert colors.shape[1] == 3, "RGB模式要求特征维度为3"
        else:
            raise ValueError(f"未知的可视化模式: {mode}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        output_dir = os.path.dirname(filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        o3d.io.write_point_cloud(filename, pcd)
        print(f"Successfully saved point cloud to {filename}")

    def visualize_affinity_weights(self, voxel_coords, neighbor_indices, affinity_weights,
                              num_voxels, K, save_path):
        import open3d as o3d
        import matplotlib.pyplot as plt

        weights_reshaped = affinity_weights.view(num_voxels, K)
        entropy = -(weights_reshaped * torch.log(weights_reshaped + 1e-8)).sum(dim=1)
        max_entropy = torch.log(torch.tensor(K, dtype=torch.float32, device=entropy.device))
        normalized_entropy = entropy / max_entropy

        print("=== AFFINITY WEIGHTS DEBUG ===")
        print(f"K (neighbors per voxel): {K}")
        print(f"Max possible entropy: {max_entropy:.3f}")
        print(f"Raw entropy range: [{entropy.min():.3f}, {entropy.max():.3f}]")
        print(f"Normalized entropy range: [{normalized_entropy.min():.3f}, {normalized_entropy.max():.3f}]")

        print("\n=== WEIGHT STATISTICS ===")
        weights_mean = weights_reshaped.mean(dim=1)
        weights_std = weights_reshaped.std(dim=1)
        weights_max = weights_reshaped.max(dim=1)[0]
        weights_min = weights_reshaped.min(dim=1)[0]

        print(f"Expected uniform weight: {1.0/K:.4f}")
        print(f"Actual weight mean range: [{weights_mean.min():.4f}, {weights_mean.max():.4f}]")
        print(f"Weight std range: [{weights_std.min():.4f}, {weights_std.max():.4f}]")
        print(f"Weight max range: [{weights_max.min():.4f}, {weights_max.max():.4f}]")
        print(f"Weight min range: [{weights_min.min():.4f}, {weights_min.max():.4f}]")

        concentrated_mask = normalized_entropy < 0.5
        print(f"Concentrated voxels (entropy < 0.5): {concentrated_mask.sum().item()}/{num_voxels} ({concentrated_mask.float().mean()*100:.1f}%)")

        very_concentrated_mask = normalized_entropy < 0.3
        print(f"Very concentrated voxels (entropy < 0.3): {very_concentrated_mask.sum().item()}/{num_voxels} ({very_concentrated_mask.float().mean()*100:.1f}%)")

        entropy_percentiles = torch.quantile(normalized_entropy, torch.tensor([0.25, 0.75], device=normalized_entropy.device))
        entropy_stretched = torch.clamp(
            (normalized_entropy - entropy_percentiles[0]) / (entropy_percentiles[1] - entropy_percentiles[0]),
            0, 1
        )
        pcd_stretched = o3d.geometry.PointCloud()
        pcd_stretched.points = o3d.utility.Vector3dVector(voxel_coords)
        colors_stretched = plt.cm.plasma(1 - entropy_stretched.cpu().numpy())[:, :3]
        pcd_stretched.colors = o3d.utility.Vector3dVector(colors_stretched)
        o3d.io.write_point_cloud(f"{save_path}_affinity_concentration_stretched.ply", pcd_stretched)

        max_weights_norm = (weights_max - weights_max.min()) / (weights_max.max() - weights_max.min())

        pcd_max = o3d.geometry.PointCloud()
        pcd_max.points = o3d.utility.Vector3dVector(voxel_coords)
        colors_max = plt.cm.plasma(max_weights_norm.cpu().numpy())[:, :3]
        pcd_max.colors = o3d.utility.Vector3dVector(colors_max)
        o3d.io.write_point_cloud(f"{save_path}_affinity_max_weights.ply", pcd_max)

        print(f"\nAffinity visualization saved:")
        print(f"  - Percentile stretched: {save_path}_affinity_concentration_stretched.ply")
        print(f"  - Max weights: {save_path}_affinity_max_weights.ply")

    def plot_affinity_heatmap_3d(self,voxel_coords, affinity_values, num_voxels, K, output_path):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import open3d as o3d
        from matplotlib.colors import LinearSegmentedColormap
        import torch

        affinity_2d = affinity_values.view(num_voxels, K).cpu().numpy()
        voxel_coords_np = voxel_coords.copy() if isinstance(voxel_coords, np.ndarray) else voxel_coords
        avg_affinity = np.mean(affinity_2d, axis=1)
        max_affinity = np.max(affinity_2d, axis=1)
        std_affinity = np.std(affinity_2d, axis=1)

        def normalize_for_color(values):
            return (values - values.min()) / (values.max() - values.min() + 1e-8)

        avg_affinity_norm = normalize_for_color(avg_affinity)
        max_affinity_norm = normalize_for_color(max_affinity)
        std_affinity_norm = normalize_for_color(std_affinity)

        colormap_options = {
            'avg': ('viridis', avg_affinity_norm, 'Average Affinity'),
            'max': ('plasma', max_affinity_norm, 'Max Affinity'),
            'diversity': ('coolwarm', std_affinity_norm, 'Affinity Diversity (Std)')
        }

        for color_type, (cmap_name, values, title) in colormap_options.items():
            cmap = plt.cm.get_cmap(cmap_name)
            colors = cmap(values)[:, :3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(voxel_coords_np)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(f"{output_path}_3d_{color_type}_affinity.ply", pcd)


        sample_size = min(200, num_voxels)
        sample_indices = np.random.choice(num_voxels, sample_size, replace=False)
        sample_affinity = affinity_2d[sample_indices, :]

        plt.figure(figsize=(15, 8))
        sns.heatmap(sample_affinity,
                    cmap='viridis',
                    cbar_kws={'label': 'Cosine Similarity'})
        plt.title(f'Affinity Matrix (Random {sample_size} Voxels vs K={K} Neighbors)')
        plt.xlabel('Neighbor Index')
        plt.ylabel('Voxel Index')
        plt.savefig(f"{output_path}_2d_affinity_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

        fig = plt.figure(figsize=(20, 5))
        ax1 = fig.add_subplot(141)
        ax1.hist(avg_affinity, bins=50, alpha=0.7, color='blue')
        ax1.set_title('Average Affinity Distribution')
        ax1.set_xlabel('Average Affinity')
        ax1.set_ylabel('Frequency')

        ax2 = fig.add_subplot(142)
        ax2.hist(max_affinity, bins=50, alpha=0.7, color='red')
        ax2.set_title('Max Affinity Distribution')
        ax2.set_xlabel('Max Affinity')
        ax2.set_ylabel('Frequency')

        ax3 = fig.add_subplot(143)
        ax3.hist(std_affinity, bins=50, alpha=0.7, color='green')
        ax3.set_title('Affinity Diversity (Std) Distribution')
        ax3.set_xlabel('Standard Deviation')
        ax3.set_ylabel('Frequency')

        ax4 = fig.add_subplot(144)
        scatter = ax4.scatter(voxel_coords_np[:, 2], avg_affinity,
                            c=avg_affinity, cmap='viridis', alpha=0.6, s=1)
        ax4.set_xlabel('Z Coordinate')
        ax4.set_ylabel('Average Affinity')
        ax4.set_title('Affinity vs Height')
        plt.colorbar(scatter, ax=ax4, label='Average Affinity')

        plt.tight_layout()
        plt.savefig(f"{output_path}_affinity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        high_affinity_idx = np.argsort(avg_affinity)[-5:]
        low_affinity_idx = np.argsort(avg_affinity)[:5]
        med_affinity_idx = np.argsort(np.abs(avg_affinity - np.median(avg_affinity)))[:5]

        representative_indices = np.concatenate([high_affinity_idx, med_affinity_idx, low_affinity_idx])

        print(f"=== Affinity Statistics ===")
        print(f"Average affinity - Mean: {avg_affinity.mean():.4f}, Std: {avg_affinity.std():.4f}")
        print(f"Max affinity - Mean: {max_affinity.mean():.4f}, Std: {max_affinity.std():.4f}")
        print(f"Affinity diversity - Mean: {std_affinity.mean():.4f}, Std: {std_affinity.std():.4f}")
        print(f"Saved 3D visualizations: {output_path}_3d_[avg|max|diversity]_affinity.ply")
        print(f"Saved 2D heatmap: {output_path}_2d_affinity_heatmap.png")
        print(f"Saved analysis plots: {output_path}_affinity_analysis.png")

        return {
            'avg_affinity': avg_affinity,
            'max_affinity': max_affinity,
            'std_affinity': std_affinity,
            'representative_indices': representative_indices
        }

    def visualize_local_neighborhoods(self,voxel_coords, neighbor_indices, affinity_values,
                                    representative_indices, num_voxels, K, output_path):

        import open3d as o3d
        import numpy as np
        import matplotlib.pyplot as plt

        affinity_2d = affinity_values.view(num_voxels, K).cpu().numpy()

        for i, voxel_idx in enumerate(representative_indices[:6]):

            center_coord = voxel_coords[voxel_idx]
            neighbors_idx = neighbor_indices[voxel_idx].cpu().numpy()
            neighbor_coords = voxel_coords[neighbors_idx]
            neighbor_affinities = affinity_2d[voxel_idx]
            pcd = o3d.geometry.PointCloud()
            all_coords = np.vstack([center_coord.reshape(1, -1), neighbor_coords])
            pcd.points = o3d.utility.Vector3dVector(all_coords)
            colors = np.zeros((len(all_coords), 3))
            colors[0] = [1, 0, 0]
            cmap = plt.cm.get_cmap('RdYlGn')
            normalized_affinities = (neighbor_affinities - neighbor_affinities.min()) / \
                                (neighbor_affinities.max() - neighbor_affinities.min() + 1e-8)
            colors[1:] = cmap(normalized_affinities)[:, :3]

            pcd.colors = o3d.utility.Vector3dVector(colors)
            lines = [[0, j+1] for j in range(len(neighbors_idx))]
            line_set = o3d.geometry.LineSet()
            line_set.points = pcd.points
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_colors = cmap(normalized_affinities)[:, :3]
            line_set.colors = o3d.utility.Vector3dVector(line_colors)


            o3d.io.write_point_cloud(f"{output_path}_neighborhood_{i}_points.ply", pcd)
            o3d.io.write_line_set(f"{output_path}_neighborhood_{i}_lines.ply", line_set)

            print(f"Voxel {voxel_idx}: avg_affinity={affinity_2d[voxel_idx].mean():.4f}, "
                f"max_affinity={affinity_2d[voxel_idx].max():.4f}")

    @torch.no_grad()
    def evaluate_scene(self, batch_data,vis_prefix = "scene0695_00"):
        K = 96
        affinity_sharpen_factor = 20
        self.affinity_student.eval()
        if self.use_lseg:
            F_xdecoder_full,text_features,logit_scale = self.lift_lseg_features(batch_data)
        elif self.use_ape:
            F_xdecoder_full, text_features, logit_scale = self.lift_ape_features(batch_data)
        else:
            F_xdecoder_full,text_features,logit_scale = self.lift_xdecoder_features(batch_data)
        (
            scene_coords,
            scene_coords_3d,
            scene_inds_reconstruct,
            scene_label,
            ori_coords_3ds,
            coords_3ds,
            feat_3ds,
            gauss_featuress,
            labels_3ds,
            binary_label_3ds,
            label_2ds,
            imgs,
            x_labels,
            y_labels,
            mask_2ds,
            inds_reconstructs,
            unique_maps,
            mappings,
            captions,
            scene_gauss_features
        ) = batch_data

        voxel_semantic_features = torch_scatter.scatter_mean(
            F_xdecoder_full.to(scene_inds_reconstruct.device),
            scene_inds_reconstruct,
            dim=0
        )

        raw_geom_features_full = scene_gauss_features[:, :6].float().to(F_xdecoder_full.device)
        voxel_geom_features = torch_scatter.scatter_mean(
            raw_geom_features_full.to(scene_inds_reconstruct.device),
            scene_inds_reconstruct,
            dim=0
        )
        voxel_features_input = torch.cat([voxel_semantic_features, voxel_geom_features], dim=1).to(self.device)

        del F_xdecoder_full, voxel_semantic_features, voxel_geom_features, raw_geom_features_full
        torch.cuda.empty_cache()

        s_input_student = ME.SparseTensor(
            features=voxel_features_input,
            coordinates=ME.utils.batched_coordinates([scene_coords_3d.to(self.device)]),
            device=voxel_features_input.device
        )
        s_output_student = self.affinity_student(s_input_student)
        F_pred_embed = F.normalize(s_output_student.F, p=2, dim=1)

        del s_output_student

        voxel_coords = s_input_student.C[:, 1:].contiguous().float().cpu().numpy()
        num_voxels = voxel_coords.shape[0]
        index = faiss.IndexFlatL2(voxel_coords.shape[1])
        index.add(voxel_coords)

        _, neighbor_indices = index.search(voxel_coords, K + 1)
        neighbor_indices = torch.from_numpy(neighbor_indices[:, 1:]).to(F_pred_embed.device)

        row_indices = torch.arange(num_voxels, device=F_pred_embed.device).repeat_interleave(K)
        col_indices = neighbor_indices.flatten()

        del index

        center_embeds = F_pred_embed.repeat_interleave(K, dim=0)
        neighbor_embeds = F_pred_embed[col_indices]

        affinity_values = torch.einsum('bd,bd->b', center_embeds, neighbor_embeds)
        del F_pred_embed, center_embeds, neighbor_embeds

        affinity_weights = F.softmax(
            affinity_values.view(num_voxels, K) * affinity_sharpen_factor, dim=1
        ).flatten()
        del affinity_values

        A_sparse_voxel_norm = torch.sparse_coo_tensor(
            indices=torch.stack([row_indices, col_indices]),
            values=affinity_weights,
            size=(num_voxels, num_voxels)
        )

        del affinity_weights, row_indices, col_indices


        F_refined_voxel = torch.sparse.mm(A_sparse_voxel_norm, voxel_features_input)

        for _ in range(18):
            F_refined_voxel = torch.sparse.mm(A_sparse_voxel_norm, F_refined_voxel)

        F_refined_points = F_refined_voxel[scene_inds_reconstruct][:, :512]
        del voxel_features_input, s_input_student, A_sparse_voxel_norm, F_refined_voxel
        torch.cuda.empty_cache()

        # # PCA
        # pca_color = get_pca_color(F_refined_points, brightness=1.2, center=True)
        # # inverse back to original scale before grid sampling
        # # point.inverse is acquired from the GirdSampling transform
        # # pca_color_for_coords_3d = pca_color[point.inverse]
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(scene_coords)
        # pcd.colors = o3d.utility.Vector3dVector(pca_color.cpu().detach().numpy())
        # o3d.io.write_point_cloud("xdecoder_test/debug_projection_vis/allpca.ply", pcd)
        # import pdb;pdb.set_trace()

        return {
            "scene_features": F_refined_points.to(text_features.device), # [N_total_points, D_feature]
            "text_features": text_features,   # [N_classes, D_feature]
            "logit_scale": logit_scale        
        }