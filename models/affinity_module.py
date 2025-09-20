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

# 先定义一个残差块
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
        # 加上残差连接
        out += identity
        return MEF.relu(out)

class AffinityPredictor(nn.Module):
    def __init__(self, input_dim, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. 输入层：将原始特征维度映射到隐藏维度
        self.input_layer = nn.Sequential(
            ME.MinkowskiConvolution(input_dim, hidden_dim, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(hidden_dim),
            ME.MinkowskiReLU(),
        )
        
        # 2. 中间层：堆叠多个残差块
        self.res_blocks = nn.Sequential(
            MinkowskiResBlock(hidden_dim),
            MinkowskiResBlock(hidden_dim),
            MinkowskiResBlock(hidden_dim),
            MinkowskiResBlock(hidden_dim), # 可以根据需要堆叠更多
        )
        
        # 3. 输出层：将隐藏特征映射到最终的嵌入维度
        self.output_layer = ME.MinkowskiConvolution(hidden_dim, embed_dim, kernel_size=1, dimension=3)
        
    def forward(self, x_s: ME.SparseTensor) -> ME.SparseTensor:
        out = self.input_layer(x_s)
        out = self.res_blocks(out)
        out = self.output_layer(out)
        return out
    
    def get_param_groups(self):
        """
        将模型的参数分为三组，以便应用差分学习率。
        - 'input': 适配器层，学习率应较小。
        - 'middle': 核心处理层，使用基准学习率。
        - 'output': 投影头，学习率可较大。
        """
        return {
            "input": list(self.input_layer.parameters()),
            "middle": list(self.res_blocks.parameters()),
            "output": list(self.output_layer.parameters())
        }

from models.modeling.meta_arch.mink_unet import mink_unet
class MinkUNetWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, D=3, arch='MinkUNet18A'):
        super().__init__()
        self.mink_unet = mink_unet(in_channels, out_channels, D, arch)
    
    def forward(self, x_s):
        temp_out, final_out = self.mink_unet(x_s)
        return final_out
    
    def get_param_groups(self):
        """
        基于MinkUNet架构特点进行参数分组，支持差分学习率：
        - input: 输入适配层，需要小心调整以适应新的输入维度
        - middle: 主干网络，包含编码器-解码器的核心特征提取
        - output: 输出投影层，通常需要更积极的学习
        """
        input_params = []
        middle_params = []
        output_params = []
        
        for name, param in self.mink_unet.named_parameters():
            if self._is_input_layer(name):
                input_params.append(param)
            elif self._is_output_layer(name):
                output_params.append(param)
            else:
                middle_params.append(param)
        
        return {
            "input": input_params,
            "middle": middle_params, 
            "output": output_params
        }
    
    def _is_input_layer(self, layer_name):
        """判断是否为输入适配层"""
        # 输入相关层：需要适配新的输入维度 (518维 -> 32维)
        input_patterns = [
            'conv0p1s1',  # 第一个卷积层：518 -> 32
            'bn0',        # 对应的BN层
        ]
        return any(pattern in layer_name for pattern in input_patterns)
    
    def _is_output_layer(self, layer_name):
        """判断是否为输出投影层"""
        # 输出相关层：将UNet特征映射到目标维度
        output_patterns = [
            'final',      # 最终投影层：96 -> embed_dim
        ]
        return any(pattern in layer_name for pattern in output_patterns)

class SonataXAffinityTrainer(nn.Module):
    """
    训练方案2的“总教练”模块。
    它封装了所有逻辑，使得主训练脚本非常简洁。
    """
    def __init__(self, cfg, xdecoder_cfg, scene_config, device='cuda', use_lseg=True):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.use_lseg = use_lseg  # 新增标志位
        self.scene_config = scene_config
        self.use_ape = cfg.use_ape

        if self.use_lseg:
            # 初始化LSeg模型（参考第二段代码）
            from models.Lseg.lseg_module import LSegModule
            from models.Lseg.lseg_multievalmodule import LSeg_MultiEvalModule
            import torchvision.transforms as transforms
            from encoding.models.sseg import BaseNet
            
            # 加载LSeg模型
            module = LSegModule.load_from_checkpoint(
                checkpoint_path=cfg.lseg_model_path,  # 需要在cfg中配置
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
            
            # model
            if isinstance(module.net, BaseNet):
                model = module.net
            else:
                model = module
            model = model.eval().cuda()
            model.mean = [0.5, 0.5, 0.5]
            model.std = [0.5, 0.5, 0.5]
            
            # # LSeg特殊配置
            model.crop_size = 640
            model.base_size = 640
            
            self.lseg_evaluator = LSeg_MultiEvalModule(
                model, scales=([1]), flip=True
            ).cuda()
            self.lseg_evaluator.eval()
            
            # 冻结LSeg参数
            for param in self.lseg_evaluator.parameters():
                param.requires_grad = False
                
            # LSeg的图像预处理
            self.lseg_normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

            self.text_model = module.net.clip_pretrained.to(torch.float32).cuda()
                
            # # # LSeg的图像预处理
            # self.lseg_normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

            # from models.gaussians.model.lseg_predictor import LSeg

            # self.lseg_evaluator = LSeg(weight_path="./pretrained/weights/lseg/demo_e200.ckpt",device=self.device)
            # # self.lseg_evaluator.eval()

            # self.text_model = self.lseg_evaluator.text_model

            # a bit of prompt engineering
            if hasattr(cfg, 'prompt_eng') and cfg.prompt_eng:
                print('Use prompt engineering: a XX in a scene')
                labelset = [ "a " + label + " in a scene" for label in cfg.all_label]
                labelset.append('background and other objects')
            print(f"Final labelset: {labelset}")  # 调试用，确认类别列表
            self.text_features = self.extract_text_feature(labelset).float()
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
            print("LSeg model loaded and frozen.")
        elif self.use_ape:
            from xdecoder_test.models.predictor_lazy import VisualizationDemo
            # 直接在代码中设置参数，对应你的命令行参数
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
            
            # opts 参数
            self.opts = [
                "train.init_checkpoint=third_party/APE/checkpoint/APE-L_D.pth",
                "model.model_language.cache_dir=",
                "model.model_vision.select_box_nums_for_evaluation=500",
                "model.model_vision.text_feature_bank_reset=True"
            ]
            
            # 设置logger
            setup_logger(name="fvcore")
            setup_logger(name="ape")
            setup_logger(name="timm")
            
            # 初始化配置和demo
            self.ape_cfg = self._setup_cfg()
            self.demo = VisualizationDemo(self.ape_cfg, args=self._create_args_namespace())
        else:
            self.xdecoder_cfg = xdecoder_cfg        
            self.xdecoder_cfg['device'] = self.device

            # 1. --- 加载并冻结“教师”模型 ---
            print("Loading teacher models...")
            # 教师1: X-Decoder (从你现有的模型加载)
            # 注意：这里我们不需要gaussians模型，因为我们只用X-Decoder的2D部分
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
            print(f"Final labelset: {labelset}")  # 调试用，确认类别列表
            self.xdecoder_teacher.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(labelset, is_eval=True)
            metadata = MetadataCatalog.get('demo')
            self.xdecoder_teacher.model.metadata = metadata
            self.xdecoder_teacher.model.sem_seg_head.num_classes = len(cfg.all_label)
            print("X-Decoder teacher loaded and frozen.")

        # 教师2: Sonata
        if flash_attn is not None:
            # sonata.utils.set_seed(53124)
            self.sonata_teacher = sonata.load("sonata", repo_id="facebook/sonata").cuda()
        else:
            custom_config = dict(
                enc_patch_size=[1024 for _ in range(5)],  # reduce patch size if necessary
                enable_flash=False,
            )
            self.sonata_teacher = sonata.load(
                "sonata", repo_id="facebook/sonata", custom_config=custom_config
            ).cuda()
        self.sonata_teacher.eval()
        for param in self.sonata_teacher.parameters():
            param.requires_grad = False
        print("Sonata teacher loaded and frozen.")

        # 2. --- 定义“学生”模型 ---
        # 确定X-Decoder输出的特征维度
        # 这通常是 mask_embed 的维度, 比如 256
        xdecoder_feature_dim = 512+6 # 请根据你的X-Decoder配置修改
        self.affinity_student = AffinityPredictor(
            input_dim=xdecoder_feature_dim, 
            embed_dim=128, 
            hidden_dim=512,
            # num_res_blocks=4
        ).cuda()
        # # 替换AffinityPredictor
        # xdecoder_feature_dim = 512 + 6  # 你的输入维度
        # embed_dim = 128  # 你希望的输出维度
        # # 使用wrapper
        # self.affinity_student = MinkUNetWrapper(
        #     in_channels=xdecoder_feature_dim,
        #     out_channels=embed_dim,
        #     D=3,
        #     arch='MinkUNet18A'
        # )
        print("AffinityPredictor student created.")
        
        # 3. 定义损失函数
        self.criterion = nn.CrossEntropyLoss()

        self.num_anchors_per_scene = 4096 # 每个场景采样多少个锚点进行学习
        self.num_negatives_per_anchor = 63  # 每个锚点配多少个负样本
        self.info_nce_temperature = 0.07   # InfoNCE损失的温度系数

    def _create_args_namespace(self):
        """创建一个简单的参数对象，模拟argparse的Namespace"""
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
        """设置配置，对应原来的setup_cfg函数"""
        # 加载配置文件
        cfg = LazyConfig.load(self.config_file)
        cfg = LazyConfig.apply_overrides(cfg, self.opts)
        
        # 设置output_dir
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
        
        # 设置confidence threshold
        if "model_vision" in cfg.model:
            cfg.model.model_vision.test_score_thresh = self.confidence_threshold
        else:
            cfg.model.test_score_thresh = self.confidence_threshold
        
        return cfg

    def extract_text_feature(self, labelset):
        # "ViT-B/32" # the model that LSeg uses
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
        # 1. --- Unpack data ---
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

        # 2. --- Initialize accumulators for feature fusion ---
        sum_features = torch.zeros((num_points, feature_dim), dtype=torch.float32, device='cpu')
        counter = torch.zeros((num_points, 1), dtype=torch.float32, device='cpu')

        # 3. --- Process in smaller batches with aggressive memory management ---
        num_batches = (num_views + max_batch_size - 1) // max_batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Processing Lseg feature:"):
            start_idx = batch_idx * max_batch_size
            end_idx = min(start_idx + max_batch_size, num_views)
            current_batch_size = end_idx - start_idx
            
            # print(f"Processing batch {batch_idx + 1}/{num_batches} (images {start_idx}-{end_idx-1})")
            
            # === 步骤A: 准备当前批次的图像 ===
            batch_images = []
            original_shapes = []
            
            for i in range(start_idx, end_idx):
                img_tensor = imgs[i].float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)
                original_shapes.append(img_tensor.shape[1:])
                
                # 预处理并移到GPU
                image_tensor_processed = self.lseg_normalize(img_tensor)
                image_tensor_processed = F.interpolate(
                    image_tensor_processed.unsqueeze(0),
                    size=(240, 320),
                    mode='bilinear',
                    align_corners=True
                ).squeeze(0)
                
                batch_images.append(image_tensor_processed)
            
            # 堆叠成batch
            batch_tensor = torch.stack(batch_images, dim=0).to(self.device)
            
            # === 步骤B: LSeg推理 ===
            with torch.no_grad():
                # 强制设置模型为eval模式，避免内部状态累积
                self.lseg_evaluator.eval()
                
                # LSeg inference
                if hasattr(self.lseg_evaluator, 'forward'):
                    batch_feat_2d = self.lseg_evaluator.forward(batch_tensor, label_set='')
                else:
                    batch_inputs = [batch_tensor[i:i+1] for i in range(current_batch_size)]
                    batch_outputs = self.lseg_evaluator.parallel_forward(batch_inputs)
                    batch_feat_2d = torch.cat([output[0] for output in batch_outputs], dim=0)
                
                # === 步骤C: 特征处理和提升 ===
                for i, feat_2d in enumerate(batch_feat_2d):
                    view_idx = start_idx + i
                    original_shape = original_shapes[i]
                    
                    # 插值回原尺寸
                    feat_2d_interpolated = F.interpolate(
                        feat_2d.unsqueeze(0),
                        size=original_shape,
                        mode='bilinear',
                        align_corners=True
                    ).squeeze(0)
                    
                    # 特征提升和融合
                    mask_this_view = (ori_coords_3ds[:, 0] == view_idx)
                    all_mask_this_view = (mask_2ds[:, 0] == view_idx)
                    
                    if mask_this_view.any():
                        mask_2d = mask_2ds[all_mask_this_view][:, 1].bool()
                        point_indices = torch.where(mask_2d)[0]
                        
                        x_coords = x_labels[mask_this_view]
                        y_coords = y_labels[mask_this_view]
                        
                        # 提取特征并立即移到CPU
                        lifted_features = feat_2d_interpolated[:, x_coords, y_coords].permute(1, 0).cpu()
                        
                        # 累积到CPU tensors
                        sum_features.index_add_(0, point_indices.cpu(), lifted_features)
                        counter.index_add_(0, point_indices.cpu(), 
                                        torch.ones((len(point_indices), 1), dtype=torch.float32))
                    
                    # 立即删除插值特征
                    del feat_2d_interpolated
            
            # === 步骤D: 激进的内存清理 ===
            # 删除所有中间变量
            del batch_images, batch_tensor, batch_feat_2d
            
            # 清空CUDA缓存
            torch.cuda.empty_cache()
            
            # # 额外的内存管理技巧
            # if batch_idx % 3 == 0:  # 每3个批次做一次深度清理
            #     # 清理LSeg模型的internal states
            #     if hasattr(self.lseg_evaluator, 'zero_grad'):
            #         self.lseg_evaluator.zero_grad()
                
            #     # 强制同步GPU操作
            #     torch.cuda.synchronize()
                
            #     # 报告内存使用
            #     allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            #     reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            #     print(f"  Memory after batch {batch_idx}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

        # 4. --- 最终特征计算 ---
        counter[counter == 0] = 1e-6
        final_scene_features = sum_features / counter
        
        # 5. --- Fill features for unseen points using KDTree ---
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
        
        # 移回GPU
        final_scene_features = final_scene_features.to(self.device)

        # # 8. --- Debug visualization (optional) ---
        # pca_color = get_pca_color(final_scene_features, brightness=1.2, center=True)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(scene_coords)
        # pcd.colors = o3d.utility.Vector3dVector(pca_color.cpu().detach().numpy())
        # o3d.io.write_point_cloud("xdecoder_test/debug_projection_vis/2dlsegpca.ply", pcd)
        # import pdb;pdb.set_trace()

        return final_scene_features, self.text_features, self.logit_scale

    # def lift_lseg_features(self, batch_data, max_batch_size=24):
    #     """
    #     Extracts features from 2D images using LSeg and lifts them to a 3D point cloud.
    #     Enhanced version with aggressive memory management to prevent accumulation.
    #     """
    #     # 1. --- Unpack data ---
    #     (
    #         scene_coords, scene_coords_3d, scene_inds_reconstruct, scene_label,
    #         ori_coords_3ds, coords_3ds, feat_3ds, gauss_featuress, labels_3ds,
    #         binary_label_3ds, label_2ds, imgs, x_labels, y_labels, mask_2ds,
    #         inds_reconstructs, unique_maps, mappings, captions, scene_gauss_features
    #     ) = batch_data

    #     num_points = scene_coords.shape[0]
    #     feature_dim = 512
    #     num_views = len(imgs)
        
    #     print(f"Processing {num_views} views with LSeg using batch size {max_batch_size}...")

    #     # 2. --- Initialize accumulators for feature fusion ---
    #     sum_features = torch.zeros((num_points, feature_dim), dtype=torch.float32, device='cpu')
    #     counter = torch.zeros((num_points, 1), dtype=torch.float32, device='cpu')

    #     # 3. --- Process in smaller batches with aggressive memory management ---
    #     num_batches = (num_views + max_batch_size - 1) // max_batch_size
        
    #     for batch_idx in tqdm(range(num_batches), desc="Processing Lseg feature:"):
    #         start_idx = batch_idx * max_batch_size
    #         end_idx = min(start_idx + max_batch_size, num_views)
    #         current_batch_size = end_idx - start_idx
            
    #         # print(f"Processing batch {batch_idx + 1}/{num_batches} (images {start_idx}-{end_idx-1})")
            
    #         # === 步骤A: 准备当前批次的图像 ===
    #         batch_images = []
    #         original_shapes = []
            
    #         for i in range(start_idx, end_idx):
    #             img_tensor = imgs[i].float() / 255.0
    #             img_tensor = img_tensor.permute(2, 0, 1)
    #             original_shapes.append(img_tensor.shape[1:])
                
    #             # 预处理并移到GPU
    #             image_tensor_processed = self.lseg_normalize(img_tensor)
    #             # image_tensor_processed = F.interpolate(
    #             #     image_tensor_processed.unsqueeze(0),
    #             #     size=(240, 320),
    #             #     mode='bilinear',
    #             #     align_corners=True
    #             # ).squeeze(0)
                
    #             batch_images.append(image_tensor_processed)
            
    #         # 堆叠成batch
    #         batch_tensor = torch.stack(batch_images, dim=0).to(self.device)
            
    #         # === 步骤B: LSeg推理 ===
    #         with torch.no_grad():
    #             # 强制设置模型为eval模式，避免内部状态累积
    #             # self.lseg_evaluator.eval()
    #             # LSeg inference
    #             batch_feat_2d = self.lseg_evaluator.extract_image_feature(
    #                 batch_tensor,
    #                 [self.scene_config.fusion.img_dim[0], self.scene_config.fusion.img_dim[1]],
    #             )
    #             # import matplotlib.pyplot as plt
    #             # from sklearn.decomposition import PCA

    #             # def visualize_alignment(original_image_tensor, feature_map_tensor, title=""):
    #             #     """
    #             #     可视化原始图像和其对应的特征图，以检查对齐情况。

    #             #     参数:
    #             #     - original_image_tensor (torch.Tensor): 经过预处理的原始图像张量，形状为 [1, 3, H, W]。
    #             #     - feature_map_tensor (torch.Tensor): 上采样后的特征图张量，形状为 [C, H, W]。
    #             #     """
    #             #     print("正在生成可视化图像...")
                    
    #             #     # --- 1. 准备原始图像 ---
    #             #     # 反归一化并将维度转换为 (H, W, 3) 以便显示
    #             #     img_display = original_image_tensor.squeeze(0).cpu().detach()
    #             #     # 反归一化公式: original = standardized * std + mean
    #             #     # 我们的归一化是 (x - 0.5) / 0.5, 所以反过来是 x * 0.5 + 0.5
    #             #     img_display = img_display * 0.5 + 0.5
    #             #     img_display = img_display.permute(1, 2, 0)
    #             #     img_display = torch.clamp(img_display, 0, 1) # 确保值在[0,1]范围内
                    
    #             #     # --- 2. 准备特征图 ---
    #             #     # 使用PCA将特征从 C 维降到 3 维
    #             #     feat_map = feature_map_tensor.cpu().detach()
    #             #     C, H, W = feat_map.shape
                    
    #             #     # a. 重塑为 (H*W, C) 以便PCA处理
    #             #     feat_map_reshaped = feat_map.permute(1, 2, 0).reshape(-1, C)
                    
    #             #     # b. 应用PCA
    #             #     pca = PCA(n_components=3)
    #             #     pca_features = pca.fit_transform(feat_map_reshaped.numpy())
                    
    #             #     # c. 将PCA结果归一化到 [0, 1] 范围以便显示为颜色
    #             #     pca_min = pca_features.min(axis=0)
    #             #     pca_max = pca_features.max(axis=0)
    #             #     pca_features = (pca_features - pca_min) / (pca_max - pca_min)
                    
    #             #     # d. 重塑回图像形状 (H, W, 3)
    #             #     pca_display = pca_features.reshape(H, W, 3)
                    
    #             #     # --- 3. 使用 Matplotlib 显示 ---
    #             #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
    #             #     ax1.imshow(img_display)
    #             #     ax1.set_title(" (Original Image)")
    #             #     ax1.axis('off')
                    
    #             #     ax2.imshow(pca_display)
    #             #     ax2.set_title(" (Feature Map PCA)")
    #             #     ax2.axis('off')
                    
    #             #     if title:
    #             #         fig.suptitle(title, fontsize=16)
                    
    #             #     plt.tight_layout()
    #             #     # plt.show() # 在交互式环境（如Jupyter）中使用
                    
    #             #     # 保存图像以便查看
    #             #     save_path = "xdecoder_test/debug_2d_seg_vis/visualization_check.png"
    #             #     plt.savefig(save_path, dpi=150)
    #             #     print(f"可视化图像已保存到: {save_path}")
    #             #     plt.close() # 关闭图像，防止在循环中不断弹出
    #             # visualize_alignment(batch_tensor[0], batch_feat_2d[0], title="check")
    #             # if hasattr(self.lseg_evaluator, 'forward'):
    #             #     batch_feat_2d = self.lseg_evaluator.forward(batch_tensor, label_set='')
    #             # else:
    #             #     batch_inputs = [batch_tensor[i:i+1] for i in range(current_batch_size)]
    #             #     batch_outputs = self.lseg_evaluator.parallel_forward(batch_inputs)
    #             #     batch_feat_2d = torch.cat([output[0] for output in batch_outputs], dim=0)
                
    #             # === 步骤C: 特征处理和提升 ===
    #             for i, feat_2d in enumerate(batch_feat_2d):
    #                 view_idx = start_idx + i
    #                 original_shape = original_shapes[i]
                    
    #                 # 插值回原尺寸
    #                 feat_2d_interpolated = F.interpolate(
    #                     feat_2d.unsqueeze(0),
    #                     size=original_shape,
    #                     mode='bilinear',
    #                     align_corners=True
    #                 ).squeeze(0)
                    
    #                 # 特征提升和融合
    #                 mask_this_view = (ori_coords_3ds[:, 0] == view_idx)
    #                 all_mask_this_view = (mask_2ds[:, 0] == view_idx)
                    
    #                 if mask_this_view.any():
    #                     mask_2d = mask_2ds[all_mask_this_view][:, 1].bool()
    #                     point_indices = torch.where(mask_2d)[0]
                        
    #                     x_coords = x_labels[mask_this_view]
    #                     y_coords = y_labels[mask_this_view]
                        
    #                     # 提取特征并立即移到CPU
    #                     lifted_features = feat_2d_interpolated[:, x_coords, y_coords].permute(1, 0).cpu()
                        
    #                     # 累积到CPU tensors
    #                     sum_features.index_add_(0, point_indices.cpu(), lifted_features)
    #                     counter.index_add_(0, point_indices.cpu(), 
    #                                     torch.ones((len(point_indices), 1), dtype=torch.float32))
                    
    #                 # 立即删除插值特征
    #                 del feat_2d_interpolated
            
    #         # === 步骤D: 激进的内存清理 ===
    #         # 删除所有中间变量
    #         del batch_images, batch_tensor, batch_feat_2d
            
    #         # 清空CUDA缓存
    #         torch.cuda.empty_cache()
            
    #         # # 额外的内存管理技巧
    #         # if batch_idx % 3 == 0:  # 每3个批次做一次深度清理
    #         #     # 清理LSeg模型的internal states
    #         #     if hasattr(self.lseg_evaluator, 'zero_grad'):
    #         #         self.lseg_evaluator.zero_grad()
                
    #         #     # 强制同步GPU操作
    #         #     torch.cuda.synchronize()
                
    #         #     # 报告内存使用
    #         #     allocated = torch.cuda.memory_allocated(self.device) / 1024**3
    #         #     reserved = torch.cuda.memory_reserved(self.device) / 1024**3
    #         #     print(f"  Memory after batch {batch_idx}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    #     # 4. --- 最终特征计算 ---
    #     counter[counter == 0] = 1e-6
    #     final_scene_features = sum_features / counter
        
    #     # 5. --- Fill features for unseen points using KDTree ---
    #     seen_mask = (counter.squeeze() > 1e-5)
    #     unseen_mask = ~seen_mask
        
    #     if unseen_mask.any() and seen_mask.any():
    #         seen_coords = scene_coords[seen_mask].cpu().numpy()
    #         unseen_coords = scene_coords[unseen_mask].cpu().numpy()
            
    #         from sklearn.neighbors import KDTree
    #         kdtree = KDTree(seen_coords)
    #         distances, indices = kdtree.query(unseen_coords, k=1)
    #         neighbor_features = final_scene_features[seen_mask][indices.flatten()]
    #         final_scene_features[unseen_mask] = neighbor_features
        
    #     # 移回GPU
    #     final_scene_features = final_scene_features.to(self.device)

    #     # # 8. --- Debug visualization (optional) ---
    #     # pca_color = get_pca_color(final_scene_features, brightness=1.2, center=True)
    #     # pcd = o3d.geometry.PointCloud()
    #     # pcd.points = o3d.utility.Vector3dVector(scene_coords)
    #     # pcd.colors = o3d.utility.Vector3dVector(pca_color.cpu().detach().numpy())
    #     # o3d.io.write_point_cloud("xdecoder_test/debug_projection_vis/2dlsegpca.ply", pcd)
    #     # import pdb;pdb.set_trace()

    #     return final_scene_features, self.text_features, self.logit_scale


    def lift_xdecoder_features(self, batch_data):
        """
        这个函数的核心逻辑完全来自于你的 `validate` 函数中，
        它负责从2D图像通过X-Decoder生成3D点的特征。
        """
        # 解包数据
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
        # batch_dict = {
        #     "scene_coords": scene_coords,
        #     "scene_coords_3d": scene_coords_3d,
        #     "scene_inds_reconstruct": scene_inds_reconstruct,
        #     "scene_label": scene_label,
        #     "ori_coords_3ds": ori_coords_3ds,
        #     "coords_3ds": coords_3ds,
        #     "feat_3ds": feat_3ds,
        #     "gauss_featuress": gauss_featuress,
        #     "labels_3ds": labels_3ds,
        #     "binary_label_3ds": binary_label_3ds,
        #     "label_2ds": label_2ds,
        #     "imgs": imgs,
        #     "x_labels": x_labels,
        #     "y_labels": y_labels,
        #     "mask_2ds": mask_2ds,
        #     "inds_reconstructs": inds_reconstructs,
        #     "unique_maps": unique_maps,
        #     "mappings": mappings,
        #     "captions": captions,
        #     "gauss_features": scene_gauss_features,
        # }

        # torch.save(batch_dict, "xdecoder_test/debug_projection_vis/wrong_batch_dict.pt")
        # import pdb;pdb.set_trace()

        # scene_feature_accumulator = torch.zeros((scene_coords.shape[0], 512), dtype=torch.float32)
        point_info = defaultdict(list)
        num_points = scene_coords.shape[0]
        num_classes = len(self.cfg.all_label)
        feature_dim = 512
        counter = torch.zeros((scene_coords.shape[0]), dtype=scene_label.dtype)


        # ======================================================================
        # 【Pass 1 (重构): 构建规则化的 Padded 张量】
        # ======================================================================

        # a. 计算每个点被多少个视角看到，并确定最大视角数 M
        view_counts = torch.zeros(num_points, dtype=torch.long, device=self.device)
        # 这个循环无法避免，因为它需要迭代Dataloader的输出，但循环内部是高效的
        for view_idx in range(len(imgs)):
            all_mask_this_view = (mask_2ds[:, 0] == view_idx)
            mask_2d = mask_2ds[all_mask_this_view][:, 1].bool().to(view_counts.device) # 获取当前视角的布尔掩码
            view_counts.index_add_(0, torch.where(mask_2d)[0], torch.ones_like(view_counts[0:1]).expand(mask_2d.sum()))
        
        M = view_counts.max().item()                        

        # b. 初始化 Padded 张量和有效性掩码
        # padded_features = torch.zeros(num_points, M, feature_dim, dtype=torch.float32, device=self.device)
        # padded_logits = torch.zeros(num_points, M, num_classes, dtype=torch.float32, device=self.device)
        # validity_mask = torch.zeros(num_points, M, dtype=torch.bool, device=self.device)
        
        # c. 再次遍历视角，填充 Padded 张量
        # 我们用 view_counts 作为每个点当前已填充数量的指针
        # current_fill_idx = torch.zeros(num_points, dtype=torch.long, device=self.device)
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
            # outputs dict_keys(['pred_logits', 'pred_masks', 'aux_outputs', 'mask_embed', 'mask_pooled_features', 'logit_scale'])

            # --- 1. 初始化和参数定义 ---
            # 假设批处理大小(B)为1，因为您的代码似乎是这样处理的
            # assert B == 1, "代码实现假设批处理大小为1"
            output_2d = []

            # 定义一些必要的参数 (这些值您可能需要根据配置进行调整)
            scores_keep_thresh = 0.0  # 实例置信度的过滤阈值
            # 从模型输出推断类别数 (减去背景/null类别)
            num_classes = outputs["pred_logits"].shape[-1] - 1

            mask_pred_results = outputs["pred_masks"]
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=tuple(self.cfg.mask_shape),
                mode="bicubic",
                align_corners=False,
                antialias=True
            )

            # --- 2. 循环处理批处理中的每个场景 (这里B=1,所以只循环一次) ---
            scene_idx = 0
            # --- 3. 提取当前场景的预测结果 ---
            mask_pred_result = mask_pred_results[scene_idx]  # [Q, H_m, W_m]
            # mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
            #     mask_pred_result, image_ori.shape, model.cfg.mask_shape[0], model.cfg.mask_shape[1]
            # )
            mask_cls_results = outputs["pred_logits"][scene_idx]  # [Q, C+1]
            pred_embeddings = outputs["mask_embed"][scene_idx]    # [Q, D]

            # 提取映射回3D点所需的2D坐标
            # 获取原始3D点，用于确定输出特征矩阵的形状
            ori_coords = batch_input["ori_coords"]
            x_label = batch_input["x_label"][ori_coords[:, 0] == scene_idx]
            y_label = batch_input["y_label"][ori_coords[:, 0] == scene_idx]                    
            points_in_scene = ori_coords[ori_coords[:, 0] == scene_idx]

            # --- 4. 过滤和筛选实例 ---
            # a. 计算类别分数
            scores, labels = F.softmax(mask_cls_results, dim=-1)[..., :-1].max(-1)
            
            # b. 根据分数阈值进行初步过滤
            keep = scores > scores_keep_thresh
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred_result[keep].sigmoid() # [Q', H_m, W_m]
            cur_embedding = pred_embeddings[keep]         # [Q', D]
            
            # 如果没有一个实例通过筛选，则为所有点生成零特征
            if cur_masks.shape[0] == 0:
                num_points = points_in_scene.shape[0]
                feat_dim = pred_embeddings.shape[-1]
                single_2d_feature = torch.zeros((num_points, feat_dim), device=pred_embeddings.device)
            else:
                # c. NMS式处理，确定每个像素的最终归属
                cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
                cur_mask_ids = cur_prob_masks.argmax(0) # [H_m, W_m]

                final_keep_indices = []
                final_masks = []
                for k in range(cur_classes.shape[0]):
                    # 检查这个实例是否在最终的像素归属中占有一席之地
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        final_keep_indices.append(k)
                        final_masks.append(mask)

                # # 1. 初始化用于存储所有类别最终结果的容器
                # final_keep_indices = []
                # final_masks = []

                # # 2. 获取所有出现过的唯一类别ID
                # unique_classes = torch.unique(cur_classes)

                # # 3. 遍历每个类别，独立进行NMS
                # for class_id in unique_classes:
                #     # a. 找到所有属于当前类别的实例的掩码(mask)
                #     #    这是实现类别感知的关键步骤
                #     is_current_class = (cur_classes == class_id)
                    
                #     # 如果当前类别没有实例，则跳过
                #     if not is_current_class.any():
                #         continue
                        
                #     # b. 获取当前类别所有实例的数据，以及它们在`cur_...`张量中的【原始索引】
                #     #    `torch.where`会返回满足条件的元素的索引，这正是我们需要的
                #     original_indices_of_class = torch.where(is_current_class)[0]
                    
                #     class_scores = cur_scores[is_current_class]
                #     class_masks = cur_masks[is_current_class]  # 形状: [N_class, H_m, W_m]

                #     # c. 【在类别内部】执行NMS的核心逻辑
                #     #    这里的竞争是公平的，因为大家都是同一个类别
                #     if class_masks.shape[0] == 0:
                #         continue
                #     class_prob_masks = class_scores.view(-1, 1, 1) * class_masks
                    
                #     # argmax的结果是相对于当前类别内部的索引 (范围从 0 到 N_class-1)
                #     class_internal_mask_ids = class_prob_masks.argmax(0) # 形状: [H_m, W_m]
                    
                #     # d. 筛选该类别内留下的实例
                #     for k_class in range(class_scores.shape[0]):
                #         # 检查该实例是否在【类别内部的竞争】中胜出
                #         mask_area = (class_internal_mask_ids == k_class).sum().item()
                #         original_area = (class_masks[k_class] >= 0.5).sum().item()
                        
                #         # 生成这个实例最终的、干净的、无重叠的掩码
                #         mask = (class_internal_mask_ids == k_class) & (class_masks[k_class] >= 0.5)

                #         if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                #             # e. 关键: 保存的是该实例在`cur_...`张量中的【原始全局索引】，而不是类别内的局部索引k_class
                #             original_global_idx = original_indices_of_class[k_class]
                            
                #             final_keep_indices.append(original_global_idx.item())
                #             final_masks.append(mask)

                # 如果经过NMS后没有实例留下
                if not final_keep_indices:
                    num_points = points_in_scene.shape[0]
                    feat_dim = pred_embeddings.shape[-1]
                    single_2d_feature = torch.zeros((num_points, feat_dim), device=pred_embeddings.device)
                else:
                    # 获取最终被保留下来的实例的特征和掩码
                    final_embedding = cur_embedding[final_keep_indices]
                    final_mask_stack = torch.stack(final_masks, dim=0) # [Q_final, H_m, W_m]
                    
                    # --- 5. 将2D特征分配到3D点 ---
                    # a. 初始化累加器和计数器
                    num_points = points_in_scene.shape[0]
                    feat_dim = final_embedding.shape[-1]
                    mask_3d_feature = torch.zeros((num_points, feat_dim), device=final_embedding.device)
                    counter2d = torch.zeros((num_points, 1), device=final_embedding.device) 

                    # b. 将2D掩码投影到3D点
                    mask_3d = final_mask_stack[:, x_label, y_label] # [Q_final, N] (注意x,y顺序)
                    mask_3d = mask_3d >= 0.5
                    # c. 累加特征和计数
                    for single_mask_3d, mask_emb in zip(mask_3d, final_embedding):
                        # single_mask_3d 是一个布尔向量，表示哪些点被当前实例覆盖
                        mask_3d_feature[single_mask_3d] += mask_emb
                        counter2d[single_mask_3d] += 1
                    
                    # d. 计算平均特征
                    counter2d[counter2d == 0] = 1e-5 # 防止除以零
                    single_2d_feature = mask_3d_feature / counter2d
                    del mask_3d_feature, counter2d, mask_3d

            # 将当前场景计算出的特征添加到最终列表中
            output_2d.append(single_2d_feature)

            # --- 6. (可选) 更新outputs字典 ---
            outputs["2d_pred_feature"] = output_2d
            # print(f"成功生成 '2d_pred_feature'，包含 {len(output_2d)} 个场景的特征。")
            # if output_2d:
            #     print(f"第一个场景的特征形状为: {output_2d[0].shape}")

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

            # feature_2d = feature_2d.to(scene_feature_accumulator.device)
            # scene_feature_accumulator[mask_2d] += feature_2d
            point_indices_in_this_view = torch.where(mask_2d)[0]
            # fill_indices = current_fill_idx[point_indices_in_this_view]
            # # 并行填充
            # padded_features[point_indices_in_this_view, fill_indices] = feature_2d.to(padded_features.device)
            # padded_logits[point_indices_in_this_view, fill_indices] = logits_pred_2d.to(padded_logits.device)
            # validity_mask[point_indices_in_this_view, fill_indices] = True
            
            # 更新指针
            # current_fill_idx.index_add_(0, point_indices_in_this_view.to(current_fill_idx.device), torch.ones_like(fill_indices))

            for i, global_idx in enumerate(point_indices_in_this_view):
                # .item() 从0维张量中取出Python数字，作为字典的键
                point_info[global_idx.item()].append({
                    "feature": feature_2d[i],
                    "logits": logits_pred_2d[i],
                })
            counter[mask_2d] += 1
            del feature_2d           # 解除对所有切片的聚合张量的引用
            del logits_pred_2d
            del single_2d_feature    # 在你的代码中，它被用来构建 feature_2d
            del img                  # 删掉输入的图像张量

            torch.cuda.empty_cache()

        # counter_ = counter.clone()
        # counter_[counter_ == 0] = 1.0
        # final_scene_features = scene_feature_accumulator / counter_.unsqueeze(1)
        # # 【步骤 2: 逐点进行共识判断与择优融合】 (不变)
        # feature_dim = feature_2d.shape[1]
        # final_scene_features = torch.zeros((scene_coords.shape[0], feature_dim), dtype=torch.float32)
        # TOP_K = 3 
        # # --- Pass 2: 遍历每个点，生成最终特征 --- (不变)
        # for point_idx in range(scene_coords.shape[0]):
        #     views_data = point_info[point_idx]
        #     if not views_data:
        #         continue
                
        #     # ======================================================================
        #     # 【步骤 2.a: 达成共识 (采纳您建议的“硬投票”法)】
        #     # ======================================================================
            
        #     # 之前的方法 (Logit平均):
        #     # avg_logits = torch.stack([v["logits"] for v in views_data]).mean(dim=0)
        #     # consensus_class_idx = torch.argmax(avg_logits)

        #     # 您的建议 (硬投票)，这更直接、更鲁棒！
        #     votes = torch.stack([torch.argmax(v["logits"]) for v in views_data])
        #     # torch.bincount 会统计每个类别出现的次数，再用 argmax 找到出现次数最多的那个
        #     consensus_class_idx = torch.bincount(votes).argmax()
        #     # ======================================================================

        #     # b. 择优：根据每个视角对“共识类别”的支持度进行排序 (不变)
        #     for view in views_data:
        #         view['agreement_score'] = view['logits'][consensus_class_idx]
            
        #     views_data.sort(key=lambda v: v['agreement_score'], reverse=True)
        #     top_k_views = views_data[:TOP_K]

        #     # c. 融合：对筛选出的Top-K个特征进行加权平均 (不变)
        #     agreement_scores = torch.tensor([v['agreement_score'] for v in top_k_views], device=self.device)
        #     fusion_weights = F.softmax(agreement_scores, dim=0)
            
        #     fused_feature = torch.zeros(feature_dim, device=self.device)
        #     for i, view in enumerate(top_k_views):
        #         fused_feature += view["feature"] * fusion_weights[i]

        #     final_scene_features[point_idx] = fused_feature

        # ======================================================================
        # 【Pass 2 (全新革命性改动): 分块向量化融合】
        # ======================================================================
        num_points = scene_coords.shape[0]
        num_classes = len(self.cfg.all_label)
        # a. 准备工作
        final_scene_features = torch.zeros((num_points, feature_dim), dtype=torch.float32, device=self.device)
        # 筛选出所有被至少一个视角看到的点
        points_with_views_indices = torch.tensor(list(point_info.keys()), device=self.device)
        num_points_with_views = len(points_with_views_indices)

        # 定义块大小
        chunk_size = 50000 # 一次处理50000个点，这个值可以根据显存灵活调整

        # b. 在一个只遍历“有视角信息的点”的循环中，分块处理
        for i in range(0, num_points_with_views, chunk_size):
            # 1. 获取当前块的点
            chunk_indices = points_with_views_indices[i:i+chunk_size]
            current_chunk_size = len(chunk_indices)

            # 2. 为【当前块】构建小型的、临时的Padded张量
            # 首先，找到这个块内的最大视角数 M_chunk
            M_chunk = max(len(point_info[idx.item()]) for idx in chunk_indices)
            
            # 创建临时的、绝对不会爆显存的Padded张量
            padded_features_chunk = torch.zeros(current_chunk_size, M_chunk, feature_dim, device=self.device)
            padded_logits_chunk = torch.zeros(current_chunk_size, M_chunk, num_classes, device=self.device)
            validity_mask_chunk = torch.zeros(current_chunk_size, M_chunk, dtype=torch.bool, device=self.device)

            # 3. 填充这个小型的Padded张量
            for j, global_idx in enumerate(chunk_indices):
                views_data = point_info[global_idx.item()]
                num_views_for_point = len(views_data)
                validity_mask_chunk[j, :num_views_for_point] = True
                for k, view_data in enumerate(views_data):
                    padded_features_chunk[j, k] = view_data["feature"]
                    padded_logits_chunk[j, k] = view_data["logits"]
            
            # 4. 在这个小张量上，执行我们之前设计的【全套向量化融合逻辑】
            # (这部分代码和上一版完全一样，只是操作对象从巨大张量变成了小chunk)
            K_chunk = min(M_chunk, 3)
            
            # a. 共识
            sum_logits_chunk = padded_logits_chunk.sum(dim=1)
            num_valid_views_chunk = validity_mask_chunk.sum(dim=1, keepdim=True).clamp(min=1)
            avg_logits_chunk = sum_logits_chunk / num_valid_views_chunk
            consensus_idx_chunk = torch.argmax(avg_logits_chunk, dim=1)

            # b. 择优
            agreement_scores_chunk = torch.gather(padded_logits_chunk, 2, consensus_idx_chunk.view(-1, 1, 1).expand(-1, M_chunk, -1)).squeeze(-1)
            agreement_scores_chunk.masked_fill_(~validity_mask_chunk, -torch.inf)

            # c. Top-K
            top_k_scores_chunk, top_k_indices_chunk = torch.topk(agreement_scores_chunk, k=K_chunk, dim=1)
            
            # d. 融合
            top_k_features_chunk = torch.gather(padded_features_chunk, 1, top_k_indices_chunk.unsqueeze(-1).expand(-1, -1, feature_dim))
            fusion_weights_chunk = F.softmax(top_k_scores_chunk, dim=1)
            fused_features_chunk = (top_k_features_chunk * fusion_weights_chunk.unsqueeze(-1)).sum(dim=1)

            # 5. 将计算好的块结果，放回最终的全场景特征张量中
            final_scene_features[chunk_indices] = fused_features_chunk

        scene_coords = scene_coords.to(counter.device)
        scene_true = scene_coords[counter != 0]
        scene_false = scene_coords[counter == 0]
        if scene_true.shape[0] > 0 and scene_false.shape[0] > 0:
            # print(f"Found {scene_true.shape[0]} seen points and {scene_false.shape[0]} unseen points. Performing KDTree fill-in.")
            
            flase_idx = torch.where(counter == 0)[0]
            true_idx = torch.where(counter != 0)[0]

            # KDTree 必须在 CPU 上运行
            kdtree = KDTree(scene_true.cpu().numpy())

            distances, indices = kdtree.query(scene_false.cpu().numpy(), k=1)

            match = true_idx[indices.flatten()]

            final_scene_features[flase_idx] = final_scene_features[match]

        elif scene_false.shape[0] > 0:
            # 这是一个边缘情况：如果一个点都没看到，我们无法填充
            # 此时，final_scene_features 很可能全是0，这本身是合理的
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
        """
        将经过unsqueeze(0).permute(0, 3, 1, 2)处理后的图像转换为detectron2格式
        
        Args:
            processed_img: shape为(1, 3, H, W)的tensor
        
        Returns:
            HWC格式的numpy数组，BGR顺序，uint8类型
        """
        # 1. 移除batch维度: (1, 3, H, W) -> (3, H, W)
        img_chw = processed_img.squeeze(0)
        
        # 2. 转为numpy并移到CPU
        if hasattr(img_chw, 'cpu'):
            img_np = img_chw.cpu().numpy()
        else:
            img_np = img_chw.numpy()
        
        # 3. CHW -> HWC: (3, H, W) -> (H, W, 3)
        img_hwc = np.transpose(img_np, (1, 2, 0))
        
        # 4. 处理数值范围
        if img_hwc.dtype != np.uint8:
            if img_hwc.max() <= 1.0:
                # 如果值在0-1范围内，转换为0-255
                img_hwc = (img_hwc * 255.0).astype(np.uint8)
            else:
                # 否则直接转为uint8
                img_hwc = img_hwc.astype(np.uint8)
        
        # 5. RGB -> BGR (detectron2默认使用BGR)
        img_bgr = img_hwc[..., ::-1]
        
        return img_bgr

    def lift_ape_features(self, batch_data):
        """
        这个函数的核心逻辑完全来自于你的 `validate` 函数中，
        它负责从2D图像通过X-Decoder生成3D点的特征。
        """
        # 解包数据
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


        # ======================================================================
        # 【Pass 1 (重构): 构建规则化的 Padded 张量】
        # ======================================================================

        # a. 计算每个点被多少个视角看到，并确定最大视角数 M
        view_counts = torch.zeros(num_points, dtype=torch.long, device=self.device)
        # 这个循环无法避免，因为它需要迭代Dataloader的输出，但循环内部是高效的
        for view_idx in range(len(imgs)):
            all_mask_this_view = (mask_2ds[:, 0] == view_idx)
            mask_2d = mask_2ds[all_mask_this_view][:, 1].bool().to(view_counts.device) # 获取当前视角的布尔掩码
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
            # outputs = dict_keys(['instances', 'scene_features_for_instances', 'text_features', 'logit_scale', 'sem_seg', 'scene_features_for_semantic'])
            

            # --- 1. 初始化和参数定义 ---
            # 假设批处理大小(B)为1，因为您的代码似乎是这样处理的
            # assert B == 1, "代码实现假设批处理大小为1"
            output_2d = []

            # 定义一些必要的参数 (这些值您可能需要根据配置进行调整)
            scores_keep_thresh = 0.0  # 实例置信度的过滤阈值
            # 从模型输出推断类别数 (减去背景/null类别)
            # num_classes = outputs["pred_logits"].shape[-1] - 1

            mask_pred_results = outputs['instances'].pred_masks
            mask_pred_results = F.interpolate(
                mask_pred_results.unsqueeze(0),
                size=tuple(self.cfg.mask_shape),
                mode="bicubic",
                align_corners=False,
                antialias=True
            )

            # --- 2. 循环处理批处理中的每个场景 (这里B=1,所以只循环一次) ---
            scene_idx = 0
            # --- 3. 提取当前场景的预测结果 ---
            mask_pred_result = mask_pred_results[scene_idx]  # [Q, H_m, W_m]
            # mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
            #     mask_pred_result, image_ori.shape, model.cfg.mask_shape[0], model.cfg.mask_shape[1]
            # )
            # mask_cls_results = outputs["pred_logits"][scene_idx]  # [Q, C+1]
            pred_embeddings = outputs['scene_features_for_instances']    # [Q, D]

            # 提取映射回3D点所需的2D坐标
            # 获取原始3D点，用于确定输出特征矩阵的形状
            ori_coords = batch_input["ori_coords"]
            x_label = batch_input["x_label"][ori_coords[:, 0] == scene_idx]
            y_label = batch_input["y_label"][ori_coords[:, 0] == scene_idx]                    
            points_in_scene = ori_coords[ori_coords[:, 0] == scene_idx]

            # --- 4. 过滤和筛选实例 ---
            # a. 计算类别分数
            # scores, labels = F.softmax(mask_cls_results, dim=-1)[..., :-1].max(-1)
            scores = outputs['instances'].scores        # [Q']
            labels = outputs['instances'].pred_classes  # [Q']
            
            # b. 根据分数阈值进行初步过滤
            keep = scores > scores_keep_thresh
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred_result[keep].sigmoid() # [Q', H_m, W_m]
            cur_embedding = pred_embeddings[keep]         # [Q', D]
            
            # 如果没有一个实例通过筛选，则为所有点生成零特征
            if cur_masks.shape[0] == 0:
                num_points = points_in_scene.shape[0]
                feat_dim = pred_embeddings.shape[-1]
                single_2d_feature = torch.zeros((num_points, feat_dim), device=pred_embeddings.device)
            else:
                # c. NMS式处理，确定每个像素的最终归属
                cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
                cur_mask_ids = cur_prob_masks.argmax(0) # [H_m, W_m]

                final_keep_indices = []
                final_masks = []
                for k in range(cur_classes.shape[0]):
                    # 检查这个实例是否在最终的像素归属中占有一席之地
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        final_keep_indices.append(k)
                        final_masks.append(mask)

                # 如果经过NMS后没有实例留下
                if not final_keep_indices:
                    num_points = points_in_scene.shape[0]
                    feat_dim = pred_embeddings.shape[-1]
                    single_2d_feature = torch.zeros((num_points, feat_dim), device=pred_embeddings.device)
                else:
                    # 获取最终被保留下来的实例的特征和掩码
                    final_embedding = cur_embedding[final_keep_indices]
                    final_mask_stack = torch.stack(final_masks, dim=0) # [Q_final, H_m, W_m]
                    
                    # --- 5. 将2D特征分配到3D点 ---
                    # a. 初始化累加器和计数器
                    num_points = points_in_scene.shape[0]
                    feat_dim = final_embedding.shape[-1]
                    mask_3d_feature = torch.zeros((num_points, feat_dim), device=final_embedding.device)
                    counter2d = torch.zeros((num_points, 1), device=final_embedding.device) 

                    # b. 将2D掩码投影到3D点
                    mask_3d = final_mask_stack[:, x_label, y_label] # [Q_final, N] (注意x,y顺序)
                    mask_3d = mask_3d >= 0.5
                    # c. 累加特征和计数
                    for single_mask_3d, mask_emb in zip(mask_3d, final_embedding):
                        # single_mask_3d 是一个布尔向量，表示哪些点被当前实例覆盖
                        mask_3d_feature[single_mask_3d] += mask_emb
                        counter2d[single_mask_3d] += 1
                    
                    # d. 计算平均特征
                    counter2d[counter2d == 0] = 1e-5 # 防止除以零
                    single_2d_feature = mask_3d_feature / counter2d
                    del mask_3d_feature, counter2d, mask_3d

            # 将当前场景计算出的特征添加到最终列表中
            output_2d.append(single_2d_feature)

            # --- 6. (可选) 更新outputs字典 ---
            outputs["2d_pred_feature"] = output_2d
            # print(f"成功生成 '2d_pred_feature'，包含 {len(output_2d)} 个场景的特征。")
            # if output_2d:
            #     print(f"第一个场景的特征形状为: {output_2d[0].shape}")

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
                # .item() 从0维张量中取出Python数字，作为字典的键
                point_info[global_idx.item()].append({
                    "feature": feature_2d[i],
                    "logits": logits_pred_2d[i],
                })
            counter[mask_2d] += 1
            del feature_2d           # 解除对所有切片的聚合张量的引用
            del logits_pred_2d
            del single_2d_feature    # 在你的代码中，它被用来构建 feature_2d
            del img                  # 删掉输入的图像张量

            torch.cuda.empty_cache()
            
        # ======================================================================
        # 【Pass 2 (全新革命性改动): 分块向量化融合】
        # ======================================================================
        num_points = scene_coords.shape[0]
        num_classes = len(self.cfg.all_label)
        # a. 准备工作
        final_scene_features = torch.zeros((num_points, feature_dim), dtype=torch.float32, device=self.device)
        # 筛选出所有被至少一个视角看到的点
        points_with_views_indices = torch.tensor(list(point_info.keys()), device=self.device)
        num_points_with_views = len(points_with_views_indices)

        # 定义块大小
        chunk_size = 50000 # 一次处理50000个点，这个值可以根据显存灵活调整

        # b. 在一个只遍历“有视角信息的点”的循环中，分块处理
        for i in range(0, num_points_with_views, chunk_size):
            # 1. 获取当前块的点
            chunk_indices = points_with_views_indices[i:i+chunk_size]
            current_chunk_size = len(chunk_indices)

            # 2. 为【当前块】构建小型的、临时的Padded张量
            # 首先，找到这个块内的最大视角数 M_chunk
            M_chunk = max(len(point_info[idx.item()]) for idx in chunk_indices)
            
            # 创建临时的、绝对不会爆显存的Padded张量
            padded_features_chunk = torch.zeros(current_chunk_size, M_chunk, feature_dim, device=self.device)
            padded_logits_chunk = torch.zeros(current_chunk_size, M_chunk, num_classes, device=self.device)
            validity_mask_chunk = torch.zeros(current_chunk_size, M_chunk, dtype=torch.bool, device=self.device)

            # 3. 填充这个小型的Padded张量
            for j, global_idx in enumerate(chunk_indices):
                views_data = point_info[global_idx.item()]
                num_views_for_point = len(views_data)
                validity_mask_chunk[j, :num_views_for_point] = True
                for k, view_data in enumerate(views_data):
                    padded_features_chunk[j, k] = view_data["feature"]
                    padded_logits_chunk[j, k] = view_data["logits"]
            
            # 4. 在这个小张量上，执行我们之前设计的【全套向量化融合逻辑】
            # (这部分代码和上一版完全一样，只是操作对象从巨大张量变成了小chunk)
            K_chunk = min(M_chunk, 3)
            
            # a. 共识
            sum_logits_chunk = padded_logits_chunk.sum(dim=1)
            num_valid_views_chunk = validity_mask_chunk.sum(dim=1, keepdim=True).clamp(min=1)
            avg_logits_chunk = sum_logits_chunk / num_valid_views_chunk
            consensus_idx_chunk = torch.argmax(avg_logits_chunk, dim=1)

            # b. 择优
            agreement_scores_chunk = torch.gather(padded_logits_chunk, 2, consensus_idx_chunk.view(-1, 1, 1).expand(-1, M_chunk, -1)).squeeze(-1)
            agreement_scores_chunk.masked_fill_(~validity_mask_chunk, -torch.inf)

            # c. Top-K
            top_k_scores_chunk, top_k_indices_chunk = torch.topk(agreement_scores_chunk, k=K_chunk, dim=1)
            
            # d. 融合
            top_k_features_chunk = torch.gather(padded_features_chunk, 1, top_k_indices_chunk.unsqueeze(-1).expand(-1, -1, feature_dim))
            fusion_weights_chunk = F.softmax(top_k_scores_chunk, dim=1)
            fused_features_chunk = (top_k_features_chunk * fusion_weights_chunk.unsqueeze(-1)).sum(dim=1)

            # 5. 将计算好的块结果，放回最终的全场景特征张量中
            final_scene_features[chunk_indices] = fused_features_chunk

        scene_coords = scene_coords.to(counter.device)
        scene_true = scene_coords[counter != 0]
        scene_false = scene_coords[counter == 0]
        if scene_true.shape[0] > 0 and scene_false.shape[0] > 0:
            # print(f"Found {scene_true.shape[0]} seen points and {scene_false.shape[0]} unseen points. Performing KDTree fill-in.")
            
            flase_idx = torch.where(counter == 0)[0]
            true_idx = torch.where(counter != 0)[0]

            # KDTree 必须在 CPU 上运行
            kdtree = KDTree(scene_true.cpu().numpy())

            distances, indices = kdtree.query(scene_false.cpu().numpy(), k=1)

            match = true_idx[indices.flatten()]

            final_scene_features[flase_idx] = final_scene_features[match]

        elif scene_false.shape[0] > 0:
            # 这是一个边缘情况：如果一个点都没看到，我们无法填充
            # 此时，final_scene_features 很可能全是0，这本身是合理的
            print(f"Warning: No points were seen for this scene. KDTree fill-in skipped. All points are treated as unseen.")
        # PCA
        pca_color = get_pca_color(final_scene_features, brightness=1.2, center=True)
        # inverse back to original scale before grid sampling
        # point.inverse is acquired from the GirdSampling transform
        # pca_color_for_coords_3d = pca_color[point.inverse] 
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scene_coords)
        pcd.colors = o3d.utility.Vector3dVector(pca_color.cpu().detach().numpy())
        o3d.io.write_point_cloud("xdecoder_test/debug_projection_vis/2dpcaape.ply", pcd)
        # import pdb;pdb.set_trace()

        return (
            final_scene_features,
            text_features,
            logit_scale,
        )

    def get_sonata_features(self, batch_data):
        """
        使用冻结的Sonata模型为点云生成目标亲和力矩阵。
        """
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
        feats = scene_gauss_features[:, :3] # 假设使用原始颜色作为输入特征
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
            # 将数据移动到GPU
            for key in point.keys():
                if isinstance(point[key], torch.Tensor):
                    point[key] = point[key].cuda(non_blocking=True)

            # Sonata前向传播获取特征
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
        """
        高效、节省内存的对比学习样本采样函数。
        """
        device = F_sonata_full.device
        num_points = F_sonata_full.shape[0]
        
        # 1. 随机选择锚点 (保持不变)
        num_anchors = min(self.num_anchors_per_scene, num_points // 3)
        if num_anchors == 0:
            # 如果点数太少，无法采样，则返回空张量以避免崩溃
            return torch.empty(0, device=device, dtype=torch.long), \
                torch.empty(0, device=device, dtype=torch.long), \
                torch.empty(0, device=device, dtype=torch.long)

        anchor_indices = torch.randperm(num_points, device=device)[:num_anchors]
        F_anchor = F_sonata_full[anchor_indices]

        # 2. 为每个锚点找到最相似的正样本 (保持不变)
        # 计算锚点与所有其他点之间的相似度
        sim_matrix = torch.einsum('ad,pd->ap', F.normalize(F_anchor, p=2, dim=1), 
                                            F.normalize(F_sonata_full, p=2, dim=1))
        
        # 将锚点自身的位置相似度设为负无穷，避免选到自己
        sim_matrix.scatter_(1, anchor_indices.unsqueeze(1), float('-inf'))
        
        positive_indices = torch.argmax(sim_matrix, dim=1)

        # ==========================================================
        # ===== 核心修改：高效的负采样逻辑 (替换你的OOM代码) =====
        # ==========================================================
        
        # 3. 高效采样负样本
        all_indices = torch.arange(num_points, device=device)
        
        # a. 创建一个 (num_anchors, num_points) 的巨大布尔掩码
        #    其中 True 代表该点 *不能* 作为当前锚点的负样本
        #    (即，它等于锚点本身或其正样本)
        exclude_mask = (all_indices.unsqueeze(0) == anchor_indices.unsqueeze(1)) | \
                    (all_indices.unsqueeze(0) == positive_indices.unsqueeze(1))

        # b. 使用掩码将无效候选项的相似度设置为负无穷
        #    这样它们就不会在后续的采样中被选中
        sim_matrix_for_neg = sim_matrix
        sim_matrix_for_neg[exclude_mask] = float('-inf')

        # c. 从有效的候选项中，为每个锚点采样 K 个最不相似的 (hard negatives)
        #    torch.topk(..., largest=False) 可以高效地找到最小值
        #    这里的 "k" 就是我们需要的负样本数量
        _, negative_indices = torch.topk(
            sim_matrix_for_neg, 
            k=self.num_negatives_per_anchor, 
            largest=False, 
            dim=1
        )
        del sim_matrix_for_neg
        
        # negative_indices 的形状已经是 [num_anchors, num_negatives_per_anchor]
        
        return anchor_indices, positive_indices, negative_indices

    @torch.no_grad()
    def sample_contrastive_pairs_hybrid(self, F_sonata_full, neighbor_indices):
        """
        一个高效的、混合了宏观全局采样和微观局部分采样的对比学习样本挖掘函数。
        它只进行一次全局相似度计算，并复用其结果。

        Args:
            F_sonata_full (torch.Tensor): [N_points, D_feature], 场景中所有点的Sonata特征。
            neighbor_indices (torch.Tensor): [N_points, K], 每个点的K个空间最近邻的索引。
        """
        device = F_sonata_full.device
        num_points, K = neighbor_indices.shape
        
        # --- 1. 锚点选择 ---
        num_anchors = min(self.num_anchors_per_scene, num_points // 3)
        if num_anchors == 0:
            return torch.empty(0, 3, device=device, dtype=torch.long).T

        anchor_indices = torch.randperm(num_points, device=device)[:num_anchors]
        F_anchor = F_sonata_full[anchor_indices]
        
        # --- 2. 一次性全局相似度计算 (宏观挖掘的基础) ---
        # 这是主要的计算开销，但为全局正/负样本所必需。
        sim_matrix_global = torch.einsum('ad,pd->ap', F.normalize(F_anchor, p=2, dim=1),
                                        F.normalize(F_sonata_full, p=2, dim=1))

        # --- 3. 全局正样本挖掘 ---
        # 做法与纯宏观版本完全相同，以获得最强的正信号。
        sim_matrix_pos = sim_matrix_global.clone()
        sim_matrix_pos.scatter_(1, anchor_indices.unsqueeze(1), float('-inf')) # 排除自身
        positive_indices = torch.argmax(sim_matrix_pos, dim=1)
        del sim_matrix_pos

        # --- 4. 混合负样本挖掘 ---
        # 定义宏观和微观负样本的数量 (这是一个可调的超参数)
        N_macro = 48 
        N_micro = self.num_negatives_per_anchor - N_macro # e.g., 63 - 48 = 15
        
        # 4.a. 宏观困难负样本挖掘
        # 复用已计算的全局相似度矩阵，开销极小。
        sim_matrix_neg = sim_matrix_global
        # 排除锚点和正样本，防止它们被选为负样本
        exclude_mask = (torch.arange(num_points, device=device).unsqueeze(0) == anchor_indices.unsqueeze(1)) | \
                    (torch.arange(num_points, device=device).unsqueeze(0) == positive_indices.unsqueeze(1))
        sim_matrix_neg[exclude_mask] = float('inf') # 设置为极大值，topk(largest=False)会避开它们
        _, macro_negative_indices = torch.topk(sim_matrix_neg, k=N_macro, largest=False, dim=1)

        # 4.b. 微观困难负样本挖掘
        # 这是新的、高效的补充部分，计算开销非常小。
        # (1) 获取锚点的K个空间近邻的索引
        anchor_neighbors_indices = neighbor_indices[anchor_indices] # [num_anchors, K]

        # (2) 从已有的全局相似度矩阵中，直接“抽取”出锚点与其邻居的相似度，无需重新计算！
        # `torch.gather`可以高效地完成这个操作。
        sims_local = torch.gather(sim_matrix_global, 1, anchor_neighbors_indices) # [num_anchors, K]

        # (3) 在这K个局部相似度中，找到最不相似的 N_micro 个。
        # 这样可以有效避免“假负例”问题。
        _, hardest_indices_in_K = torch.topk(sims_local, k=N_micro, largest=False, dim=1)
        
        # (4) 使用这些“最差邻居”的索引，从邻居列表中提取出最终的微观负样本。
        micro_negative_indices = torch.gather(anchor_neighbors_indices, 1, hardest_indices_in_K)

        # --- 5. 合并与返回 ---
        negative_indices = torch.cat([macro_negative_indices, micro_negative_indices], dim=1)

        return anchor_indices, positive_indices, negative_indices

    def forward(self, batch_data):
        """
        完整"全部在线"的前向传播和损失计算。
        """
        # --- 1. 教师模型在线工作 (此部分不变) ---
        with torch.no_grad():
            if self.use_lseg:
                # F_xdecoder_full, _, _ = self.lift_lseg_features(batch_data)
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
            # <--- 新增开始 --->
            # 为了给混合采样函数提供“微观”信息，我们需要计算每个点的空间最近邻。
            # 这个操作每个batch只执行一次，使用faiss来保证效率。
            # 1. 从 batch_data 中获取原始点云坐标
            (scene_coords, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _) = batch_data

            # 2. 使用FAISS高效计算K-Nearest Neighbors
            # K值是一个超参数，这里我们选择96作为邻居数量
            K = 96 
            coords_np = scene_coords.cpu().numpy()
            index = faiss.IndexFlatL2(coords_np.shape[1])
            index.add(coords_np)
            # 搜索K+1个，因为第一个总是点本身
            _, neighbor_indices_with_self = index.search(coords_np, K + 1)
            
            # 移除点本身（第一列），并转换回Tensor
            neighbor_indices = torch.from_numpy(neighbor_indices_with_self[:, 1:]).to(self.device)
            # <--- 新增结束 --->
            anchor_idx, positive_idx, negative_idx = self.sample_contrastive_pairs_hybrid(F_sonata_full,neighbor_indices)
            
            # anchor_idx, positive_idx, negative_idx = self.sample_contrastive_pairs(F_sonata_full)
            
            # print("--- Running Fair Comparison Visualization ---")
            # vis_output_dir = "xdecoder_test/paper"
            # vis_prefix = "scene0247_01"
            # # [新逻辑] 创建一个映射回原始点云的 "Before" 特征版本
            # # F_xdecoder_full 代表了未经优化的、仅经过体素平均后的特征在原始点云上的样子
            # (
            #     scene_coords,
            #     scene_coords_3d,
            #     scene_inds_reconstruct,
            #     scene_label,
            #     ori_coords_3ds,
            #     coords_3ds,
            #     feat_3ds,
            #     gauss_featuress,
            #     labels_3ds,
            #     binary_label_3ds,
            #     label_2ds,
            #     imgs,
            #     x_labels,
            #     y_labels,
            #     mask_2ds,
            #     inds_reconstructs,
            #     unique_maps,
            #     mappings,
            #     captions,
            #     scene_gauss_features
            # ) = batch_data

            # # 可视化 1: 优化前的特征 @ 原始点云 (论文中的 "Before" / "图1b")
            # self.visualize_and_save_point_cloud(
            #     coords=scene_coords, # <--- 使用原始点云坐标
            #     features=F_xdecoder_full, # <--- 使用映射回点云的“优化前”特征
            #     filename=os.path.join(vis_output_dir, f"{vis_prefix}_before_refinement_on_points_pca.ply"),
            #     mode='kmeans'
            # )
            
            # # 可视化 2: 优化后的特征 @ 原始点云 (论文中的 "After" / "图2d")
            # self.visualize_and_save_point_cloud(
            #     coords=scene_coords, # <--- 使用相同的原始点云坐标
            #     features=F_sonata_full, # <--- 使用“优化后”的特征
            #     filename=os.path.join(vis_output_dir, f"{vis_prefix}_after_refinement_on_points_pca.ply"),
            #     mode='kmeans'
            # )
            
            # # 可视化 3: 原始RGB颜色 (论文中的 "Input" / "图1a")
            
            # original_rgb = scene_gauss_features[:, :3] 
            # self.visualize_and_save_point_cloud(
            #     coords=scene_coords,
            #     features=original_rgb,
            #     filename=os.path.join(vis_output_dir, f"{vis_prefix}_original_rgb.ply"),
            #     mode='rgb'
            # )
            # print("--- Visualization Complete ---")
            # import pdb;pdb.set_trace()

            del F_sonata_full

        
        # --- 2. 准备学生网络的输入 (全新的、高效的逻辑) ---
        # a. 解包Dataloader提供的数据
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
        # b. 收集所有采样点的索引 (在原始点云中的索引)
        all_sample_indices, point_to_batch_map = torch.unique(
            torch.cat([anchor_idx, positive_idx, negative_idx.flatten()]),
            return_inverse=True
        )
        
        # c. 使用`scene_inds_reconstruct`找到这些采样点对应的【体素索引】
        # 这是连接“点空间”和“体素空间”的桥梁！
        voxel_indices_for_samples = scene_inds_reconstruct.to(all_sample_indices.device)[all_sample_indices]
        
        # d. 找到我们需要处理的【唯一体素索引】以及从“采样点”到“唯一体素”的映射
        unique_voxel_indices, sample_to_voxel_map = torch.unique(
            voxel_indices_for_samples, return_inverse=True
        )

        # e. 为这些唯一的体素，聚合它们的输入特征
        # 我们使用 scatter_mean 来高效地平均落入同一个体素的点的特征
        # 1. 获取所有采样点的 F_xdecoder 特征
        features_for_samples = F_xdecoder_full[all_sample_indices]
        # 2. 按体素进行平均
        voxel_features_input = torch_scatter.scatter_mean(
            features_for_samples, sample_to_voxel_map, dim=0
        )

        # f. 获取这些唯一体素的坐标
        voxel_coords_input = scene_coords_3d.to(unique_voxel_indices.device)[unique_voxel_indices]

        # g. 创建【精简且高效】的稀疏张量，只包含我们关心的体素
        s_input_student = ME.SparseTensor(
            features=voxel_features_input,
            coordinates=ME.utils.batched_coordinates([voxel_coords_input]),
            device=voxel_features_input.device
        )
        del voxel_features_input, voxel_coords_input


        # # e. 为这些唯一的体素，聚合它们的【语义特征】
        # # 我们使用 scatter_mean 来高效地平均落入同一个体素的点的特征
        # # 1. 获取所有采样点的 F_xdecoder 特征
        # features_for_samples = F_xdecoder_full[all_sample_indices]
        # # 2. 按体素进行平均，得到【语义特征】
        # # [为了清晰，重命名变量]
        # voxel_semantic_features = torch_scatter.scatter_mean(
        #     features_for_samples, sample_to_voxel_map, dim=0
        # )

        # # [新代码] e_prime. 聚合【原始几何特征】并与语义特征拼接
        # # 1. 从 scene_gauss_features 提取原始几何特征 (颜色+法线)
        # # 根据您的描述，前3维是颜色(feats_in)，接下来3维是法线(gauss_normals_np)，共6维
        # # scene_gauss_features[:,:3] = scene_gauss_features[:,:3]/255.0
        # raw_geom_features_full = scene_gauss_features[:, :6].float().to(F_xdecoder_full.device)

        # # 2. 获取所有采样点的原始几何特征
        # geom_features_for_samples = raw_geom_features_full[all_sample_indices]
        # del raw_geom_features_full

        # # 3. 同样地，按体素对几何特征进行平均，得到【几何特征】
        # #    使用完全相同的 sample_to_voxel_map 来确保特征一一对应
        # voxel_geom_features = torch_scatter.scatter_mean(
        #     geom_features_for_samples, sample_to_voxel_map, dim=0
        # )
        # del geom_features_for_samples

        # # 4. 核心步骤：将语义特征和几何特征沿特征维度(dim=1)拼接
        # voxel_features_input = torch.cat([voxel_semantic_features, voxel_geom_features], dim=1)

        # del voxel_semantic_features, voxel_geom_features

        # # f. 获取这些唯一体素的坐标
        # voxel_coords_input = scene_coords_3d.to(unique_voxel_indices.device)[unique_voxel_indices]

        # # g. 创建【精简且高效】的稀疏张量，其特征现在是拼接后的新特征
        # s_input_student = ME.SparseTensor(
        #     features=voxel_features_input, # <--- 使用拼接后的新特征
        #     coordinates=ME.utils.batched_coordinates([voxel_coords_input]),
        #     device=voxel_features_input.device
        # )
        # del voxel_features_input, voxel_coords_input

        # --- 3. 学生模型学习与损失计算 ---
        # a. 学生模型前向传播，得到【体素级别】的预测
        s_output_student = self.affinity_student(s_input_student)
        
        # b. 将体素级别的预测，通过 `sample_to_voxel_map` 广播回【采样点级别】
        F_pred_on_samples = s_output_student.F[sample_to_voxel_map]
        del s_output_student
        
        # c. 归一化点级别的预测嵌入
        F_pred_on_samples_norm = F.normalize(F_pred_on_samples, p=2, dim=1)
        
        # d. 将点级别的预测，通过 `point_to_batch_map` 映射回锚点、正负样本的形状
        num_anchors = len(anchor_idx)
        F_pred_anchor = F_pred_on_samples_norm[point_to_batch_map[:num_anchors]]
        F_pred_positive = F_pred_on_samples_norm[point_to_batch_map[num_anchors:2*num_anchors]]
        F_pred_negative = F_pred_on_samples_norm[point_to_batch_map[2*num_anchors:]].reshape(num_anchors, self.num_negatives_per_anchor, -1)
        
        # e. 计算 InfoNCE 损失 (此部分逻辑不变)
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
        """
        [最终更新版] 一个通用的点云可视化并保存的辅助函数，新增K-Means模式。

        Args:
            coords (torch.Tensor or np.ndarray): 点云的XYZ坐标，形状 [N, 3]。
            features (torch.Tensor or np.ndarray): 用于着色的特征或颜色。
            filename (str): 输出文件的路径。
            mode (str): 着色模式。'pca', 'rgb', 或 'kmeans'。
            kmeans_k (int): 当 mode='kmeans' 时，指定的聚类数量。
        """
        import open3d as o3d
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        import os
        import numpy as np
        import matplotlib.cm as cm

        print(f"Visualizing with mode '{mode}' and saving to {filename}...")

        # --- 1. 确保数据在CPU上且为Numpy格式 ---
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        # --- 2. 根据模式计算颜色 ---
        if mode == 'pca':
            # a. 应用PCA将特征降到3维
            pca = PCA(n_components=3)
            # 确保特征是float64以获得最佳精度
            features_3d = pca.fit_transform(features.astype(np.float64))
            
            # b. 归一化到[0, 1]范围，以便作为颜色使用
            min_vals = features_3d.min(axis=0)
            max_vals = features_3d.max(axis=0)
            # 防止除以零
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1.0 
            colors = (features_3d - min_vals) / range_vals
        
        elif mode == 'kmeans':
            # a. 对高维特征应用K-Means聚类
            print(f"Running K-Means with k={kmeans_k}...")
            kmeans = KMeans(n_clusters=kmeans_k, random_state=0, n_init='auto')
            labels = kmeans.fit_predict(features.astype(np.float64))
            
            # b. 将离散的类别标签映射到鲜明的颜色
            # 使用一个高对比度的颜色映射表，例如 'tab20'
            # cmap函数返回的是RGBA，我们只需要RGB部分
            cmap = cm.get_cmap('tab20')
            colors = cmap(labels / (kmeans_k - 1))[:, :3] if kmeans_k > 1 else np.tile([0.5, 0.5, 0.5], (len(labels), 1))

        elif mode == 'rgb':
            # 假设输入已经是[0,1]范围的RGB，如果不是则进行处理
            if features.max() > 1.0:
                colors = features / 255.0
            else:
                colors = features
            assert colors.shape[1] == 3, "RGB模式要求特征维度为3"
        else:
            raise ValueError(f"未知的可视化模式: {mode}")

        # ... [创建和保存Open3D点云对象的代码保持不变] ...
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        output_dir = os.path.dirname(filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Successfully saved point cloud to {filename}")

    # 在计算affinity_weights后添加
    def visualize_affinity_weights(self, voxel_coords, neighbor_indices, affinity_weights, 
                              num_voxels, K, save_path):
        """可视化亲和力权重分布"""
        
        # 导入必要的库
        import open3d as o3d
        import matplotlib.pyplot as plt
        
        # 方案1: 为每个体素可视化其权重分布的统计特性
        weights_reshaped = affinity_weights.view(num_voxels, K)
        
        # 计算每个体素的权重集中度 (熵)
        entropy = -(weights_reshaped * torch.log(weights_reshaped + 1e-8)).sum(dim=1)
        max_entropy = torch.log(torch.tensor(K, dtype=torch.float32, device=entropy.device))  # 修复设备问题
        normalized_entropy = entropy / max_entropy  # [0,1], 0表示高度集中，1表示均匀分布
        
        # === DEBUG信息 ===
        print("=== AFFINITY WEIGHTS DEBUG ===")
        print(f"K (neighbors per voxel): {K}")
        print(f"Max possible entropy: {max_entropy:.3f}")
        print(f"Raw entropy range: [{entropy.min():.3f}, {entropy.max():.3f}]")
        print(f"Normalized entropy range: [{normalized_entropy.min():.3f}, {normalized_entropy.max():.3f}]")
        
        # 检查权重分布的统计信息
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
        
        # 检查有多少体素的权重是"集中"的
        concentrated_mask = normalized_entropy < 0.5
        print(f"Concentrated voxels (entropy < 0.5): {concentrated_mask.sum().item()}/{num_voxels} ({concentrated_mask.float().mean()*100:.1f}%)")
        
        very_concentrated_mask = normalized_entropy < 0.3
        print(f"Very concentrated voxels (entropy < 0.3): {very_concentrated_mask.sum().item()}/{num_voxels} ({very_concentrated_mask.float().mean()*100:.1f}%)")
        
        # === 解决方案2: 使用百分位数拉伸 ===
        # 修复设备问题
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
        
        # === 解决方案4: 直接用最大权重可视化 ===
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
        """在3D点云上可视化亲和力热力图"""
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import open3d as o3d
        from matplotlib.colors import LinearSegmentedColormap
        import torch
        
        # 准备数据
        affinity_2d = affinity_values.view(num_voxels, K).cpu().numpy()
        voxel_coords_np = voxel_coords.copy() if isinstance(voxel_coords, np.ndarray) else voxel_coords
        
        # === 方法1：基于平均亲和力的点云着色 ===
        # 计算每个体素的亲和力统计量
        avg_affinity = np.mean(affinity_2d, axis=1)  # 平均亲和力
        max_affinity = np.max(affinity_2d, axis=1)   # 最大亲和力
        std_affinity = np.std(affinity_2d, axis=1)   # 亲和力标准差（多样性）
        
        # 归一化到[0,1]用于着色
        def normalize_for_color(values):
            return (values - values.min()) / (values.max() - values.min() + 1e-8)
        
        avg_affinity_norm = normalize_for_color(avg_affinity)
        max_affinity_norm = normalize_for_color(max_affinity)
        std_affinity_norm = normalize_for_color(std_affinity)
        
        # 创建不同的颜色方案
        colormap_options = {
            'avg': ('viridis', avg_affinity_norm, 'Average Affinity'),
            'max': ('plasma', max_affinity_norm, 'Max Affinity'),
            'diversity': ('coolwarm', std_affinity_norm, 'Affinity Diversity (Std)')
        }
        
        for color_type, (cmap_name, values, title) in colormap_options.items():
            # 使用matplotlib colormap生成颜色
            cmap = plt.cm.get_cmap(cmap_name)
            colors = cmap(values)[:, :3]  # 只取RGB，去掉alpha
            
            # 创建点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(voxel_coords_np)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 保存点云
            o3d.io.write_point_cloud(f"{output_path}_3d_{color_type}_affinity.ply", pcd)
        
        # === 方法2：创建传统2D热力图用于对比 ===
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
        
        # === 方法3：创建亲和力分布的3D可视化 ===
        fig = plt.figure(figsize=(20, 5))
        
        # 子图1：平均亲和力分布
        ax1 = fig.add_subplot(141)
        ax1.hist(avg_affinity, bins=50, alpha=0.7, color='blue')
        ax1.set_title('Average Affinity Distribution')
        ax1.set_xlabel('Average Affinity')
        ax1.set_ylabel('Frequency')
        
        # 子图2：最大亲和力分布
        ax2 = fig.add_subplot(142)
        ax2.hist(max_affinity, bins=50, alpha=0.7, color='red')
        ax2.set_title('Max Affinity Distribution')
        ax2.set_xlabel('Max Affinity')
        ax2.set_ylabel('Frequency')
        
        # 子图3：亲和力多样性分布
        ax3 = fig.add_subplot(143)
        ax3.hist(std_affinity, bins=50, alpha=0.7, color='green')
        ax3.set_title('Affinity Diversity (Std) Distribution')
        ax3.set_xlabel('Standard Deviation')
        ax3.set_ylabel('Frequency')
        
        # 子图4：亲和力与空间位置的关系（以Z坐标为例）
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
        
        # === 方法4：创建局部邻域可视化（选择几个代表性体素） ===
        # 选择亲和力最高、最低和中等的几个体素进行详细可视化
        high_affinity_idx = np.argsort(avg_affinity)[-5:]  # 最高的5个
        low_affinity_idx = np.argsort(avg_affinity)[:5]    # 最低的5个
        med_affinity_idx = np.argsort(np.abs(avg_affinity - np.median(avg_affinity)))[:5]  # 中等的5个
        
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
        """可视化代表性体素的局部邻域"""
        
        import open3d as o3d
        import numpy as np
        import matplotlib.pyplot as plt
        
        affinity_2d = affinity_values.view(num_voxels, K).cpu().numpy()
        
        for i, voxel_idx in enumerate(representative_indices[:6]):  # 只可视化前6个
            # 获取中心体素和其邻居
            center_coord = voxel_coords[voxel_idx]
            neighbors_idx = neighbor_indices[voxel_idx].cpu().numpy()
            neighbor_coords = voxel_coords[neighbors_idx]
            neighbor_affinities = affinity_2d[voxel_idx]
            
            # 创建点云
            pcd = o3d.geometry.PointCloud()
            
            # 所有点的坐标
            all_coords = np.vstack([center_coord.reshape(1, -1), neighbor_coords])
            pcd.points = o3d.utility.Vector3dVector(all_coords)
            
            # 颜色设置：中心点为红色，邻居根据亲和力着色
            colors = np.zeros((len(all_coords), 3))
            colors[0] = [1, 0, 0]  # 中心点红色
            
            # 邻居点颜色：亲和力高->绿色，亲和力低->蓝色
            cmap = plt.cm.get_cmap('RdYlGn')
            normalized_affinities = (neighbor_affinities - neighbor_affinities.min()) / \
                                (neighbor_affinities.max() - neighbor_affinities.min() + 1e-8)
            colors[1:] = cmap(normalized_affinities)[:, :3]
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 添加连接线显示邻域关系
            lines = [[0, j+1] for j in range(len(neighbors_idx))]
            line_set = o3d.geometry.LineSet()
            line_set.points = pcd.points
            line_set.lines = o3d.utility.Vector2iVector(lines)
            
            # 线条颜色也根据亲和力设置
            line_colors = cmap(normalized_affinities)[:, :3]
            line_set.colors = o3d.utility.Vector3dVector(line_colors)
            
            # 保存点云和线集
            o3d.io.write_point_cloud(f"{output_path}_neighborhood_{i}_points.ply", pcd)
            o3d.io.write_line_set(f"{output_path}_neighborhood_{i}_lines.ply", line_set)
            
            print(f"Voxel {voxel_idx}: avg_affinity={affinity_2d[voxel_idx].mean():.4f}, "
                f"max_affinity={affinity_2d[voxel_idx].max():.4f}")


    @torch.no_grad()
    def evaluate_scene(self, batch_data,vis_prefix = "scene0695_00"):
        """
        在评估（推理）期间，返回经过学生网络增强后的最终场景特征。
        此方法模仿CLIP-DINOiser的评估策略，不使用Sonata教师模型。
        """
        # --- 0. 定义超参数 ---
        # K: 为每个体素寻找多少个最近邻进行池化
        K = 96 
        # affinity_sharpen_factor: 亲和力锐化系数 (类似温度系数的倒数)
        # 值越大，权重分布越集中于最相似的邻居
        affinity_sharpen_factor = 20
        # 1. 设置学生网络为评估模式
        self.affinity_student.eval()

        # 2. 获取基础语义特征：运行X-Decoder并提升至3D
        # 这是我们想要“净化”的原始特征
        if self.use_lseg:
            F_xdecoder_full,text_features,logit_scale = self.lift_lseg_features(batch_data)
            # return {
            #     "scene_features": F_xdecoder_full.to(text_features.device), # 形状: [N_total_points, D_feature]
            #     "text_features": text_features,   # 形状: [N_classes, D_feature]
            #     "logit_scale": logit_scale        # 一个标量
            # }
        elif self.use_ape:
            F_xdecoder_full, text_features, logit_scale = self.lift_ape_features(batch_data)
            # return {
            #     "scene_features": F_xdecoder_full.to(text_features.device), # 形状: [N_total_points, D_feature]
            #     "text_features": text_features,   # 形状: [N_classes, D_feature]
            #     "logit_scale": logit_scale        # 一个标量
            # }
        else:
            F_xdecoder_full,text_features,logit_scale = self.lift_xdecoder_features(batch_data)
            # return {
            #     "scene_features": F_xdecoder_full.to(text_features.device), # 形状: [N_total_points, D_feature]
            #     "text_features": text_features,   # 形状: [N_classes, D_feature]
            #     "logit_scale": logit_scale        # 一个标量
            # }

        # 3. 准备学生网络的输入：使用整个场景的体素化数据
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
        
        # # 使用整个场景的体素化坐标和聚合后的特征作为输入
        # # 注意：这里我们假设 F_xdecoder_full 是 N_orig x D 的
        # # 我们需要先将其聚合到体素级别
        # voxel_features_input = torch_scatter.scatter_mean(
        #     F_xdecoder_full.to(scene_inds_reconstruct.device), scene_inds_reconstruct, dim=0
        # ).to(self.device)

        # # del F_xdecoder_full
        # # torch.cuda.empty_cache() # 立即执行清理

        # # 创建稀疏张量，只包含非空体素
        # s_input_student = ME.SparseTensor(
        #     features=voxel_features_input,
        #     coordinates=ME.utils.batched_coordinates([scene_coords_3d.to(self.device)]), # 假设 scene_coords_3d 已经是唯一的体素坐标
        #     device=voxel_features_input.device
        # )

        # --- 新的特征聚合逻辑，模仿forward函数 ---
        # a. 聚合【语义特征】到体素级别
        voxel_semantic_features = torch_scatter.scatter_mean(
            F_xdecoder_full.to(scene_inds_reconstruct.device), 
            scene_inds_reconstruct, 
            dim=0
        )

        # b. 聚合【原始几何特征】到体素级别
        # 从 scene_gauss_features 提取前9维（颜色+法线）
        raw_geom_features_full = scene_gauss_features[:, :6].float().to(F_xdecoder_full.device)
        voxel_geom_features = torch_scatter.scatter_mean(
            raw_geom_features_full.to(scene_inds_reconstruct.device), 
            scene_inds_reconstruct, 
            dim=0
        )

        # c. 拼接语义特征和几何特征
        voxel_features_input = torch.cat([voxel_semantic_features, voxel_geom_features], dim=1).to(self.device)

        # 清理中间变量
        del F_xdecoder_full, voxel_semantic_features, voxel_geom_features, raw_geom_features_full
        torch.cuda.empty_cache()

        # 创建稀疏张量，现在使用拼接后的特征
        s_input_student = ME.SparseTensor(
            features=voxel_features_input,  # <--- 现在是语义+几何的拼接特征
            coordinates=ME.utils.batched_coordinates([scene_coords_3d.to(self.device)]),
            device=voxel_features_input.device
        )
        
        # --- 2. 运行学生网络，得到几何嵌入 ---
        s_output_student = self.affinity_student(s_input_student)
        F_pred_embed = F.normalize(s_output_student.F, p=2, dim=1)

        del s_output_student
        
        # --- 3. K-近邻搜索 (在体素空间中) ---
        voxel_coords = s_input_student.C[:, 1:].contiguous().float().cpu().numpy()
        num_voxels = voxel_coords.shape[0]
        
        # a. 构建faiss索引
        index = faiss.IndexFlatL2(voxel_coords.shape[1])
        index.add(voxel_coords)
        
        # b. 搜索 K+1 个邻居 (因为第一个总是体素本身)
        _, neighbor_indices = index.search(voxel_coords, K + 1)
        neighbor_indices = torch.from_numpy(neighbor_indices[:, 1:]).to(F_pred_embed.device) # 移除自己
        
        # --- 4. 构建稀疏局部亲和力矩阵 ---
        # a. 准备索引
        row_indices = torch.arange(num_voxels, device=F_pred_embed.device).repeat_interleave(K)
        col_indices = neighbor_indices.flatten()

        del index        
        # b. 高效计算局部亲和力值
        # 提取中心点和邻居点的嵌入
        center_embeds = F_pred_embed.repeat_interleave(K, dim=0)
        neighbor_embeds = F_pred_embed[col_indices]
        
        # 计算亲和力 (余弦相似度)
        affinity_values = torch.einsum('bd,bd->b', center_embeds, neighbor_embeds)
        # # 在你的代码中添加调用
        # vis_output_dir = "xdecoder_test/paper"
        # vis_prefix = "oriplyscene0247_01"
        # # 主要可视化函数
        # affinity_stats = self.plot_affinity_heatmap_3d(
        #     voxel_coords=voxel_coords, 
        #     affinity_values=affinity_values, 
        #     num_voxels=num_voxels, 
        #     K=K, 
        #     output_path=os.path.join(vis_output_dir, vis_prefix)
        # )

        # # 可选：详细的局部邻域可视化
        # self.visualize_local_neighborhoods(
        #     voxel_coords=voxel_coords,
        #     neighbor_indices=neighbor_indices,
        #     affinity_values=affinity_values,
        #     representative_indices=affinity_stats['representative_indices'],
        #     num_voxels=num_voxels,
        #     K=K,
        #     output_path=os.path.join(vis_output_dir, vis_prefix)
        # )
        # import pdb;pdb.set_trace()

        # affinity_values_reshaped = affinity_values.view(num_voxels, K)
        # print("\n=== RAW AFFINITY VALUES (pre-softmax logits) DEBUG ===")
        # print(f"Affinity values mean: {affinity_values_reshaped.mean():.4f}")
        # print(f"Affinity values std: {affinity_values_reshaped.std():.4f}")
        # print(f"Affinity values range: [{affinity_values_reshaped.min():.4f}, {affinity_values_reshaped.max():.4f}]")

        # # 绘制直方图，看得更清楚
        # import matplotlib.pyplot as plt
        # plt.hist(affinity_values.cpu().numpy(), bins=100)
        # plt.title("Distribution of Raw Affinity Values")
        # plt.xlabel("Cosine Similarity")
        # plt.ylabel("Frequency")
        # plt.savefig("xdecoder_test/debug_projection_vis/affinity_values_histogram.png")
        # plt.close()
        # print("Saved affinity values histogram to affinity_values_histogram.png")

        # # 在计算 F_pred_embed 之后
        # # 随机采样1000个嵌入向量
        # rand_indices = torch.randperm(F_pred_embed.shape[0])[:1000]
        # sample_embeds = F_pred_embed[rand_indices]

        # # 计算这1000个向量两两之间的余弦相似度
        # cos_sim_matrix = torch.einsum('ad,bd->ab', sample_embeds, sample_embeds)

        # # 移除对角线（自身与自身的相似度为1）
        # cos_sim_matrix.fill_diagonal_(0)

        # print("\n=== STUDENT EMBEDDING (`F_pred_embed`) DEBUG ===")
        # print(f"Pairwise similarity of 1000 random embeds:")
        # print(f"Mean: {cos_sim_matrix.mean():.4f}")
        # print(f"Std: {cos_sim_matrix.std():.4f}")
        # print(f"Range: [{cos_sim_matrix.min():.4f}, {cos_sim_matrix.max():.4f}]")
        # import pdb;pdb.set_trace()

        del F_pred_embed, center_embeds, neighbor_embeds
        
        # # c. 对每个点的K个邻居亲和力进行Softmax锐化，得到归一化的权重
        # affinity_values_2d = affinity_values.view(num_voxels, K)
        # # === 新增：Z-Score归一化，强制拉伸局部对比度 ===
        # mean = affinity_values_2d.mean(dim=1, keepdim=True)
        # std = affinity_values_2d.std(dim=1, keepdim=True)
        # # 加上一个很小的epsilon防止除以零
        # affinity_values_normalized = (affinity_values_2d - mean) / (std + 1e-6) 

        # # 使用归一化后的值进行锐化和Softmax
        # affinity_weights = F.softmax(
        #     affinity_values_normalized * affinity_sharpen_factor, # 注意：factor可能需要重新设为较小的值，比如1.0或2.0
        #     dim=1
        # ).flatten()
        affinity_weights = F.softmax(
            affinity_values.view(num_voxels, K) * affinity_sharpen_factor, dim=1
        ).flatten()

        # # 直接使用余弦相似度，但保持正确的张量形状
        # affinity_values_2d = affinity_values.view(num_voxels, K)
        # affinity_weights_2d = torch.clamp(affinity_values_2d, min=0)  # 只保留正相似度
        # affinity_weights_2d = affinity_weights_2d / (affinity_weights_2d.sum(dim=1, keepdim=True) + 1e-8)
        # affinity_weights = affinity_weights_2d.flatten()

        # # 方案2: 使用ReLU激活函数
        # affinity_values_2d = affinity_values.view(num_voxels, K)
        # affinity_weights_2d = torch.relu(affinity_values_2d)  # ReLU激活，去除负相似度
        # affinity_weights_2d = affinity_weights_2d / (affinity_weights_2d.sum(dim=1, keepdim=True) + 1e-8)
        # affinity_weights = affinity_weights_2d.flatten()

        del affinity_values        
        # d. 构建最终的、归一化后的稀疏亲和力矩阵
        A_sparse_voxel_norm = torch.sparse_coo_tensor(
            indices=torch.stack([row_indices, col_indices]),
            values=affinity_weights,
            size=(num_voxels, num_voxels)
        )

        # # 在你的代码中添加调用
        # vis_output_dir = "xdecoder_test/paper"
        # vis_prefix = "scene0247_01"
        # self.visualize_affinity_weights(
        #     voxel_coords, neighbor_indices, affinity_weights, 
        #     num_voxels, K, os.path.join(vis_output_dir, vis_prefix)
        # )
        # import pdb;pdb.set_trace()

        del affinity_weights, row_indices, col_indices        
        # --- 5. 执行稀疏引导池化 ---
        # 使用稀疏矩阵乘法，对“源特征”进行加权平均
        F_refined_voxel = torch.sparse.mm(A_sparse_voxel_norm, voxel_features_input)

        for _ in range(18):
            F_refined_voxel = torch.sparse.mm(A_sparse_voxel_norm, F_refined_voxel)

        # alpha = 0.2
        # F_refined_voxel = alpha * F_refined_voxel + (1 - alpha) * voxel_features_input
        
        # --- 6. 将净化后的体素特征，映射回原始点云 ---
        F_refined_points = F_refined_voxel[scene_inds_reconstruct][:, :512]

        # print("--- Running Fair Comparison Visualization ---")
        # vis_output_dir = "xdecoder_test/paper"
        # # [新逻辑] 创建一个映射回原始点云的 "Before" 特征版本
        # # F_xdecoder_full 代表了未经优化的、仅经过体素平均后的特征在原始点云上的样子
        # (
        #     scene_coords,
        #     scene_coords_3d,
        #     scene_inds_reconstruct,
        #     scene_label,
        #     ori_coords_3ds,
        #     coords_3ds,
        #     feat_3ds,
        #     gauss_featuress,
        #     labels_3ds,
        #     binary_label_3ds,
        #     label_2ds,
        #     imgs,
        #     x_labels,
        #     y_labels,
        #     mask_2ds,
        #     inds_reconstructs,
        #     unique_maps,
        #     mappings,
        #     captions,
        #     scene_gauss_features
        # ) = batch_data

        # # 可视化 1: 优化前的特征 @ 原始点云 
        # self.visualize_and_save_point_cloud(
        #     coords=scene_coords, # <--- 使用原始点云坐标
        #     features=F_xdecoder_full, # <--- 使用映射回点云的“优化前”特征
        #     filename=os.path.join(vis_output_dir, f"oriply{vis_prefix}_before.ply"),
        #     mode='kmeans'
        # )
        
        # # 可视化 2: 优化后的特征 @ 原始点云
        # self.visualize_and_save_point_cloud(
        #     coords=scene_coords, # <--- 使用相同的原始点云坐标
        #     features=F_refined_points, # <--- 使用“优化后”的特征
        #     filename=os.path.join(vis_output_dir, f"oriply{vis_prefix}_after.ply"),
        #     mode='kmeans'
        # )
        
        # # 可视化 3: 原始RGB颜色 (论文中的 "Input" / "图1a")
        
        # original_rgb = scene_gauss_features[:, :3] 
        # self.visualize_and_save_point_cloud(
        #     coords=scene_coords,
        #     features=original_rgb,
        #     filename=os.path.join(vis_output_dir, f"oriply{vis_prefix}_original_rgb.ply"),
        #     mode='rgb'
        # )
        # print("--- Visualization Complete ---")
        # import pdb;pdb.set_trace()

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
        
        # 返回最终的、经过几何增强的特征
        return {
            "scene_features": F_refined_points.to(text_features.device), # 形状: [N_total_points, D_feature]
            "text_features": text_features,   # 形状: [N_classes, D_feature]
            "logit_scale": logit_scale        # 一个标量
        }