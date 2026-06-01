from glob import glob
from os.path import join
import imageio.v3 as imageiov3
import os
import torch
import numpy as np
from tqdm import tqdm
import cv2
import numpy as np
from dataset.point_loader import Point3DLoader

from models.scene import Scene
from models.utils.fusion_util import PointCloudToImageMapper,PointCloudToImageMappermatterport
import copy

class ScannetLoaderFull(Point3DLoader):
    def __init__(
        self,
        datapath_prefix,
        datapath_prefix_2d,
        label_2d,
        category_split,
        scannet200=False,
        val_keep=10000000,
        caption_path="data/caption/caption_view_scannet_vit-gpt2-image-captioning_.json",
        entity_path="data/caption/caption_entity_scannet_vit-gpt2-image-captioning_.json",
        voxel_size=0.05,
        split="train",
        aug=False,
        memcache_init=False,
        identifier=7791,
        loop=1,
        eval_all=False,
        input_color=False,
        specific_ids=None, # Add this parameter to select files based on name

        scene_config=None,
    ):
        super().__init__(
            datapath_prefix=datapath_prefix,
            voxel_size=voxel_size,
            split=split,
            aug=aug,
            memcache_init=memcache_init,
            identifier=identifier,
            loop=loop,
            eval_all=eval_all,
            input_color=input_color,
        )
        self.scannet200 = scannet200
        self.aug = aug
        self.input_color = input_color
        self.category_split = category_split

        self.scene_config = scene_config

        self.datapath_2d = datapath_prefix_2d

        self.val_keep = val_keep









        self.data_ids_mapping = {}
        self.data_ids_mapping_all = {}
        label_2d_id = label_2d

        self.label_3d_id = label_2d
        for i, id in enumerate(self.label_3d_id):

            self.data_ids_mapping_all.update({id: i})

        if len(self.data_paths) == 0:
            raise Exception("0 file is loaded in the feature loader.")
        self.epoch = None

        num_files = len(self.data_paths)
        if specific_ids is not None:
            print(f"Loading files based on specific_ids: {specific_ids}")
            self.data_paths = [path for path in self.data_paths 
                   if os.path.basename(path).replace('.pth', '') in specific_ids]

            print(f"Loaded {len(self.data_paths)} files based on specific_ids")
        
        self.split = split

        self.samples = []
        print("Preprocessing dataset to create (scene, view) samples...")

        for scene_data_path in tqdm(self.data_paths, desc="Loading scene configs"):
            scene_name = os.path.basename(scene_data_path).split('.pth')[0]
            temp_scene_config = copy.deepcopy(self.scene_config.scene)
            temp_scene_config.scene_path = os.path.join(temp_scene_config.scene_path, scene_name.split('_')[0])
            try:
                scene = Scene(temp_scene_config, scene_data_path,resolution_scales=[1.0]) 
                views = scene.getTrainCameras(scale=1.0)
                
                for i, view in enumerate(views):
                    self.samples.append({
                        "scene_data_path": scene_data_path,
                        "scene_name": scene_name,
                        "view": view,
                        "view_idx": i, # 视图的索引
                        "intrinsics": views.camera_info[i].intrinsics,
                    })
            except Exception as e:
                print(f"Warning: Could not process scene {scene_name}. Error: {e}. Skipping.")

        print(f"Dataset created with {len(self.samples)} total views.")

        self.scene_cache = {} # 用于缓存已加载的场景数据

    def __len__(self):

        return len(self.samples) * self.loop

    def read_bytes(self, path):

        with open(path, "rb") as f:
            file_bytes = f.read()
        return file_bytes

    def __getitem__(self, index_long):
        
        index = index_long % len(self.samples) 
        sample_info = self.samples[index]

        scene_name = sample_info["scene_name"]
        scene_data_path = sample_info["scene_data_path"]
        view = sample_info["view"]       

        if scene_name in self.scene_cache:
            scene_data = self.scene_cache[scene_name]
            ori_scene_path = join(self.datapath_2d, scene_name)
            img_dirs = sorted(
                glob(join(ori_scene_path, "color/*")), key=lambda x: int(os.path.basename(x)[:-4])
            )
        else:
            locs_in, feats_in, original_scan_normals,labels_in = torch.load(scene_data_path)

            rgb_min, rgb_max = feats_in.min(), feats_in.max()
            if rgb_min >= -1.0 and rgb_max <= 1.0:
                feats_in = (feats_in.astype(np.float64) + 1.0) / 2.0
            
            scene_name = os.path.basename(scene_data_path).split('.pth')[0]
            print(f"scene_name: {scene_name}")
            point_features = np.concatenate([feats_in, original_scan_normals], axis=1)
        
            labels_in[labels_in == -100] = self.category_split.ignore_category[-1]
            labels_in[labels_in == 255] = self.category_split.ignore_category[-1]
            labels_in_clone = labels_in.copy()

            if self.split in ["val", "test"]:
                pass

            if np.isscalar(feats_in) and feats_in == 0:
                feats_in = np.zeros_like(locs_in)
            else:
                feats_in = feats_in * 255.0

            scene_data = {
                "locs_in": locs_in,
                "labels_in": labels_in, # 计算好的标签
                "point_features": point_features,
            }
            self.scene_cache[scene_name] = scene_data        

        locs_in = scene_data["locs_in"]
        labels_in = scene_data["labels_in"]
        labels_in_clone = labels_in.copy()
        point_features = scene_data["point_features"]
        if view:
            img_dir = str(view.image_path)

            R = view.R
            T = view.T
            pose = np.eye(4)
            pose[:3, :3] = R.transpose()
            pose[:3, 3] = T
            img = view.original_image.permute(1, 2, 0).cpu().numpy() * 255
            img = img.astype(np.uint8) # 转成np.uint8类型，以便后续cv2使用

            depth_dir = img_dir.replace('color', 'depth')
            _, img_type, yaw_id = img_dir.split('/')[-1].split('_')
            depth_dir = depth_dir[:-8] + 'd'+img_type[1] + '_' + yaw_id[0] + '.png'
            depth = imageiov3.imread(depth_dir) / self.scene_config.fusion.depth_scale

        with torch.no_grad():
            mapper = PointCloudToImageMappermatterport(
                self.scene_config.fusion.img_dim,
                self.scene_config.fusion.visibility_threshold,
                self.scene_config.fusion.cut_boundary,
            )
            mapping = np.ones([locs_in.shape[0], 4], dtype=int)
            mapping[:, 1:4] = mapper.compute_mapping(
                camera_to_world = view.world_view_transform.cpu().numpy().T,
                coords = locs_in,
                depth = depth,
                intrinsic = sample_info["intrinsics"]
            )
        if mapping[:, 3].sum() == 0:  # no points corresponds to this image, skip
            return None

        mask = mapping[:, 3]
        label_3d = labels_in[mask == 1].copy()
        feature_3d = point_features[mask == 1].copy()
        locals_3d = locs_in[mask == 1].copy()
        label_3d_clone = labels_in_clone[mask == 1].copy()
        unique_map = mapping.copy()
        zero_rows = np.all(mapping != 0, axis=1)
        mapping = mapping[zero_rows]

        binary_label = label_3d_clone

        binary_label[
            np.isin(label_3d_clone, self.category_split["base_category"])
        ] = 1
        binary_label[
            np.isin(label_3d_clone, self.category_split["novel_category"])
        ] = 0

        binary_label = torch.from_numpy(binary_label).float()
        valid_point_num = np.sum(
            ~np.isin(binary_label, np.array(self.category_split.ignore_category))
        )

        if self.split == "train":
            if np.sum(mask) < 400 or np.sum(mask) > 65000:
                return None
        else:
            if (
                np.sum(mask) < 400
                or np.sum(mask) > self.val_keep
            ):
                return None

        img = cv2.resize(img, (self.scene_config.fusion.img_dim[0], self.scene_config.fusion.img_dim[1]))
        entity = None

        label_2d = np.zeros((self.scene_config.fusion.img_dim[1], self.scene_config.fusion.img_dim[0]), dtype=np.uint8)

        img = torch.from_numpy(img).float()

        locals_3d = self.prevoxel_transforms(locals_3d) if self.aug else locals_3d

        locs, feats, _, inds_reconstruct = self.voxelizer.voxelize(
            locals_3d, feature_3d, label_3d
        )
        feats = feats[:,:3]

        labels = label_3d

        if self.aug:
            locs, feats, labels = self.input_transforms(locs, feats, labels)
        coords = torch.from_numpy(locs).int()
        coords = torch.cat(
            (torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1
        )
        if self.input_color:
            feats = torch.from_numpy(feats).float() / 255.0
        else:

            feats = torch.ones(coords.shape[0], 3)

        labels = torch.from_numpy(labels).long()

        label_2d = torch.from_numpy(label_2d).long()

        x_label = mapping[:, 1][mapping[:, 1] != 0]
        y_label = mapping[:, 2][mapping[:, 2] != 0]
        x_label = torch.from_numpy(x_label).long()
        y_label = torch.from_numpy(y_label).long()

        mask_2d = torch.from_numpy(mask).bool()

        inds_reconstruct = torch.from_numpy(inds_reconstruct).long()
        unique_map = torch.from_numpy(unique_map).long()
        locals_3d = torch.from_numpy(locals_3d).float()
        locals_3d = torch.cat(
            (torch.ones(locals_3d.shape[0], 1, dtype=torch.float), locals_3d), dim=1
        )
        mapping = torch.from_numpy(mapping)
        
        locs, _, _, scene_inds_reconstruct = self.voxelizer.voxelize(
            locs_in, point_features, labels_in
        )
        locs = torch.from_numpy(locs).float()
        scene_inds_reconstruct = torch.from_numpy(scene_inds_reconstruct).long()
        locs_in = torch.from_numpy(locs_in).float()
        labels_in = torch.from_numpy(labels_in).long()
        feature_3d = torch.from_numpy(feature_3d).float()
        point_features = torch.from_numpy(point_features).float()
        return (
            locs_in,
            locs,
            scene_inds_reconstruct,
            labels_in,
            locals_3d,
            coords,
            feats,
            feature_3d,
            labels,
            binary_label,
            label_2d,
            img,
            x_label,
            y_label,
            mask_2d,
            inds_reconstruct,
            unique_map,
            mapping,
            None,
            point_features
        )

import torch
from torch.utils.data import Sampler
from collections import defaultdict
import random

class SceneBatchSampler(Sampler):
    def __init__(self, samples_list, shuffle=True):
        self.samples_list = samples_list
        self.shuffle = shuffle
        
        self.scene_to_indices = defaultdict(list)
        for idx, sample in enumerate(self.samples_list):
            self.scene_to_indices[sample['scene_name']].append(idx)
            
        self.scenes = list(self.scene_to_indices.keys())

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.scenes)
        
        for scene_name in self.scenes:
            indices = self.scene_to_indices[scene_name]
            yield indices

    def __len__(self):
        return len(self.scenes)

import torch
from torch.utils.data._utils.collate import default_collate

import torch
from torch.utils.data._utils.collate import default_collate

def scene_based_collate_fn(batch):    
    batch = [item for item in batch if item is not None]

    if not batch:
        return None

    (
        locs_in_b, scene_coords_3d_b, scene_inds_reconstruct_b, scene_label_b,
        locals_3d_b, coords_3ds_b, feat_3ds_b, gauss_featuress_b, labels_3ds_b,
        binary_label_3ds_b, label_2ds_b, imgs_b, x_labels_b, y_labels_b,
        mask_2ds_b, inds_reconstructs_b, unique_maps_b, mappings_b,
        captions_b, scene_gauss_features_b
    ) = list(zip(*batch))

    coords_list = list(coords_3ds_b)
    locals_3d_list = list(locals_3d_b)
    inds_recons_list = list(inds_reconstructs_b)
    mask_2ds_list = list(mask_2ds_b)
    mask_all = torch.cat(mask_2ds_list, dim=0)  # shape [N_total], dtype=torch.bool

    batch_idx = torch.cat([
        torch.full(
            (m.shape[0],),      # 长度 = 当前 sample 的点数
            fill_value=i,       # 用样本编号 i 填充
            dtype=torch.long,   # 整型
            device=m.device     # 保持和 mask 同设备
        )
        for i, m in enumerate(mask_2ds_list)
    ], dim=0)  # shape [N_total], dtype=torch.long


    mask_all_int = mask_all.to(torch.int)            # bool→0/1
    batch_and_mask = torch.stack([batch_idx,mask_all_int], dim=1)
    
    accmulate_points_num = 0
    for i in range(len(batch)):
        coords_list[i][:, 0] = i
        locals_3d_list[i][:, 0] = i
        inds_recons_list[i] += accmulate_points_num
        accmulate_points_num += coords_list[i].shape[0]

    return (
        locs_in_b[0],
        scene_coords_3d_b[0],
        scene_inds_reconstruct_b[0],
        scene_label_b[0],        

        torch.cat(locals_3d_list),      # 已处理批次索引
        torch.cat(coords_list),         # 已处理批次索引
        torch.cat(feat_3ds_b),
        torch.cat(gauss_featuress_b),
        torch.cat(labels_3ds_b),
        torch.cat(binary_label_3ds_b),        

        torch.stack(label_2ds_b, dim=0),
        torch.stack(imgs_b, dim=0),        

        torch.cat(x_labels_b),
        torch.cat(y_labels_b),
        batch_and_mask,       

        torch.cat(inds_recons_list),    # 已处理索引偏移
        torch.cat(unique_maps_b), # 假设unique_map尺寸固定
        torch.cat(mappings_b),       

        captions_b,                     # captions保持为元组
        scene_gauss_features_b[0]
    )