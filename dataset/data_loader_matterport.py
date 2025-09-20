from glob import glob
from os.path import join
import imageio.v2 as imageio
import imageio.v3 as imageiov3
import os
import re
import torch
import numpy as np
import SharedArray as SA
from tqdm import tqdm
import cv2
import pandas as pd
import json
import numpy as np
from dataset.point_loader import Point3DLoader
from models.utils.mapping_util import getMapping

from models.gaussians.utils.system_utils import searchForMaxIteration
from models.gaussians.scene import Scene
from models.gaussians.model import GaussianModel, render, render_chn
from models.utils.fusion_util import PointCloudToImageMapper,PointCloudToImageMappermatterport
from models.gaussians.utils.dataset_utils import load_point_ply
from models.gaussians.model.render_utils import get_mapped_label, get_text_features, render_palette
from models.gaussians.model.gaussian_label import assign_labels_knn_with_features, preprocess_labels, assign_entity_indices_knn
from models.gaussians.utils.sh_utils import RGB2SH, SH2RGB
import pickle
import copy
import pdb
import collections
from sklearn.neighbors import NearestNeighbors
import open3d as o3d

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
        # gaussians=None,
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
        # self.gaussians = gaussians
        self.scene_config = scene_config

        self.datapath_2d = datapath_prefix_2d
        # self.point2img_mapper = getMapping()
        self.val_keep = val_keep
        # with open(caption_path, "r") as f:
        #     self.captions_view = json.load(f)
        # self.caption = {}
        # for key in ["view", "entity"]:
        #     self.caption[key] = {}
        # with open(caption_path, "r") as f:
        #     self.caption["view"] = json.load(f)
        # with open(entity_path, "r") as f:
        #     self.caption["entity"] = json.load(f)
        self.data_ids_mapping = {}
        self.data_ids_mapping_all = {}
        label_2d_id = label_2d

        self.label_3d_id = label_2d
        if self.split in ["val", "test"]:
            self.label_2d_id = label_2d_id

        else:

            self.label_2d_id = [
                label_2d_id[category]
                for category in self.category_split["base_category"]
            ]
        for i, id in enumerate(self.label_2d_id):

            self.data_ids_mapping.update({id: i})
        for i, id in enumerate(self.label_3d_id):

            self.data_ids_mapping_all.update({id: i})

        if len(self.data_paths) == 0:
            raise Exception("0 file is loaded in the feature loader.")
        self.epoch = None

        # Modified code to limit dataset:
        num_files = len(self.data_paths)
        if specific_ids is not None:
            print(f"Loading files based on specific_ids: {specific_ids}")
            self.data_paths = [path for path in self.data_paths 
                   if os.path.basename(path).replace('.pth', '') in specific_ids]
            # self.data_paths = [path for path in self.data_paths if any(id in path for id in specific_ids)]
            print(f"Loaded {len(self.data_paths)} files based on specific_ids")
        # elif load_percentage < 1.0:
        #     num_files_to_load = int(num_files * load_percentage)
        #     if num_files_to_load == 0:
        #       num_files_to_load = 1
        #     self.data_paths = np.random.choice(self.data_paths, num_files_to_load, replace=False).tolist()
        #     print(f"Loaded {len(self.data_paths)} files with {load_percentage * 100}% of the dataset")
        
        self.split = split

        # 新增逻辑：创建 (场景, 视图) 样本列表
        self.samples = []
        print("Preprocessing dataset to create (scene, view) samples...")
        # 假设 self.data_paths 是你的场景路径列表
        # import pdb;pdb.set_trace()
        for scene_data_path in tqdm(self.data_paths, desc="Loading scene configs"):
            scene_name = os.path.basename(scene_data_path).split('.pth')[0]
            
            # 这里只加载必要的场景配置来获取视图，避免加载完整的点云
            # 注意：这部分需要根据你的 Scene 类进行调整
            temp_scene_config = copy.deepcopy(self.scene_config.scene)
            
            # scene_pattern = r"scene\d{4}_\d{2}"
            # if re.search(scene_pattern, temp_scene_config.scene_path):
            #     temp_scene_config.scene_path = re.sub(scene_pattern, '', temp_scene_config.scene_path)
            temp_scene_config.scene_path = os.path.join(temp_scene_config.scene_path, scene_name.split('_')[0])
            try:
                # 只加载相机/视图信息
                scene = Scene(temp_scene_config, scene_data_path,resolution_scales=[1.0]) 
                views = scene.getTrainCameras(scale=1.0)
                
                for i, view in enumerate(views):
                # for i in range(5):
                    # view = views[i]
                    # 每个样本是一个元组，包含处理该视图所需的信息
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
        
        # # ======================= DEBUG STEP 1 =======================
        # print("\n[DEBUG] Verifying samples in Dataset.__init__...")
        # scene_counts = defaultdict(int)
        # for sample in self.samples:
        #     scene_counts[sample['scene_name']] += 1

        # print(f"Found {len(self.samples)} total samples across {len(scene_counts)} unique scenes.")
        # print("Sample counts per scene:")
        # # 打印前10个场景的样本数，以检查是否符合预期（例如，每个场景5个视图）
        # for i, (scene, count) in enumerate(scene_counts.items()):
        #     if i >= 10: break
        #     print(f"  - Scene '{scene}': {count} views")
        # print("====================================================\n")
        # # ==========================================================

    def __len__(self):
        # 长度是总视图数，而不是场景数
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
        # # ======================= DEBUG STEP 3 =======================
        # # 使用 index_long 来看原始请求的索引
        # print(f"  [GETITEM CALLED] with index: {index}")
        # print(f"  [GETITEM PROCESSING] Scene: '{sample_info['scene_name']}'")
        # # ==========================================================
        
        # --- 步骤1: 加载场景级数据 (使用缓存) ---
        if scene_name in self.scene_cache:
            scene_data = self.scene_cache[scene_name]
            ori_scene_path = join(self.datapath_2d, scene_name)
            img_dirs = sorted(
                glob(join(ori_scene_path, "color/*")), key=lambda x: int(os.path.basename(x)[:-4])
            )
        else:
            locs_in, feats_in, original_scan_normals,labels_in = torch.load(scene_data_path)
            # _, original_rgb, _ = torch.load(scene_data_path)
            rgb_min, rgb_max = feats_in.min(), feats_in.max()
            if rgb_min >= -1.0 and rgb_max <= 1.0:
                feats_in = (feats_in.astype(np.float64) + 1.0) / 2.0
            
            scene_name = os.path.basename(scene_data_path).split('.pth')[0]
            print(f"scene_name: {scene_name}")

            # scene_pattern = r"scene\d{4}_\d{2}"
            # # 检查 scene_config.scene.scene_path 是否包含类似的场景路径
            # if re.search(scene_pattern, self.scene_config.scene.scene_path):
            #     self.scene_config.scene.scene_path = re.sub(scene_pattern, '', self.scene_config.scene.scene_path)

            # scene_path = os.path.join(self.scene_config.scene.scene_path, scene_name)
            # self.scene_config.scene.scene_path = scene_path

            # original_ply_path = os.path.join(self.scene_config.scene.scene_path, "points3d.labels.ply")
            # original_xyz, _, original_labels, original_scan_normals = load_point_ply(original_ply_path,islabel=True)
            point_features = np.concatenate([feats_in, original_scan_normals], axis=1)
            # labels_in = labels_in.cpu().numpy()
        
            labels_in[labels_in == -100] = self.category_split.ignore_category[-1]
            labels_in[labels_in == 255] = self.category_split.ignore_category[-1]
            labels_in_clone = labels_in.copy()

            if self.split in ["val", "test"]:
                pass

            else:

                indices_to_replace = self.category_split["novel_category"] + [
                    self.category_split.ignore_category[0]
                ]

                labels_in[
                    np.isin(labels_in, indices_to_replace)
                ] = self.category_split.ignore_category[-1]
                for i, replace in enumerate(indices_to_replace):
                    labels_in[labels_in > replace - i] -= 1

            if np.isscalar(feats_in) and feats_in == 0:

                feats_in = np.zeros_like(locs_in)
            else:
                # feats_in = (feats_in + 1.0) * 127.5
                feats_in = feats_in * 255.0

            # if "scannet_3d" in self.dataset_name:
            #     scene_name = scene_data_path[:-15].split("/")[-1]
            # else:
            #     scene_name = scene_data_path[:-4].split("/")[-1]

            # ori_scene_path = join(self.datapath_2d, scene_name)
            # img_dirs = sorted(
            #     glob(join(ori_scene_path, "color/*")), key=lambda x: int(os.path.basename(x)[:-4])
            # )
            scene_data = {
                "locs_in": locs_in,
                "labels_in": labels_in, # 计算好的标签
                "point_features": point_features,
            }
            self.scene_cache[scene_name] = scene_data
        
        # 从缓存中获取数据
        locs_in = scene_data["locs_in"]
        labels_in = scene_data["labels_in"]
        labels_in_clone = labels_in.copy()
        point_features = scene_data["point_features"]

        if view:
            # ii=16
            img_dir = str(view.image_path)

            # 获取相机位姿
            R = view.R
            T = view.T
            pose = np.eye(4)
            pose[:3, :3] = R.transpose()
            pose[:3, 3] = T            
            
            # 使用 Camera 对象中的图像
            img = view.original_image.permute(1, 2, 0).cpu().numpy() * 255
            img = img.astype(np.uint8) # 转成np.uint8类型，以便后续cv2使用
            # load depth
            depth_dir = img_dir.replace('color', 'depth')
            _, img_type, yaw_id = img_dir.split('/')[-1].split('_')
            depth_dir = depth_dir[:-8] + 'd'+img_type[1] + '_' + yaw_id[0] + '.png'
            depth = imageiov3.imread(depth_dir) / self.scene_config.fusion.depth_scale            

        # if np.any(np.isinf(pose)):
        #     print(f"Invalid pose detected in {posepath}")
        #     continue
        # calculate the 3d-2d mapping based on the depth
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

            if np.sum(mask) < 400 or valid_point_num < 10 or np.sum(mask) > 65000:
                return None
        else:
            if (
                np.sum(mask) < 400
                or valid_point_num < 10
                or np.sum(mask) > self.val_keep
            ):
                return None

        img = cv2.resize(img, (self.scene_config.fusion.img_dim[0], self.scene_config.fusion.img_dim[1]))

        # image_idx = os.path.basename(img_dir)[0:-4]

        # if self.scannet200:
        #     labelimg_dir = os.path.join(ori_scene_path, "label_200", image_idx + ".png")
        #     label_2d = imageio.imread(
        #         labelimg_dir
        #     ).astype(np.int32)
        #     # label_2d = imageio.imread(
        #     #     img_dir.replace("color", "label_200").replace(".jpg", ".png")
        #     # ).astype(np.int32)
        # else:
        #     labelimg_dir = os.path.join(ori_scene_path, "label", image_idx + ".png")
        #     label_2d = imageio.imread(
        #         labelimg_dir
        #     ).astype(np.int32)

        entity = None

        vectorized_map = np.vectorize(
            lambda value: self.data_ids_mapping.get(value, value)
        )
        # label_2d[~np.isin(label_2d, self.label_2d_id)] = 255
        # label_2d = vectorized_map(label_2d)

        # if self.split in ["val", "test"]:
        #     pass

        # else:
        #     label_2d[label_2d == 255] = len(self.category_split["base_category"])

        # label_2d = cv2.resize(label_2d, (self.scene_config.fusion.img_dim[0], self.scene_config.fusion.img_dim[1]), interpolation=cv2.INTER_NEAREST)
        label_2d = np.zeros((self.scene_config.fusion.img_dim[1], self.scene_config.fusion.img_dim[0]), dtype=np.uint8)

        img = torch.from_numpy(img).float()

        locals_3d = self.prevoxel_transforms(locals_3d) if self.aug else locals_3d

        locs, feats, _, inds_reconstruct = self.voxelizer.voxelize(
            locals_3d, feature_3d, label_3d
        )
        # print(f"locs min: {locs.min()}, max: {locs.max()}")
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
        # locs_in = torch.from_numpy(locs_in).float()
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
        
        # # ======================= DEBUG STEP 2A =======================
        # print("\n[DEBUG] Verifying groups in SceneBatchSampler.__init__...")
        # print(f"Sampler created with {len(self.scenes)} scene groups.")
        # # 打印前5组的详细信息
        # for i, scene_name in enumerate(self.scenes):
        #     if i >= 5: break
        #     indices = self.scene_to_indices[scene_name]
        #     print(f"  - Group '{scene_name}': {len(indices)} indices -> {indices[:10]}...") # 打印前10个索引
        # print("========================================================\n")
        # # ===========================================================

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.scenes)
        
        for scene_name in self.scenes:
            indices = self.scene_to_indices[scene_name]
            # # ======================= DEBUG STEP 2B =======================
            # print(f"\n[SAMPLER YIELDING] Scene: '{scene_name}', Num Views: {len(indices)}, Indices: {indices}")
            # # ===========================================================
            yield indices

    def __len__(self):
        return len(self.scenes)

import torch
from torch.utils.data._utils.collate import default_collate

import torch
from torch.utils.data._utils.collate import default_collate

def scene_based_collate_fn(batch):
    # 1. 过滤掉无效样本
    # print(f"[COLLATE_FN] Received batch with {len(batch)} samples")
    
    # # 统计None的数量和位置
    # none_count = sum(1 for item in batch if item is None)
    # valid_indices = [i for i, item in enumerate(batch) if item is not None]
    # none_indices = [i for i, item in enumerate(batch) if item is None]
    
    # print(f"[COLLATE_FN] Found {none_count} None samples at indices: {none_indices[:10]}...")  # 只显示前10个
    # print(f"[COLLATE_FN] Valid samples at indices: {valid_indices[:10]}...")
    
    batch = [item for item in batch if item is not None]
    # print(f"[COLLATE_FN] After filtering: {len(batch)} valid samples")
    if not batch:
        return None

    # 2. 使用 zip(*batch) 优雅地解包所有20项数据
    #    变量名后的 "_b" 代表这是一个包含批次中所有样本的元组 (tuple)
    (
        locs_in_b, scene_coords_3d_b, scene_inds_reconstruct_b, scene_label_b,
        locals_3d_b, coords_3ds_b, feat_3ds_b, gauss_featuress_b, labels_3ds_b,
        binary_label_3ds_b, label_2ds_b, imgs_b, x_labels_b, y_labels_b,
        mask_2ds_b, inds_reconstructs_b, unique_maps_b, mappings_b,
        captions_b, scene_gauss_features_b
    ) = list(zip(*batch))

    # 3. 采纳您的核心逻辑，处理批次索引和重建索引
    #    (将元组转为列表以进行原地修改)
    coords_list = list(coords_3ds_b)
    locals_3d_list = list(locals_3d_b)
    inds_recons_list = list(inds_reconstructs_b)
    mask_2ds_list = list(mask_2ds_b)
    mask_all = torch.cat(mask_2ds_list, dim=0)  # shape [N_total], dtype=torch.bool

    # 2) 用 list comprehension + full() 生成每段对应的 batch idx，然后拼接
    batch_idx = torch.cat([
        torch.full(
            (m.shape[0],),      # 长度 = 当前 sample 的点数
            fill_value=i,       # 用样本编号 i 填充
            dtype=torch.long,   # 整型
            device=m.device     # 保持和 mask 同设备
        )
        for i, m in enumerate(mask_2ds_list)
    ], dim=0)  # shape [N_total], dtype=torch.long

    # 3) （可选）把它们合成一个 [N_total, 2] 的“索引＋mask 值”张量
    #    如果你只关心批次号，其实 batch_idx 就够了；否则可以再做一步：
    mask_all_int = mask_all.to(torch.int)            # bool→0/1
    batch_and_mask = torch.stack([batch_idx,mask_all_int], dim=1)
    
    accmulate_points_num = 0
    for i in range(len(batch)):
        # a. 为每个视图的点云设置正确的批次索引 (0, 1, 2...)
        coords_list[i][:, 0] = i
        locals_3d_list[i][:, 0] = i
        
        # b. 累加偏移，修正重建索引 (非常关键的修正!)
        inds_recons_list[i] += accmulate_points_num
        accmulate_points_num += coords_list[i].shape[0]

    # 4. 组合最终输出，对每一项应用正确的处理方式 (cat, stack, or take first)
    return (
        # --- 场景级数据: 只从批次中取第一个 ---
        locs_in_b[0],
        scene_coords_3d_b[0],
        scene_inds_reconstruct_b[0],
        scene_label_b[0],
        
        # --- 视图级，可变尺寸: 拼接 (cat) ---
        torch.cat(locals_3d_list),      # 已处理批次索引
        torch.cat(coords_list),         # 已处理批次索引
        torch.cat(feat_3ds_b),
        torch.cat(gauss_featuress_b),
        torch.cat(labels_3ds_b),
        torch.cat(binary_label_3ds_b),
        
        # --- 视图级，固定/场景尺寸: 堆叠 (stack) ---
        torch.stack(label_2ds_b, dim=0),
        torch.stack(imgs_b, dim=0),
        
        # --- 视图级，可变尺寸: 拼接 (cat) ---
        torch.cat(x_labels_b),
        torch.cat(y_labels_b),
        
        # --- 视图级，固定/场景尺寸: 堆叠 (stack) ---
        # torch.cat(mask_2ds_b),
        batch_and_mask,
        
        # --- 视图级，可变尺寸: 拼接 (cat) ---
        torch.cat(inds_recons_list),    # 已处理索引偏移
        torch.cat(unique_maps_b), # 假设unique_map尺寸固定
        torch.cat(mappings_b),
        
        # --- 特殊 & 场景级数据 ---
        captions_b,                     # captions保持为元组
        scene_gauss_features_b[0]
    )