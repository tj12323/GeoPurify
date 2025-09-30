#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
import torch
import numpy as np
from typing import NamedTuple
from plyfile import PlyData, PlyElement
from models.utils.mapping_util import vertex_normal

from models.utils.graphics_utils import (
    getWorld2View2,
    focal2fov,
    fov2focal,
    getWorld2View2,
    getProjectionMatrix,
    BasicPointCloud,
)
from models.utils.sh_utils import RGB2SH


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    intrinsics: np.array


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


class ImageInfo(NamedTuple):
    id: int = None
    R: np.ndarray = None
    T: np.ndarray = None
    fovx: float = None
    fovy: float = None
    width: int = None
    height: int = None
    view_matrix: torch.Tensor = None
    projection_matrix: torch.Tensor = None
    camera_center: torch.Tensor = None
    image_path: str = None
    ply_path: str = None


def read_cameras(path, ply_path, extensions=[".png", ".jpg"]):
    cam_infos = []

    with open(os.path.join(path, "transforms_train.json")) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            for extension in extensions:
                cam_name = os.path.join(path, frame["file_path"] + extension)
                if os.path.exists(cam_name):
                    break

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            if np.isinf(c2w).any():
                continue
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = cam_name

            cam_infos.append(
                ImageInfo(
                    id=idx,
                    R=R,
                    T=T,
                    fovx=fovx,
                    image_path=image_path,
                    ply_path=ply_path,
                )
            )

    return cam_infos


def load_gaussian_ply(path, feature_type="all"):
    plydata = PlyData.read(path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros(
        (
            xyz.shape[0],
            3,
        )
    )
    features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))

    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    if feature_type == "all":
        return xyz, np.concatenate([opacities, features_dc, features_extra, scales, rots], axis=-1)
    elif feature_type == "color":
        return xyz, np.concatenate([features_dc, features_extra], axis=-1)
    # return xyz, np.concatenate([features_dc, features_extra], axis=-1)


def load_dynamic_gaussian_npz(path, t):
    params = dict(np.load(path))
    params = {k: np.array(v).astype(np.float32) for k, v in params.items()}

    xyz = params["means3D"][t]
    features_dc = RGB2SH(params["rgb_colors"][t])
    rots = params["unnorm_rotations"][t]
    opacities = params["logit_opacities"]
    scales = params["log_scales"]
    features_extra = np.zeros((xyz.shape[0], 45), dtype=np.float32)

    return xyz, np.concatenate([opacities, features_dc, features_extra, scales, rots], axis=-1)
    # return xyz, np.concatenate([features_dc, features_extra], axis=-1)


def load_point_ply(path,islabel=False):
    plydata = PlyData.read(path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )

    rgb = np.stack(
        (
            np.asarray(plydata.elements[0]["red"]),
            np.asarray(plydata.elements[0]["green"]),
            np.asarray(plydata.elements[0]["blue"]),
        ),
        axis=1,
    )
    if islabel:
        label = np.asarray(plydata.elements[0]["label"])
    else:
        label = None

    faces = np.stack(plydata["face"].data["vertex_indices"], axis=0)
    normal = vertex_normal(xyz, faces).astype(np.float32)

    return xyz, rgb, label, normal


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    # normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=None)

def fetchPth(path):
    """
    从pth文件中加载点云数据，返回与fetchPly相同格式的BasicPointCloud对象
    
    Args:
        path: pth文件路径
        
    Returns:
        BasicPointCloud: 包含points, colors, normals的点云对象
    """
    try:
        # 加载pth文件，根据process_one_scene函数，保存的格式是(coords, colors, normal, vertex_labels)
        data = torch.load(path, map_location='cpu')
        
        if len(data) != 4:
            raise ValueError(f"Expected 4 elements in pth file, got {len(data)}")
        
        coords, colors, normals, vertex_labels = data
        
        # 转换为numpy数组（如果是torch tensor）
        if hasattr(coords, 'numpy'):
            coords = coords.numpy()
        if hasattr(colors, 'numpy'):
            colors = colors.numpy()
        if hasattr(normals, 'numpy'):
            normals = normals.numpy()
        
        # 确保数据类型正确
        positions = coords.astype(np.float32)
        
        # 颜色处理：从[-1, 1]范围转换到[0, 1]范围
        # 因为保存时是: colors = v[:, -3:] / 127.5 - 1
        # 所以加载时需要逆转换: (colors + 1) * 127.5 / 255
        colors_normalized = ((colors + 1) * 127.5 / 255.0).astype(np.float32)
        
        # 确保颜色值在[0, 1]范围内
        colors_normalized = np.clip(colors_normalized, 0.0, 1.0)
        
        # 法向量处理
        if normals is not None:
            normals = normals.astype(np.float32)
        
        # 检查数据维度
        if positions.shape[1] != 3:
            raise ValueError(f"Expected 3D coordinates, got shape {positions.shape}")
        if colors_normalized.shape[1] != 3:
            raise ValueError(f"Expected RGB colors, got shape {colors_normalized.shape}")
        if normals is not None and normals.shape[1] != 3:
            raise ValueError(f"Expected 3D normals, got shape {normals.shape}")
        
        # 检查点数是否一致
        num_points = positions.shape[0]
        if colors_normalized.shape[0] != num_points:
            raise ValueError(f"Mismatch in number of points: positions {num_points}, colors {colors_normalized.shape[0]}")
        if normals is not None and normals.shape[0] != num_points:
            raise ValueError(f"Mismatch in number of points: positions {num_points}, normals {normals.shape[0]}")
        
        print(f"Loaded {num_points} points from {path}")
        # print(f"Position range: [{positions.min():.3f}, {positions.max():.3f}]")
        # print(f"Color range: [{colors_normalized.min():.3f}, {colors_normalized.max():.3f}]")
        
        return BasicPointCloud(
            points=positions, 
            colors=colors_normalized, 
            normals=normals
        )
        
    except FileNotFoundError:
        raise FileNotFoundError(f"PTH file not found: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load PTH file {path}: {str(e)}")



def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
