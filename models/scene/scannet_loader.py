import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

from models.utils.sh_utils import SH2RGB
from models.utils.graphics_utils import BasicPointCloud, focal2fov, fov2focal
from models.utils.dataset_utils import SceneInfo, CameraInfo, getNerfppNorm, storePly, fetchPly,fetchPth


def readScanNetInfo(path, white_background, eval, llffhold=8, extensions=[".png", ".jpg"]):
    path = Path(path)
    image_dir = path / "color"
    pose_dir = path / "pose"
    image_sorted = list(sorted(image_dir.iterdir(), key=lambda x: int(x.name.split(".")[0])))
    pose_sorted = list(sorted(pose_dir.iterdir(), key=lambda x: int(x.name.split(".")[0])))

    cam_infos = []
    K = np.loadtxt(os.path.join(path, "intrinsic/intrinsic_color.txt"))
    first_img = np.array(Image.open(image_sorted[0]).convert("RGBA"))
    width, height = first_img.shape[1], first_img.shape[0]

    fovx = focal2fov(K[0, 0], K[0, 2] * 2)
    fovy = focal2fov(K[1, 1], K[1, 2] * 2)

    i = 0
    for img, pose in zip(image_sorted, pose_sorted):
        i += 1
        idx = int(img.name.split(".")[0])
        idx_pose = int(pose.name.split(".")[0])
        if idx != idx_pose:
            print(f"Image {idx} and pose {idx_pose} do not match. Skipping...")
        if idx % 20 != 0:
            continue
        c2w = np.loadtxt(pose)
        c2w = np.array(c2w).reshape(4, 4).astype(np.float32)
        # ScanNet pose use COLMAP coordinates (Y down, Z forward), so no need to flip the axis
        # c2w[:3, 1:3] *= -1
        # We cannot accept files directly, as some of the poses are invalid
        if np.isinf(c2w).any():
            continue

        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = img
        image_name = Path(img).stem

        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=fovy,
                FovX=fovx,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
                intrinsics=K,
            )
        )

    nerf_normalization = getNerfppNorm(cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    pcd = fetchPly(ply_path)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info

def readMatterport3DInfo(path, ply_path,white_background, eval, llffhold=8, extensions=[".jpg", ".png"], 
                         bbox=None, use_bbox_filter=True,istest = True):
    """
    读取Matterport3D数据集信息，包含重要的场景边界筛选功能
    注意：此函数要求必须存在points3d.ply点云文件
    
    Args:
        path: Matterport3D数据集根目录路径
        white_background: 背景颜色设置
        eval: 是否为评估模式
        llffhold: 测试集划分间隔
        extensions: 支持的图像文件扩展名
        sample_interval: 图像采样间隔
        bbox: 场景边界框 [bbox_l, bbox_h]，如果为None则从点云自动计算
        use_bbox_filter: 是否使用边界框筛选相机
    """
    path = Path(path)
    image_dir = path / "color"
    pose_dir = path / "pose" 
    intrinsic_dir = path / "intrinsic"
    
    # 首先检查点云文件是否存在
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Required point cloud file not found: {ply_path}")
    
    # 加载点云
    try:
        pcd = fetchPth(ply_path)
        # print(f"Loaded point cloud from {ply_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load point cloud from {ply_path}: {e}")
    
    # 获取所有图像文件并排序
    image_files = []
    for ext in extensions:
        image_files.extend(list(image_dir.glob(f"*{ext}")))
    
    image_sorted = sorted(image_files, key=lambda x: x.stem)
    
    if len(image_sorted) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    # 读取第一张图像获取分辨率
    first_img = np.array(Image.open(image_sorted[0]).convert("RGBA"))
    width, height = first_img.shape[1], first_img.shape[0]
    
    # 第一步：读取所有有效的相机数据
    all_cam_data = []
    # print("Loading all camera data...")
    
    for img_path in image_sorted:  # 应用采样
        img_name = img_path.stem
        pose_path = pose_dir / f"{img_name}.txt"
        intrinsic_path = intrinsic_dir / f"{img_name}.txt"
        
        # 检查文件存在性
        if not pose_path.exists() or not intrinsic_path.exists():
            continue
            
        try:
            # 读取位姿和内参
            extrinsic = np.loadtxt(pose_path).astype(np.float32)
            if extrinsic.shape != (4, 4):
                extrinsic = extrinsic.reshape(4, 4)
            
            K = np.loadtxt(intrinsic_path).astype(np.float32)
            if K.shape != (3, 3):
                K = K.reshape(3, 3)
            
            # 检查有效性
            if np.isinf(extrinsic).any() or np.isnan(extrinsic).any():
                continue
            if np.isinf(K).any() or np.isnan(K).any():
                continue
                
            # 存储相机数据
            all_cam_data.append({
                'img_path': img_path,
                'img_name': img_name,
                'extrinsic': extrinsic,
                'intrinsic': K
            })
            
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            continue
    
    if len(all_cam_data) == 0:
        raise ValueError("No valid camera data found")
    
    # print(f"Loaded {len(all_cam_data)} valid camera poses")
    
    # 第二步：计算场景边界框（如果未提供）
    if bbox is None and use_bbox_filter:
        # print("Computing scene bounding box from point cloud...")
        bbox_l = np.min(pcd.points, axis=0)
        bbox_h = np.max(pcd.points, axis=0)
        bbox = [bbox_l, bbox_h]  # 重要：更新bbox变量
        # print(f"Bounding box from point cloud: {bbox_l} to {bbox_h}")
    elif bbox is not None:
        bbox_l, bbox_h = bbox[0], bbox[1]
        # print(f"Using provided bounding box: {bbox_l} to {bbox_h}")
    
    # 第三步：场景内筛选（核心逻辑，对应get_matterport_camera_data中的ind_in_scene）
    cam_infos = []
    
    if use_bbox_filter and bbox is not None:
        print("Filtering cameras within scene bounds...")
        
        # 提取所有相机位置
        cam_positions = []
        for cam_data in all_cam_data:
            w2c = cam_data['extrinsic']
            cam_pos = w2c[:3, 3]  # 相机位置
            cam_positions.append(cam_pos)
        
        cam_positions = np.array(cam_positions)
        
        # 场景内筛选逻辑（完全对应原代码的ind_in_scene）
        ind_in_scene = (cam_positions[:, 0] > bbox_l[0]) & (cam_positions[:, 0] < bbox_h[0]) & \
                       (cam_positions[:, 1] > bbox_l[1]) & (cam_positions[:, 1] < bbox_h[1]) & \
                       (cam_positions[:, 2] > bbox_l[2]) & (cam_positions[:, 2] < bbox_h[2])
        
        cameras_in_scene = [all_cam_data[i] for i in range(len(all_cam_data)) if ind_in_scene[i]]
        
        print(f"Cameras in scene: {len(cameras_in_scene)} / {len(all_cam_data)}")
        
        # 如果场景内没有相机（对应原代码的特殊处理）
        if len(cameras_in_scene) == 0:
            if istest:  # 类似原代码的test模式处理
                print('No views inside scene bounds, taking the nearest 100 cameras')
                # 计算边界框中心
                centroid = (bbox_l + bbox_h) / 2
                # 计算所有相机到中心的距离
                dist_centroid = np.linalg.norm(cam_positions - centroid, axis=-1)
                # 选择最近的100个相机（或所有相机如果少于100个）
                num_nearest = min(100, len(all_cam_data))
                ind_nearest = np.argsort(dist_centroid)[:num_nearest]
                cameras_in_scene = [all_cam_data[i] for i in ind_nearest]
                print(f"Selected {len(cameras_in_scene)} nearest cameras")
            else:
                # 训练模式下，如果没有场景内的相机，这可能是个问题
                print("Warning: No cameras found within scene bounds for training!")
                cameras_in_scene = all_cam_data  # 使用所有相机
    else:
        # 不使用边界框筛选
        cameras_in_scene = all_cam_data
        print(f"Using all {len(cameras_in_scene)} cameras (no bbox filtering)")
    
    # cameras_sampled = cameras_in_scene[::4]
    # 第四步：创建CameraInfo对象
    for i, cam_data in enumerate(cameras_in_scene):
        w2c = cam_data['extrinsic']
        K = cam_data['intrinsic']
        
        # 提取旋转和平移
        R = np.transpose(w2c[:3, :3])  # 转置以适配CUDA GLM
        T = w2c[:3, 3]
        
        # 计算FOV
        fx, fy = K[0, 0], K[1, 1]
        fovx = focal2fov(fx, width)
        fovy = focal2fov(fy, height)
        
        cam_info = CameraInfo(
            uid=i,
            R=R,
            T=T,
            FovY=fovy,
            FovX=fovx,
            image_path=str(cam_data['img_path']),
            image_name=cam_data['img_name'],
            width=width,
            height=height,
            intrinsics=K,
        )
        
        cam_infos.append(cam_info)
    
    # print(f"Created {len(cam_infos)} camera info objects")
    
    # 计算NeRF标准化参数
    nerf_normalization = getNerfppNorm(cam_infos)
    
    # 划分训练集和测试集
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    
    # print(f"Final: Train cameras: {len(train_cam_infos)}, Test cameras: {len(test_cam_infos)}")
    
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    
    return scene_info
