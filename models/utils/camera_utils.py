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

import numpy as np
from PIL import Image
from models.scene.camera import Camera
from models.utils.general_utils import PILtoTorch
from models.utils.graphics_utils import fov2focal, focal2fov

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    bg = np.array([1, 1, 1]) if args.white_background else np.array([0, 0, 0])

    image = Image.open(cam_info.image_path)
    im_data = np.array(image.convert("RGBA"))
    norm_data = im_data / 255.0
    arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
    image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

    orig_w, orig_h = image.size

    if args.downscale_ratio == -1:
        if orig_w > 1600:
            global WARNED
            if not WARNED:
                print(
                    "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                    "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                )
                WARNED = True
            global_down = orig_w / 1600
        else:
            global_down = 1
    else:
        global_down = 1 / args.downscale_ratio

    scale = float(global_down) * float(resolution_scale)
    resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(image, resolution)
    # print("resized_image_rgb.shape", resized_image_rgb.shape)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        image_path=cam_info.image_path,
        uid=id,
        device=args.device,
    )


def get_camera_from_directions(scene_camera, R, T):
    return Camera(
        colmap_id=scene_camera.colmap_id,
        R=R,
        T=T,
        FoVx=scene_camera.FoVx,
        FoVy=scene_camera.FoVy,
        image=scene_camera.original_image,
        gt_alpha_mask=None,
        image_name=scene_camera.image_name,
        image_path=scene_camera.image_path,
        uid=scene_camera.uid,
        device=scene_camera.data_device,
    )


def get_camera_viser(scene_camera, R, T, fovy, wh_ratio):
    fovx = focal2fov(fov2focal(fovy, 1000), 1000 * wh_ratio)
    return Camera(
        colmap_id=scene_camera.colmap_id,
        R=R,
        T=T,
        FoVx=fovx,
        FoVy=fovy,
        image=scene_camera.original_image,
        gt_alpha_mask=None,
        image_name=scene_camera.image_name,
        image_path=scene_camera.image_path,
        uid=scene_camera.uid,
        device=scene_camera.data_device,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
