# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import sys
import logging

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

from PIL import Image
import numpy as np
np.random.seed(1)

import torch
from torchvision import transforms

from utils.arguments import load_opt_command

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.visualizer import Visualizer
from utils.distributed import init_distributed
import pdb
import matplotlib.pyplot as plt
def get_color_palette(num_classes=20):
    """Generate a color palette for semantic classes (e.g., ScanNet 20 classes)."""
    # Using matplotlib's tab20 colormap as an example
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20)[:3] for i in range(num_classes)]  # RGB tuples (0-1 range)
    return np.array(colors) * 255  # Scale to 0-255
logger = logging.getLogger(__name__)


def main(args=None):
    '''
    Main execution point for PyLearn.
    '''
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['base_path'] = absolute_user_dir
    opt = init_distributed(opt)

    # META DATA
    pretrained_pth = os.path.join(opt['RESUME_FROM'])
    output_root = './output'
    image_pth = '/root/code/XMask3D/data/scannet_2d/scene0050_02/color/320.jpg'
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    # # ===========================
    # # —— 在此处插入：单独保存 sem_seg_head 的权重文件 —— 
    # # 提取 sem_seg_head 的 state_dict 并保存
    # basesave_path = 'checkpoint'
    # sem_seg_head_state_dict = model.model.sem_seg_head.state_dict()
    # save_path = os.path.join(basesave_path, 'XdecoderHead.pth')
    # # 确保输出目录存在
    # os.makedirs(basesave_path, exist_ok=True)
    # torch.save(sem_seg_head_state_dict, save_path)
    # logger.info(f"Saved sem_seg_head weights to: {save_path}")
    # # ===========================

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    stuff_classes = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','desk',
  'curtain','refrigerator','shower curtain','toilet','sink','bathtub']
    stuff_colors = get_color_palette(num_classes=len(stuff_classes))
    stuff_dataset_id_to_contiguous_id = {x:x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes + ["background"], is_eval=True)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(stuff_classes)

    with torch.no_grad():
        image_ori = Image.open(image_pth).convert("RGB")
        width = image_ori.size[0]
        height = image_ori.size[1]
        image = transform(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

        batch_inputs = [{'image': images, 'height': height, 'width': width}]
        outputs = model.forward(batch_inputs)
        print(outputs[-1].keys())
        visual = Visualizer(image_ori, metadata=metadata)

        sem_seg = outputs[-1]['sem_seg'].max(0)[1]
        demo = visual.draw_sem_seg(sem_seg.cpu(), alpha=0.5) # rgb Image

        if not os.path.exists(output_root):
            os.makedirs(output_root)
        demo.save(os.path.join(output_root, 'sem.png'))


if __name__ == "__main__":
    main()
    sys.exit(0)