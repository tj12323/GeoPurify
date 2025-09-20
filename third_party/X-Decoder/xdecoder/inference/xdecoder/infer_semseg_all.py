import os
import sys
import logging
from tqdm import tqdm # 引入tqdm来显示进度条
import pdb

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

from PIL import Image
import numpy as np
np.random.seed(1)

import torch
import torch.nn.functional as F # 引入F
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
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20)[:3] for i in range(num_classes)]
    return np.array(colors) * 255

logger = logging.getLogger(__name__)

# +++ 新增：IoU计算函数 +++
def calculate_iou(pred, gt, num_classes, ignore_index=255):
    """
    计算并累加Intersection和Union。
    pred: 预测的类别张量 (H, W)
    gt: 真实标签张量 (H, W)
    num_classes: 类别总数
    ignore_index: 要忽略的标签值
    """
    # 创建一个忽略掩码
    ignore_mask = (gt == ignore_index)
    # 将pred和gt中对应ignore_mask的位置设为一个不会影响计算的值
    # 这里我们将它设为0，并在后面计算时排除掉这个组合
    pred[ignore_mask] = 0
    gt[ignore_mask] = 0

    # 使用bincount高效计算混淆矩阵
    # 核心技巧：将二维问题展平为一维
    confusion_matrix = torch.bincount(
        pred.flatten() * num_classes + gt.flatten(),
        minlength=num_classes**2
    ).reshape(num_classes, num_classes).cpu().numpy()

    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    
    return intersection, union


def main(args=None):
    '''
    Main execution point for PyLearn.
    '''
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['base_path'] = absolute_user_dir
    opt = init_distributed(opt)

    # --- START: 参数与路径配置 ---
    pretrained_pth = os.path.join(opt['RESUME_FROM'])
    output_root = './output_2d_val'
    scannet_base_path = '/root/code/XMask3D/data/scannet_2d/'
    evaluation_list_file = '/root/code/XMask3D/scannetevaluation.txt'
    SAVE_VISUALIZATION = False
    # --- END: 参数与路径配置 ---

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    # --- START: 核心修正 - 使用您提供的精确映射关系 (更鲁棒的实现) ---

    # 1. 定义您提供的映射关系
    reverse_label_mapping = {
        1.0: [0.0], 2.0: [1.0], 3.0: [2.0], 4.0: [3.0], 5.0: [4.0],
        6.0: [5.0], 7.0: [6.0], 8.0: [7.0], 9.0: [8.0], 10.0: [9.0],
        11.0: [10.0], 12.0: [11.0], 14.0: [12.0], 16.0: [13.0],
        24.0: [14.0], 28.0: [15.0], 33.0: [16.0], 34.0: [17.0],
        36.0: [18.0],
    }

    # 2. 更新类别列表以匹配20个类别
    stuff_classes = [
        'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
        'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub'
    ]
    num_classes = len(stuff_classes)
    logger.info(f"Using provided mapping with {num_classes} classes.")

    # 3. 创建一个完整且鲁棒的映射表 (0-255)
    ignore_index = 255
    # 创建一个256大小的数组，因为uint8的取值范围是0-255
    raw_id_to_train_id = np.full(256, ignore_index, dtype=np.uint8)
    for raw_id_float, train_id_list in reverse_label_mapping.items():
        raw_id = int(raw_id_float)
        train_id = int(train_id_list[0])
        raw_id_to_train_id[raw_id] = train_id
    
    mapping_tensor = torch.from_numpy(raw_id_to_train_id).cuda()
    
    # --- END: 核心修正 ---

    stuff_colors = get_color_palette(num_classes=num_classes)
    stuff_dataset_id_to_contiguous_id = {x:x for x in range(num_classes)}
    MetadataCatalog.get("demo").set(
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes + ["background"], is_eval=True)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = num_classes
    
    total_intersection = np.zeros(num_classes, dtype=np.float64)
    total_union = np.zeros(num_classes, dtype=np.float64)
    image_count = 0

    try:
        with open(evaluation_list_file, 'r') as f:
            val_scenes = [line.strip() for line in f]
    except FileNotFoundError:
        logger.error(f"Evaluation list file not found at: {evaluation_list_file}")
        return

    for scene_name in tqdm(val_scenes, desc="Processing Scenes"):
        color_dir = os.path.join(scannet_base_path, scene_name, 'color')
        label_dir = os.path.join(scannet_base_path, scene_name, 'label')
        if not os.path.exists(color_dir): continue

        image_files = sorted([f for f in os.listdir(color_dir) if f.endswith('.jpg')])
        for image_name in image_files:
            image_pth = os.path.join(color_dir, image_name)
            label_pth = os.path.join(label_dir, image_name.replace('.jpg', '.png'))
            if not os.path.exists(label_pth): continue

            try:
                with torch.no_grad():
                    image_ori = Image.open(image_pth).convert("RGB")
                    width, height = image_ori.size
                    image = transform(image_ori)
                    image = np.asarray(image)
                    image_ori_np = np.asarray(image_ori)
                    images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

                    # a. *** 鲁棒地加载GT Label并确保是二维的 ***
                    # 使用.convert('L')确保加载为单通道灰度图，避免格式问题
                    gt_label_img = Image.open(label_pth).convert('L')
                    gt_label_raw = torch.from_numpy(np.array(gt_label_img)).cuda()
                    # b. 应用ID映射
                    gt_label_mapped = torch.take(mapping_tensor, gt_label_raw.long())

                    batch_inputs = [{'image': images, 'height': height, 'width': width}]
                    outputs = model.forward(batch_inputs)
                    
                    sem_seg_logits = outputs[-1]['sem_seg']
                    sem_seg_resized = F.interpolate(
                        sem_seg_logits.unsqueeze(0), 
                        size=(height, width), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                    pred_label = sem_seg_resized.max(0)[1]

                    # c. 使用映射后的GT进行IoU计算
                    intersection, union = calculate_iou(pred_label, gt_label_mapped, num_classes, ignore_index)
                    total_intersection += intersection
                    total_union += union
                    image_count += 1

                    if SAVE_VISUALIZATION:
                        pass

            except Exception as e:
                logger.error(f"Error processing {image_pth}: {e}")
    
    # ... (最后的指标输出部分完全不变) ...
    # (代码与上一版相同，此处省略以保持简洁)
    logger.info(f"Evaluation finished on {image_count} images.")
    iou_per_class = total_intersection / (total_union + 1e-10)
    base_labels_set = set(['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'curtain'])
    base_indices = [i for i, name in enumerate(stuff_classes) if name in base_labels_set]
    novel_indices = [i for i, name in enumerate(stuff_classes) if name not in base_labels_set]
    iou_base = iou_per_class[base_indices]
    iou_novel = iou_per_class[novel_indices]
    mIoU = np.nanmean(iou_per_class)
    mIoU_base = np.nanmean(iou_base)
    mIoU_novel = np.nanmean(iou_novel)
    print("\n" + "="*60)
    print("2D Semantic Segmentation Validation Results")
    print("="*60)
    print(f"{'Overall Mean IoU (mIoU)':<30s}: {mIoU:.4f}")
    print(f"{'Base Classes mIoU':<30s}: {mIoU_base:.4f} ({len(base_indices)} classes)")
    print(f"{'Novel Classes mIoU':<30s}: {mIoU_novel:.4f} ({len(novel_indices)} classes)")
    print("-" * 60)
    print("\nIoU per class (Base):")
    for i in base_indices:
        print(f"{stuff_classes[i]:<20s}: {iou_per_class[i]:.4f}")
    print("\nIoU per class (Novel):")
    for i in novel_indices:
        print(f"{stuff_classes[i]:<20s}: {iou_per_class[i]:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
    sys.exit(0)