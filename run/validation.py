import os
import time
import random
import numpy as np
import logging
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
import imageio
from MinkowskiEngine import SparseTensor
from sklearn.neighbors import KDTree
from util import config
import cv2
import open3d as o3d
from util.util import (
    AverageMeter,
    intersectionAndUnionGPU,
    poly_learning_rate,
    cosine_learning_rate,
    save_checkpoint,
    export_pointcloud,
    get_palette
)


# from dataset.data_loader_infer import (
#     ScannetLoaderFull,
#     collation_fn_eval_all_full,
# )
from dataset.data_loader_matterport import (
    ScannetLoaderFull,
    SceneBatchSampler,
    scene_based_collate_fn
)

import MinkowskiEngine as ME

from omegaconf import OmegaConf
#gaussians model
from models.gaussians.model import GaussianModel, render, render_chn
from models.utils.visualization import visualize_2d_semantic, get_color_palette, save_3d_point_cloud
from pathlib import Path
from xdecoder.utils.arguments import load_opt_from_config_files
from detectron2.utils.memory import retry_if_cuda_oom
from xdecoder.modeling.modules import sem_seg_postprocess
from xdecoder_test.models.affinity_module import SonataXAffinityTrainer

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def worker_init_fn(worker_id):
    """"""
    random.seed(time.time() + worker_id)

def get_logger():
    """"""

    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    return logger_in

def main_process():
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0
    )

def get_parser():
    """"""
    parser = argparse.ArgumentParser(description="xmask3d.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/scannet/xmask3d_scannet_infer_B12N7.yaml",
        help="config file",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args_in = parser.parse_args()
    assert args_in.config is not None
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    if args_in.opts:
        cfg = config.merge_cfg_from_list(cfg, args_in.opts)
    os.makedirs(cfg.save_path, exist_ok=True)
    model_dir = os.path.join(cfg.save_path, "model")
    result_dir = os.path.join(cfg.save_path, "result")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir + "/last", exist_ok=True)
    os.makedirs(result_dir + "/best", exist_ok=True)
    return cfg

def main():
    """"""

    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:

        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    if not hasattr(args, "use_shm"):
        args.use_shm = True

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.ngpus_per_node = len(args.train_gpu)

    config = OmegaConf.load("./config/scannet/fusion_matterport.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)
    print(OmegaConf.to_yaml(config))

    xdecoder_cfg = load_opt_from_config_files(["./config/scannet/xdecoder_focall_lang.yaml"])
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(
            main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args)
        )
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args, config, xdecoder_cfg)

def main_worker(gpu, ngpus_per_node, argss, scene_config, xdecoder_cfg):
    global args
    args = argss

    # 确定当前 GPU 设备
    device = torch.device(f"cuda:{gpu[0]}")

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        print("start")
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        print("over")

    # gaussians = GaussianModel(scene_config.model.sh_degree)
    # 将 device 传入模型构造函数中
    model = SonataXAffinityTrainer(args, xdecoder_cfg, scene_config,device,False).to(device)

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")

    # 2. --- 定义优化器 (全新版本，采用差分学习率) ---
    if main_process():
        logger.info("=> setting up optimizer with differential learning rates...")
    
    student_model = model.module.affinity_student if args.distributed else model.affinity_student
    param_groups = student_model.get_param_groups()

    # 从 args 中获取基准学习率
    base_lr = args.lr_3d 

    optimizer_params = [
        # 输入层: 使用较小的学习率进行精细微调
        {"params": param_groups["input"], "lr": base_lr * 0.1, "name": "input_group"},
        # 中间核心层: 使用基准学习率
        {"params": param_groups["middle"], "lr": base_lr, "name": "middle_group"},
        # 输出层: 使用较大的学习率以快速学习新的投影
        {"params": param_groups["output"], "lr": base_lr * 5.0, "name": "output_group"}
    ]

    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=args.weight_decay) # 强烈建议在 args 中添加 weight_decay

    args.index_split = 0

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.infer_batch_size_val = int(args.infer_batch_size_val / ngpus_per_node)
        args.workers = int(args.infer_workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device), device_ids=[gpu], find_unused_parameters=True
        )
    else:
        model = model.to(device)

    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # --- 修改后的 resume 逻辑 ---
    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            
            # 加载检查点
            checkpoint = torch.load(args.resume, map_location=device)
            
            # 加载模型状态字典
            if 'model_state_dict' in checkpoint:
                if args.distributed:
                    model.module.affinity_student.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.affinity_student.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 兼容旧格式
                if args.distributed:
                    model.module.affinity_student.load_state_dict(checkpoint)
                else:
                    model.affinity_student.load_state_dict(checkpoint)
            
            # 从检查点文件中获取 epoch 信息
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始训练
            else:
                # 如果检查点中没有 epoch 信息，尝试从文件名解析
                import re
                match = re.search(r'epoch_(\d+)', args.resume)
                if match:
                    args.start_epoch = int(match.group(1)) + 1
                else:
                    args.start_epoch = 0
            
            # 加载优化器状态（如果存在）
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 为了解决 TensorBoard 记录问题，我们需要记录之前的训练和验证信息
            if 'tensorboard_scalars' in checkpoint:
                previous_scalars = checkpoint['tensorboard_scalars']
                if main_process():
                    # 恢复之前的 TensorBoard 记录
                    for scalar_name, scalar_data in previous_scalars.items():
                        for step, value in scalar_data.items():
                            writer.add_scalar(scalar_name, value, step)
            
            if main_process():
                logger.info(
                    "=> loaded checkpoint '{}' (will start from epoch {})".format(
                        args.resume, args.start_epoch
                    )
                )
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))
            args.start_epoch = 0
    else:
        args.start_epoch = 0

    tensorboard_scalars = {}

    if not hasattr(args, "input_color"):

        args.input_color = False

    # 3. --- 加载你的Dataloader，无需任何改动！---
    # 使用训练集而不是验证集
    train_scene_file = "scannet_train.txt" # 假设你有训练场景列表
    with open(train_scene_file, "r") as f:
        scene_ids_train = [line.strip() for line in f if line.strip()]

    scene_file = "matterport_evaluation.txt"
    # scene_file = "/data1/scripts/processingdata/ScanNet/Tasks/Benchmark/scannetv2_val.txt"
    if not os.path.exists(scene_file):
        raise FileNotFoundError(f"Scene list file '{scene_file}' not found.")

    with open(scene_file, "r") as f:
        scene_ids_val = [line.strip() for line in f if line.strip()]

    # scene_ids_train = scene_ids_train[:2]
    # scene_ids_train = ['scene0588_03']
    # scene_ids_val = scene_ids_val[:1]
    # scene_ids_val = ['scene0207_00','scene0645_01','scene0231_02','scene0435_00','scene0221_00','scene0046_00',
    #                  'scene0353_00','scene0695_00','scene0670_01','scene0011_01']

    def get_batch_scenes(scene_ids_val, batch_idx, total_batches=4):
        """
        获取指定批次的场景ID
        batch_idx: 0, 1, 2, 3 (批次索引)
        """
        total_scenes = len(scene_ids_val)
        batch_size = total_scenes // total_batches
        remainder = total_scenes % total_batches
        
        # 计算当前批次的起始和结束索引
        start_idx = batch_idx * batch_size + min(batch_idx, remainder)
        if batch_idx < remainder:
            end_idx = start_idx + batch_size + 1
        else:
            end_idx = start_idx + batch_size
        
        return scene_ids_val[start_idx:end_idx]
    # # 0,1,2,3->01 23 45 67 -> 0 1 45 67
    # scene_ids_val = get_batch_scenes(scene_ids_val, 3)
    
    # 0,1,2,3,4,5,6,7
    # 0 2,3 4,5 3 8,9 5 6 7
    scene_ids_val = get_batch_scenes(scene_ids_val, 0)

    OurLoaderFull = ScannetLoaderFull
    # 设置 multiprocessing 的启动方式为 spawn
    mp.set_start_method('spawn', force=True)

    # --- 验证数据加载器 (如果需要) ---

    val_loader = None # 先初始化为 None
    val_data = OurLoaderFull(
        datapath_prefix=args.data_root,
        datapath_prefix_2d=args.data_root_2d,
        category_split=args.category_split,
        label_2d=args.label_2d,
        caption_path=args.caption_path,
        scannet200=args.scannet200,
        val_keep=args.val_keep,
        voxel_size=args.voxel_size,
        split="test",
        aug=False,
        memcache_init=args.use_shm,
        eval_all=True,
        input_color=args.input_color,
        # gaussians=gaussians,
        scene_config=scene_config,
        specific_ids=scene_ids_val,
    )
    
    # 为验证集创建 sampler，通常不打乱顺序 (shuffle=False)
    val_batch_sampler = SceneBatchSampler(val_data.samples, shuffle=False)
    
    val_loader = torch.utils.data.DataLoader(
        val_data,
        num_workers=args.infer_workers,
        pin_memory=True,
        collate_fn=scene_based_collate_fn, # -> 同样使用新的 collate_fn
        batch_sampler=val_batch_sampler, # -> 添加 batch_sampler
    )

    # 检查分布式模式下是否需要设置 epoch
    if args.distributed and val_sampler is not None:
        val_sampler.set_epoch(args.start_epoch)
    try:
        assert args.infer_batch_size_val == 1
    except AssertionError:
        print(f"Error: Expected infer_batch_size_val to be 1, but got {args.infer_batch_size_val}.")

    if main_process():
        logger.info("=> setting up learning rate scheduler (Warmup + Cosine Annealing)...")

    # --- 评估阶段 ---
    model.eval() 
    if args.distributed and val_sampler is not None:
        val_sampler.set_epoch(epoch)
    
    # 3. 执行验证
    (mIoU_2d_Base, mIoU_2d_Novel) = validate(val_loader, model, args.save_path)

    # # 4. 记录验证结果
    # if main_process():
    #     writer.add_scalar("mIoU_2d_Base", mIoU_2d_Base, epoch_log)
    #     writer.add_scalar("mIoU_2d_Novel", mIoU_2d_Novel, epoch_log)
        
    #     # 保存到 tensorboard_scalars 字典中
    #     if "mIoU_2d_Base" not in tensorboard_scalars:
    #         tensorboard_scalars["mIoU_2d_Base"] = {}
    #     if "mIoU_2d_Novel" not in tensorboard_scalars:
    #         tensorboard_scalars["mIoU_2d_Novel"] = {}
    #     tensorboard_scalars["mIoU_2d_Base"][epoch_log] = mIoU_2d_Base
    #     tensorboard_scalars["mIoU_2d_Novel"][epoch_log] = mIoU_2d_Novel

    if main_process():
        writer.close()
        logger.info("==> Train/Eval done!")

def validate(val_loader, model,save_path=None):
    """"""
    ""
    # debug_vis_dir = os.path.join(save_path, "debug_projection_vis")
    # os.makedirs(debug_vis_dir, exist_ok=True)
    # print(f"!!! 投影调试模式已启用 !!!")
    # print(f"可视化结果将保存在: {debug_vis_dir}")
    torch.backends.cudnn.enabled = False

    intersection_meter_Base = AverageMeter()
    intersection_meter_Novel = AverageMeter()
    union_meter_Base = AverageMeter()
    union_meter_Novel = AverageMeter()
    target_meter_Base = AverageMeter()
    target_meter_Novel = AverageMeter()

    intersection_meter_2d_Base = AverageMeter()
    intersection_meter_2d_Novel = AverageMeter()
    union_meter_2d_Base = AverageMeter()
    union_meter_2d_Novel = AverageMeter()
    target_meter_2d_Base = AverageMeter()
    target_meter_2d_Novel = AverageMeter()

    # 添加所有类别的整体指标meters
    intersection_meter_2d_All = AverageMeter()
    union_meter_2d_All = AverageMeter()
    target_meter_2d_All = AverageMeter()

    model.eval()

    torch.rand(1)
    np.random.rand(1)

    palette = get_palette()
    vis_dir = "xdecoder_test/paper"
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            if batch_data is None:
                print(f"Warning: batch_data is None at iteration {i}, skipping...")
                continue
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

            # Get scene_id (needs access to val_loader.dataset properties)
            # Assuming data_paths stores the path used to load the scene
            try:
                scene_id = val_loader.dataset.data_paths[i % len(val_loader.dataset.data_paths)].split('/')[-1].split('_vh_clean_2.pth')[0]
            except Exception as e:
                logger.warning(f"Could not determine scene_id for visualization: {e}")
                scene_id = f"scene_{i:04d}"
            eval_results = model.evaluate_scene(batch_data,vis_prefix = scene_id)
            scene_features_2d = eval_results["scene_features"] # shape: [N, D]
            text_features = eval_results["text_features"]     # shape: [C, D]
            logit_scale = eval_results["logit_scale"]

            # <<< 核心改动 3: 直接从场景级特征生成预测 >>>
            # 1. 归一化特征
            scene_features_2d = F.normalize(scene_features_2d, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # 2. 计算 logits (相似度分数)
            logits_pred_2d = logit_scale * (scene_features_2d @ text_features.t())
            # logits_pred_2d = scene_features_2d @ text_features.t()
            
            # 3. 获取最终类别预测 (Argmax)
            # 这就是最终的、未经填充的预测结果
            scene_pred_2d = torch.max(logits_pred_2d, 1)[1]

            # <<< 核心改动 4: 适应性地保留 K-近邻填充逻辑 >>>
            # 检查 evaluate_scene 是否为某些点生成了零特征
            # 这是比 counter 更直接的判断“未见点”的方式
            unseen_mask = (torch.sum(scene_features_2d.abs(), dim=1) == 0)
            unseen_mask = unseen_mask.to(scene_coords.device)
            
            if unseen_mask.any():
                seen_mask = ~unseen_mask
                
                # 获取有特征和无特征的点的坐标
                seen_coords = scene_coords[seen_mask][:, 1:4].cpu() # KDTree 需要CPU数据
                unseen_coords = scene_coords[unseen_mask][:, 1:4].cpu()

                # 如果存在有效点，则进行填充
                if seen_coords.shape[0] > 0:
                    kdtree = KDTree(seen_coords)
                    distances, indices = kdtree.query(unseen_coords, k=1)
                    
                    # 找到无特征点 对应的 最近的有特征点的 原始索引
                    original_seen_indices = torch.where(seen_mask)[0]
                    matched_indices = original_seen_indices[indices.flatten()]
                    
                    # 获取无特征点的原始索引
                    original_unseen_indices = torch.where(unseen_mask)[0]

                    # 进行填充
                    scene_pred_2d[original_unseen_indices] = scene_pred_2d[matched_indices]

                    del kdtree

            # logger.info(f"Visualizing scene {i}: {scene_id}")

            # # # Call save function (Prediction)
            # save_3d_point_cloud(scene_coords, scene_pred_2d, scene_label, palette, vis_dir, scene_id, i)

            # # Call save function (Ground Truth)
            # save_3d_point_cloud(scene_coords, scene_pred_2d, scene_label, palette, vis_dir, scene_id, i, use_gt=True)

            intersection_2d, union_2d, target_2d = intersectionAndUnionGPU(
                scene_pred_2d.to(scene_label.device),
                scene_label.detach(),
                args.test_classes,
                args.test_ignore_label,
            )

            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
                dist.all_reduce(intersection_2d), dist.all_reduce(
                    union_2d
                ), dist.all_reduce(target_2d)
                dist.all_reduce(intersection_3d), dist.all_reduce(
                    union_3d
                ), dist.all_reduce(target_3d)

            intersection_2d, union_2d, target_2d = (
                intersection_2d.cpu().numpy(),
                union_2d.cpu().numpy(),
                target_2d.cpu().numpy(),
            )

            intersection_meter_2d_Base.update(
                intersection_2d[args.category_split.base_category]
            ), union_meter_2d_Base.update(
                union_2d[args.category_split.base_category]
            ), target_meter_2d_Base.update(
                target_2d[args.category_split.base_category]
            )

            intersection_meter_2d_Novel.update(
                intersection_2d[args.category_split.novel_category]
            ), union_meter_2d_Novel.update(
                union_2d[args.category_split.novel_category]
            ), target_meter_2d_Novel.update(
                target_2d[args.category_split.novel_category]
            )

            # 添加所有类别的整体指标更新
            intersection_meter_2d_All.update(intersection_2d)
            union_meter_2d_All.update(union_2d)  
            target_meter_2d_All.update(target_2d)

            if main_process():
                # logger.info(
                #     "Checking: [{}/{}/{}]".format(
                #         scene_pred.shape, scene_true.shape, scene_false.shape
                #     )
                # )
                logger.info("Process: [{}/{}]".format(i, len(val_loader)))
            
            del batch_data
            del eval_results
            del scene_features_2d
            del logits_pred_2d
            del scene_pred_2d
            if 'text_features' in locals(): del text_features
            torch.cuda.empty_cache()
            
            if main_process():
                # Base类别指标
                iou_class_2d_Base = intersection_meter_2d_Base.sum / (
                    union_meter_2d_Base.sum + 1e-10
                )
                accuracy_class_2d_Base = intersection_meter_2d_Base.sum / (
                    target_meter_2d_Base.sum + 1e-10
                )
                mIoU_2d_Base = np.mean(iou_class_2d_Base)
                mAcc_2d_Base = np.mean(accuracy_class_2d_Base)
                allAcc_2d_Base = sum(intersection_meter_2d_Base.sum) / (
                    sum(target_meter_2d_Base.sum) + 1e-10
                )

                # Novel类别指标
                iou_class_2d_Novel = intersection_meter_2d_Novel.sum / (
                    union_meter_2d_Novel.sum + 1e-10
                )
                accuracy_class_2d_Novel = intersection_meter_2d_Novel.sum / (
                    target_meter_2d_Novel.sum + 1e-10
                )
                mIoU_2d_Novel = np.mean(iou_class_2d_Novel)
                mAcc_2d_Novel = np.mean(accuracy_class_2d_Novel)
                allAcc_2d_Novel = sum(intersection_meter_2d_Novel.sum) / (
                    sum(target_meter_2d_Novel.sum) + 1e-10
                )

                # 所有类别的整体指标
                iou_class_2d_All = intersection_meter_2d_All.sum / (
                    union_meter_2d_All.sum + 1e-10
                )
                accuracy_class_2d_All = intersection_meter_2d_All.sum / (
                    target_meter_2d_All.sum + 1e-10
                )
                mIoU_2d_All = np.mean(iou_class_2d_All)
                mAcc_2d_All = np.mean(accuracy_class_2d_All)
                allAcc_2d_All = sum(intersection_meter_2d_All.sum) / (
                    sum(target_meter_2d_All.sum) + 1e-10
                )
                # 在原代码的log输出部分添加：
                logger.info("Raw stats Base: intersection {}, union {}, target {}".format(
                    intersection_meter_2d_Base.sum, union_meter_2d_Base.sum, target_meter_2d_Base.sum
                ))
                logger.info("Raw stats Novel: intersection {}, union {}, target {}".format(
                    intersection_meter_2d_Novel.sum, union_meter_2d_Novel.sum, target_meter_2d_Novel.sum
                ))
                logger.info("Raw stats All: intersection {}, union {}, target {}".format(
                    intersection_meter_2d_All.sum, union_meter_2d_All.sum, target_meter_2d_All.sum
                ))

                # 输出各类别指标
                logger.info(
                    "Val 2d result: mIoU_Base/mAcc_Base/allAcc_Base {:.4f}/{:.4f}/{:.4f}.".format(
                        mIoU_2d_Base, mAcc_2d_Base, allAcc_2d_Base
                    )
                )
                logger.info(
                    "Val 2d result: mIoU_Novel/mAcc_Novel/allAcc_Novel {:.4f}/{:.4f}/{:.4f}.".format(
                        mIoU_2d_Novel, mAcc_2d_Novel, allAcc_2d_Novel
                    )
                )
                
                # 添加所有类别的整体指标输出
                logger.info(
                    "Val 2d result: mIoU_All/mAcc_All/allAcc_All {:.4f}/{:.4f}/{:.4f}.".format(
                        mIoU_2d_All, mAcc_2d_All, allAcc_2d_All
                    )
                )
                
                logger.info("iou_class_Base '{}'".format(iou_class_2d_Base))
                logger.info("iou_class_Novel '{}'".format(iou_class_2d_Novel))
                logger.info("iou_class_All '{}'".format(iou_class_2d_All))
    return (
        mIoU_2d_Base,
        mIoU_2d_Novel,
    )

if __name__ == "__main__":
    main()