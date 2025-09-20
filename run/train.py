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
)


# from dataset.data_loader_infer import (
#     ScannetLoaderFull,
#     collation_fn_eval_all_full,
# )
from dataset.data_loader_ablation import (
    ScannetLoaderFull,
    SceneBatchSampler,
    scene_based_collate_fn
)

# from models.xmask3dall import XMASK3d as Model
# from models.checkpoint import XMask3dCheckpointer
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

    config = OmegaConf.load("./config/scannet/fusion_scannet.yaml")
    # config = OmegaConf.load("./config/scannet/fusion_matterport.yaml")
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
    def analyze_model_parameters(model, show_details=True):
        """全面分析模型参数"""
        print("=" * 60)
        print("模型参数分析")
        print("=" * 60)
        
        # 总体统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        print(f"总参数数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"可训练参数数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"不可训练参数数量: {non_trainable_params:,} ({non_trainable_params/1e6:.2f}M)")
        
        if trainable_params > 0:
            print(f"可训练参数占比: {trainable_params/total_params*100:.2f}%")
        
        if show_details:
            print("\n按模块详细统计:")
            print("-" * 60)
            for name, module in model.named_children():
                module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                module_total = sum(p.numel() for p in module.parameters())
                if module_total > 0:
                    print(f"{name:20s}: {module_trainable:>10,} / {module_total:>10,} ({module_trainable/1e6:>6.2f}M)")
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }

    param_info = analyze_model_parameters(model)
    # import pdb;pdb.set_trace()

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

    scene_file = "scannetevaluation.txt"
    if not os.path.exists(scene_file):
        raise FileNotFoundError(f"Scene list file '{scene_file}' not found.")

    with open(scene_file, "r") as f:
        scene_ids_val = [line.strip() for line in f if line.strip()]

    # scene_ids_train = scene_ids_train[:1]
    # scene_ids_train = ['scene0247_01']
    # scene_ids_val = scene_ids_val[:1]


    OurLoaderFull = ScannetLoaderFull

    # train_data = OurLoaderFull(
    #     datapath_prefix=args.data_root,
    #     datapath_prefix_2d=args.data_root_2d,
    #     category_split=args.category_split,
    #     label_2d=args.label_2d,
    #     caption_path=args.caption_path,
    #     entity_path=args.entity_path,
    #     scannet200=args.scannet200,
    #     val_keep=args.val_keep,
    #     voxel_size=args.voxel_size,
    #     split="train",
    #     aug=args.aug,
    #     memcache_init=args.use_shm,
    #     loop=args.loop,
    #     input_color=args.input_color,
    #     gaussians=gaussians,
    #     scene_config=scene_config,
    #     specific_ids=scene_ids_train,
    # )
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if args.distributed else None
    
    # train_loader = torch.utils.data.DataLoader(
    #     train_data,
    #     batch_size=args.batch_size, # 训练批处理大小，例如 1 或 2，因为在线生成数据很耗内存
    #     shuffle=(train_sampler is None),
    #     num_workers=args.workers,
    #     pin_memory=False,
    #     drop_last=True, # 训练时可以丢弃最后一个不完整的batch
    #     collate_fn=collation_fn_eval_all_full,
    #     sampler=train_sampler,
    #     worker_init_fn=worker_init_fn,
    # )

    # if args.evaluate:
    #     val_data = OurLoaderFull(
    #         datapath_prefix=args.data_root,
    #         datapath_prefix_2d=args.data_root_2d,
    #         category_split=args.category_split,
    #         label_2d=args.label_2d,
    #         caption_path=args.caption_path,
    #         scannet200=args.scannet200,
    #         val_keep=args.val_keep,
    #         voxel_size=args.voxel_size,
    #         split="val",
    #         aug=False,
    #         memcache_init=args.use_shm,
    #         eval_all=True,
    #         input_color=args.input_color,
    #         gaussians=gaussians,
    #         scene_config=scene_config,
    #         specific_ids=scene_ids_val,
    #     )
    #     val_sampler = (
    #         torch.utils.data.distributed.DistributedSampler(val_data)
    #         if args.distributed
    #         else None
    #     )
    #     val_loader = torch.utils.data.DataLoader(
    #         val_data,
    #         batch_size=args.infer_batch_size_val,
    #         shuffle=False,
    #         num_workers=args.infer_workers,
    #         pin_memory=True,
    #         drop_last=False,
    #         collate_fn=collation_fn_eval_all_full,
    #         sampler=val_sampler,
    #     )
    # 1. 实例化数据集 (这部分不变)
    train_data = OurLoaderFull(
        datapath_prefix=args.data_root,
        datapath_prefix_2d=args.data_root_2d,
        category_split=args.category_split,
        label_2d=args.label_2d,
        caption_path=args.caption_path,
        entity_path=args.entity_path,
        scannet200=args.scannet200,
        val_keep=args.val_keep,
        voxel_size=args.voxel_size,
        split="train",
        aug=args.aug,
        memcache_init=args.use_shm,
        loop=args.loop,
        input_color=args.input_color,
        # gaussians=gaussians,
        scene_config=scene_config,
        specific_ids=scene_ids_train,
    )

    # 2. 创建我们新的 SceneBatchSampler (替换旧的 sampler)
    #    它会接管批处理逻辑，确保一个批次只包含一个场景
    train_batch_sampler = SceneBatchSampler(train_data.samples, shuffle=True)

    # 3. 创建 DataLoader (注意参数变化)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        # -> 移除 batch_size, shuffle, sampler, drop_last
        num_workers=args.workers,
        pin_memory=True,  # 建议为 True 以提升 GPU 数据传输速度
        collate_fn=scene_based_collate_fn, # -> 使用新的 collate_fn
        batch_sampler=train_batch_sampler, # -> 添加 batch_sampler
        worker_init_fn=worker_init_fn,
    )

    # --- 验证数据加载器 (如果需要) ---

    val_loader = None # 先初始化为 None
    # if args.evaluate:
    #     val_data = OurLoaderFull(
    #         datapath_prefix=args.data_root,
    #         datapath_prefix_2d=args.data_root_2d,
    #         category_split=args.category_split,
    #         label_2d=args.label_2d,
    #         caption_path=args.caption_path,
    #         scannet200=args.scannet200,
    #         val_keep=args.val_keep,
    #         voxel_size=args.voxel_size,
    #         split="val",
    #         aug=False,
    #         memcache_init=args.use_shm,
    #         eval_all=True,
    #         input_color=args.input_color,
    #         # gaussians=gaussians,
    #         scene_config=scene_config,
    #         # specific_ids=scene_ids_val,
    #     )
        
    #     # 为验证集创建 sampler，通常不打乱顺序 (shuffle=False)
    #     val_batch_sampler = SceneBatchSampler(val_data.samples, shuffle=False)
        
    #     val_loader = torch.utils.data.DataLoader(
    #         val_data,
    #         num_workers=args.infer_workers,
    #         pin_memory=True,
    #         collate_fn=scene_based_collate_fn, # -> 同样使用新的 collate_fn
    #         batch_sampler=val_batch_sampler, # -> 添加 batch_sampler
    #     )

    # 检查分布式模式下是否需要设置 epoch
    if args.distributed and val_sampler is not None:
        val_sampler.set_epoch(args.start_epoch)
    try:
        assert args.infer_batch_size_val == 1
    except AssertionError:
        print(f"Error: Expected infer_batch_size_val to be 1, but got {args.infer_batch_size_val}.")

    if main_process():
        logger.info("=> setting up learning rate scheduler (Warmup + Cosine Annealing)...")

    # 调度器1: 线性预热
    warmup_epochs = args.warmup_epochs # 在 args 中添加 warmup_epochs, 建议为 1 或 2
    warmup_iters = warmup_epochs * len(train_loader)
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_iters)

    # 调度器2: 余弦退火
    # 注意：T_max 是退火阶段的总迭代次数
    main_epochs = args.epochs - warmup_epochs
    main_iters = main_epochs * len(train_loader)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=main_iters, eta_min=base_lr * 1e-3) # eta_min 防止学习率降为0

    # 组合调度器 (需要 PyTorch 1.11.0+)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_iters])

    # --- 如果是 resume，需要快进调度器到正确的步数 ---
    if args.resume and args.start_epoch > 0:
        # 计算已经训练过的总步数
        total_steps_done = args.start_epoch * len(train_loader)
        
        # 快进调度器
        for _ in range(total_steps_done):
            scheduler.step()
        
        if main_process():
            current_lr = scheduler.get_last_lr()[0]  # 获取当前学习率
            logger.info(f"=> Scheduler fast-forwarded to step {total_steps_done}, current LR: {current_lr:.7f}")
            
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        # --- 训练阶段 ---
        if args.train_s:
            # 1. 设置模型为训练模式
            model.train() 
            if args.distributed:
                train_batch_sampler.set_epoch(epoch)
            
            # 2. 初始化损失记录器
            loss_meter = AverageMeter()
            max_iter = args.epochs * len(train_loader)
            # 3. 遍历训练数据
            for i, batch_data in enumerate(train_loader):
                optimizer.zero_grad()
                
                # 前向传播和损失计算
                loss = model(batch_data)
                # current_iter = epoch * len(train_loader) + i + 1
                # assert args.learning_rate_type in ["poly", "cosine"]
                # if args.learning_rate_type == "poly":
                #     current_lr_1 = poly_learning_rate(
                #         args.lr_3d, current_iter, max_iter, power=args.power
                #     )
                # elif args.learning_rate_type == "cosine":
                #     current_lr_1 = cosine_learning_rate(args.lr_3d, current_iter, max_iter)
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = current_lr_1
                
                if isinstance(loss, dict):
                    loss = loss['loss']
                
                loss.backward()
                optimizer.step()

                scheduler.step()
                
                loss_meter.update(loss.item())
                
                if main_process() and i % args.print_freq == 0:
                    # 记录第一个参数组的学习率作为代表
                    current_lr = scheduler.get_last_lr()[1] 
                    logger.info(f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t Loss: {loss.item():.4f}\t LR: {current_lr:.7f}")
                    
                    writer.add_scalar("lr", current_lr, epoch_log)
                    # 保存到 tensorboard_scalars 字典中
                    if "lr" not in tensorboard_scalars:
                        tensorboard_scalars["lr"] = {}
                    tensorboard_scalars["lr"][epoch_log] = current_lr

            # 4. 记录 epoch 的平均损失
            if main_process():
                writer.add_scalar("loss_train", loss_meter.avg, epoch_log)
                # 保存到 tensorboard_scalars 字典中
                if "loss_train" not in tensorboard_scalars:
                    tensorboard_scalars["loss_train"] = {}
                tensorboard_scalars["loss_train"][epoch_log] = loss_meter.avg
        
        # # --- 评估阶段 ---
        # if args.evaluate:
        #     # 1. 检查是否到达评估频率
        #     if epoch_log % args.eval_freq == 0:
        #         # 2. 设置模型为评估模式
        #         model.eval() 
        #         if args.distributed and val_sampler is not None:
        #             val_sampler.set_epoch(epoch)
                
        #         # 3. 执行验证
        #         (mIoU_2d_Base, mIoU_2d_Novel) = validate(val_loader, model, args.save_path)

        #         # 4. 记录验证结果
        #         if main_process():
        #             writer.add_scalar("mIoU_2d_Base", mIoU_2d_Base, epoch_log)
        #             writer.add_scalar("mIoU_2d_Novel", mIoU_2d_Novel, epoch_log)
                    
        #             # 保存到 tensorboard_scalars 字典中
        #             if "mIoU_2d_Base" not in tensorboard_scalars:
        #                 tensorboard_scalars["mIoU_2d_Base"] = {}
        #             if "mIoU_2d_Novel" not in tensorboard_scalars:
        #                 tensorboard_scalars["mIoU_2d_Novel"] = {}
        #             tensorboard_scalars["mIoU_2d_Base"][epoch_log] = mIoU_2d_Base
        #             tensorboard_scalars["mIoU_2d_Novel"][epoch_log] = mIoU_2d_Novel

        # --- 保存检查点阶段 ---
        # 只有在进行了训练的情况下才保存模型
        if args.train_s and main_process():
            if epoch_log % args.save_freq == 0:
                student_state_dict = model.module.affinity_student.state_dict() if args.distributed else model.affinity_student.state_dict()
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': student_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'tensorboard_scalars': tensorboard_scalars
                }
                torch.save(checkpoint, os.path.join(args.save_path, "model", 'affinity_predictor_last.pth'))
            
            if epoch_log % 5 == 0 or epoch == args.epochs - 1:
                logger.info(f"Saving checkpoint at epoch {epoch}...")
                student_state_dict = model.module.affinity_student.state_dict() if args.distributed else model.affinity_student.state_dict()
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': student_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'tensorboard_scalars': tensorboard_scalars
                }
                torch.save(checkpoint, os.path.join(args.save_path, "model", f'affinity_predictor_epoch_{epoch}.pth'))

    # # --- 评估阶段 ---
    # model.eval() 
    # if args.distributed and val_sampler is not None:
    #     val_sampler.set_epoch(epoch)
    
    # # 3. 执行验证
    # (mIoU_2d_Base, mIoU_2d_Novel) = validate(val_loader, model, args.save_path)

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

if __name__ == "__main__":
    # 设置 multiprocessing 的启动方式为 spawn
    mp.set_start_method('spawn', force=True)
    main()