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

from dataset.data_loader_ablation import (
    ScannetLoaderFull,
    SceneBatchSampler,
    scene_based_collate_fn
)

import MinkowskiEngine as ME
from omegaconf import OmegaConf
from models.utils.visualization import visualize_2d_semantic, get_color_palette, save_3d_point_cloud
from pathlib import Path
from xdecoder.utils.arguments import load_opt_from_config_files
from detectron2.utils.memory import retry_if_cuda_oom
from xdecoder.modeling.modules import sem_seg_postprocess
from models.affinity_module import SonataXAffinityTrainer

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
    parser = argparse.ArgumentParser(description="geopurify.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/geopurify_scannet.yaml",
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
    config = OmegaConf.load("./config/fusion_scannet.yaml")
    # config = OmegaConf.load("./config/fusion_matterport.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)
    print(OmegaConf.to_yaml(config))

    xdecoder_cfg = load_opt_from_config_files(["./config/xdecoder_focall_lang.yaml"])
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

    model = SonataXAffinityTrainer(args, xdecoder_cfg, scene_config,device,False).to(device)
    def analyze_model_parameters(model, show_details=True):
        print("=" * 60)
        print("Model Parameter Analysis")
        print("=" * 60)


        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        print(f"Total number of parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Number of trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"Number of untrainable parameters: {non_trainable_params:,} ({non_trainable_params/1e6:.2f}M)")

        if trainable_params > 0:
            print(f"Percentage of trainable parameters: {trainable_params/total_params*100:.2f}%")

        if show_details:
            print("\nDetailed statistics by module:")
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

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")

    if main_process():
        logger.info("=> setting up optimizer with differential learning rates...")

    student_model = model.module.affinity_student if args.distributed else model.affinity_student
    param_groups = student_model.get_param_groups()

    base_lr = args.lr_3d

    optimizer_params = [
        {"params": param_groups["input"], "lr": base_lr * 0.1, "name": "input_group"},
        {"params": param_groups["middle"], "lr": base_lr, "name": "middle_group"},
        {"params": param_groups["output"], "lr": base_lr * 5.0, "name": "output_group"}
    ]

    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=args.weight_decay)

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

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume, map_location=device)
            if 'model_state_dict' in checkpoint:
                if args.distributed:
                    model.module.affinity_student.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.affinity_student.load_state_dict(checkpoint['model_state_dict'])
            else:
                if args.distributed:
                    model.module.affinity_student.load_state_dict(checkpoint)
                else:
                    model.affinity_student.load_state_dict(checkpoint)

            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1
            else:
                import re
                match = re.search(r'epoch_(\d+)', args.resume)
                if match:
                    args.start_epoch = int(match.group(1)) + 1
                else:
                    args.start_epoch = 0

            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'tensorboard_scalars' in checkpoint:
                previous_scalars = checkpoint['tensorboard_scalars']
                if main_process():
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

    train_scene_file = "scannet_train.txt"
    with open(train_scene_file, "r") as f:
        scene_ids_train = [line.strip() for line in f if line.strip()]

    scene_file = "scannetevaluation.txt"
    if not os.path.exists(scene_file):
        raise FileNotFoundError(f"Scene list file '{scene_file}' not found.")

    with open(scene_file, "r") as f:
        scene_ids_val = [line.strip() for line in f if line.strip()]

    OurLoaderFull = ScannetLoaderFull
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
        scene_config=scene_config,
        specific_ids=scene_ids_train,
    )

    train_batch_sampler = SceneBatchSampler(train_data.samples, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_data,

        num_workers=args.workers,
        pin_memory=True,
        collate_fn=scene_based_collate_fn,
        batch_sampler=train_batch_sampler,
        worker_init_fn=worker_init_fn,
    )

    val_loader = None
    if args.distributed and val_sampler is not None:
        val_sampler.set_epoch(args.start_epoch)
    try:
        assert args.infer_batch_size_val == 1
    except AssertionError:
        print(f"Error: Expected infer_batch_size_val to be 1, but got {args.infer_batch_size_val}.")

    if main_process():
        logger.info("=> setting up learning rate scheduler (Warmup + Cosine Annealing)...")


    warmup_epochs = args.warmup_epochs
    warmup_iters = warmup_epochs * len(train_loader)
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_iters)

    main_epochs = args.epochs - warmup_epochs
    main_iters = main_epochs * len(train_loader)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=main_iters, eta_min=base_lr * 1e-3)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_iters])

    if args.resume and args.start_epoch > 0:
        total_steps_done = args.start_epoch * len(train_loader)
        for _ in range(total_steps_done):
            scheduler.step()

        if main_process():
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"=> Scheduler fast-forwarded to step {total_steps_done}, current LR: {current_lr:.7f}")

    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        if args.train_s:
            model.train()
            if args.distributed:
                train_batch_sampler.set_epoch(epoch)
            loss_meter = AverageMeter()
            max_iter = args.epochs * len(train_loader)

            for i, batch_data in enumerate(train_loader):
                optimizer.zero_grad()
                loss = model(batch_data)
                if isinstance(loss, dict):
                    loss = loss['loss']

                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_meter.update(loss.item())
                if main_process() and i % args.print_freq == 0:
                    current_lr = scheduler.get_last_lr()[1]
                    logger.info(f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t Loss: {loss.item():.4f}\t LR: {current_lr:.7f}")

                    writer.add_scalar("lr", current_lr, epoch_log)
                    if "lr" not in tensorboard_scalars:
                        tensorboard_scalars["lr"] = {}
                    tensorboard_scalars["lr"][epoch_log] = current_lr

            if main_process():
                writer.add_scalar("loss_train", loss_meter.avg, epoch_log)
                if "loss_train" not in tensorboard_scalars:
                    tensorboard_scalars["loss_train"] = {}
                tensorboard_scalars["loss_train"][epoch_log] = loss_meter.avg

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

    if main_process():
        writer.close()
        logger.info("==> Train/Eval done!")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()