import os
import random
import shutil
import time
from collections import OrderedDict
from os.path import isfile, join

import cv2
import numpy as np
import torch
import torch.nn as nn
from absl.logging import debug, flags, info
from pudb import set_trace

# DistributedDataParallelとDistributedSamplerを有効にする
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.testing._internal.common_quantization import AverageMeter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from optimizer import Optimizer
from dataloader.evunroll_simulated_dataset import get_evunroll_simulated_dataset_with_config
from loss import get_loss, get_metric
from get_model import get_model
from visualize import get_visulization

# 分散学習用モジュールのインポート
import torch.distributed as dist

FLAGS = flags.FLAGS

def move_tensors_to_cuda(data, device):
    """
    データ構造（辞書、リスト、タプル）を再帰的にたどり、
    見つかった全てのTensorを特定のデバイスに移動する。
    """
    if isinstance(data, dict):
        return {key: move_tensors_to_cuda(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_tensors_to_cuda(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_tensors_to_cuda(item, device) for item in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    else:
        return data

def get_model_parameters_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

class ParallelLaunch:
    def __init__(self, config):
        self.config = config
        if self.config.DISTRIBUTED:
            self.rank = int(os.environ["LOCAL_RANK"])
            info(f"Initialized DDP Process with rank: {self.rank}")

            # 分散学習環境の初期化
            dist.init_process_group(backend="nccl", init_method="env://")
        else:
            self.rank = 0

        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        random.seed(config.SEED)
        np.random.seed(config.SEED)
        
        # TensorBoardはランク0のみで記録
        if self.rank == 0:
            self.tb_recoder = SummaryWriter(FLAGS.log_dir)
        else:
            self.tb_recoder = None
            
        self.start_time = time.time()
        self.visualizer = None

    def run(self):
        #val_dataset = get_evunroll_simulated_dataset_with_config(self.config.DATASET)
        train_dataset, val_dataset = get_evunroll_simulated_dataset_with_config(self.config.DATASET)
        model = get_model(self.config.MODEL)

        num_params = get_model_parameters_count(model)
        info(f"モデルのパラメータ総数: {num_params}")

        criterion = get_loss(self.config.LOSS)
        metrics = get_metric(self.config.METRICS)
        opt = Optimizer(self.config.OPTIMIZER, model)

        if self.config.DISTRIBUTED:
        # モデルをGPUに移動し、DDPでラップする
            device_id = self.rank
            model = model.to(device_id)
            model = DDP(model, device_ids=[device_id])
        else:
            device_id = 0
            model = model.to(device_id)

        if self.config.RESUME.PATH:
            if not isfile(self.config.RESUME.PATH):
                raise ValueError(f"File not found, {self.config.RESUME.PATH}")

            # DDPに対応したチェックポイントのロード
            checkpoint = torch.load(self.config.RESUME.PATH, map_location=f'cuda:{device_id}')

            if self.config.RESUME.SET_EPOCH:
                self.config.START_EPOCH = checkpoint["epoch"]
                opt.optimizer.load_state_dict(checkpoint["optimizer"])
                opt.scheduler.load_state_dict(checkpoint["scheduler"])
            
            # DDPは 'module.' を自動で付与するため、ここでは 'module.' を除去しない
            if self.config.RESUME_STRICT:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint["state_dict"], strict=False)

        if self.config.DISTRIBUTED:
        # DDPに対応したデータサンプラーを使用
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            train_shuffle = False # Samplerとshuffleは同時に使えない
            val_shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            train_shuffle = True # シングルGPUではshuffleを有効にする
            val_shuffle = False
        
        
        train_loader = DataLoader(
            dataset=train_dataset,
            sampler=train_sampler,
            shuffle=train_shuffle,
            batch_size=self.config.TRAIN_BATCH_SIZE,
            num_workers=self.config.JOBS,
            pin_memory=True,
            drop_last=True,
        )
        

        val_loader = DataLoader(
            dataset=val_dataset,
            sampler=val_sampler,
            shuffle=val_shuffle,
            batch_size=self.config.VAL_BATCH_SIZE,
            num_workers=self.config.JOBS,
            pin_memory=True,
            drop_last=True,
        )

        if self.config.TEST_ONLY:
            self.valid(val_loader, model, criterion, metrics, 0)
            return
        
        min_loss = 123456789.0
        for epoch in range(self.config.START_EPOCH, self.config.END_EPOCH):
            if self.config.DISTRIBUTED:
                train_sampler.set_epoch(epoch)
            
            if self.rank == 0:
                total_epochs = self.config.END_EPOCH - self.config.START_EPOCH
                elapsed_epochs = epoch - self.config.START_EPOCH + 1
                elapsed_time_total = time.time() - self.start_time
                avg_time_per_epoch = elapsed_time_total / elapsed_epochs
                remaining_epochs = total_epochs - elapsed_epochs
                remaining_time = remaining_epochs * avg_time_per_epoch
                rem_time_h = int(remaining_time // 3600)
                rem_time_m = int((remaining_time % 3600) // 60)
                rem_time_s = int(remaining_time % 60)
                info(f"残り時間推定: {rem_time_h}h {rem_time_m}m {rem_time_s}s")
                self.tb_recoder.add_scalar("Train/EstimatedRemainingTime", remaining_time, epoch)

            self.train(train_loader, model, criterion, metrics, opt, epoch)

            val_loss = None # val_lossを初期化
            if epoch % self.config.VAL_INTERVAL == 0:
                # ▼▼▼▼▼ 修正箇所 ▼▼▼▼▼
                # すべてのランクがvalid関数を呼び出す
                val_loss = self.valid(val_loader, model, criterion, metrics, epoch)
                # ▲▲▲▲▲ 修正箇所 ▲▲▲▲▲

            # dist.barrier() はvalid関数内で実行されるので、ここでは不要な場合が多い
            # dist.barrier()

            # チェックポイントの保存やベストモデルの更新は rank 0 のみで行う
            if self.rank == 0:
                checkpoint = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": opt.optimizer.state_dict(),
                    "scheduler": opt.scheduler.state_dict(),
                }
                path = join(self.config.SAVE_DIR, "checkpoint.pth.tar")
                torch.save(checkpoint, path)

                # validが実行されたエポックでのみベストモデルを評価
                if val_loss is not None:
                    if val_loss < min_loss:
                        min_loss = val_loss
                        copy_path = join(self.config.SAVE_DIR, "model_best.pth.tar")
                        shutil.copy(path, copy_path)

    def train(self, train_loader, model, criterion, metrics, opt, epoch):
        model.train()
        info(f"Train Epoch[{epoch}/{self.config.END_EPOCH}]:len({len(train_loader)})")
        length = len(train_loader)
        losses_meter = {"TotalLoss": AverageMeter(f"Valid/TotalLoss")}
        for config in self.config.LOSS:
            losses_meter[config.NAME] = AverageMeter(f"Train/{config.NAME}")
        metric_meter = {}
        for config in self.config.METRICS:
            metric_meter[config.NAME] = AverageMeter(f"Train/{config.NAME}")
        batch_time_meter = AverageMeter("Train/BatchTime")
        start_time = time.time()
        time_recoder = time.time()
        scaler = torch.cuda.amp.GradScaler()
        for index, batch in enumerate(train_loader):
            device_id = self.rank
            batch = move_tensors_to_cuda(batch, device_id)
            if self.config.MIX_PRECISION:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(batch)
                    losses, name_to_loss = criterion(outputs)
                    name_to_measure = metrics(outputs)
                scaler.scale(losses).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            else:
                outputs = model(batch)
                losses, name_to_loss = criterion(outputs)
                name_to_measure = metrics(outputs)
                opt.zero_grad()
                losses.backward()
                opt.step()
            now = time.time()
            batch_time_meter.update(now - time_recoder)
            time_recoder = now
            losses_meter["TotalLoss"].update(losses.detach().item())
            for name, loss_item in name_to_loss:
                loss_item = loss_item.detach().item()
                losses_meter[name].update(loss_item)
            for name, measure_item in name_to_measure:
                measure_item = measure_item.detach().item()
                metric_meter[name].update(measure_item)

            if index % self.config.LOG_INTERVAL == 0 and self.rank == 0:
                info(f"Train Epoch[{epoch}/{self.config.END_EPOCH}, {index}/{length}]:")
                for name, meter in losses_meter.items():
                    info(f"    loss:    {name}: {meter.avg}")
                for name, measure in metric_meter.items():
                    info(f"    measure: {name}: {measure.avg}")
        
        if self.rank == 0:
            epoch_time = time.time() - start_time
            batch_time = batch_time_meter.avg
            info(
                f"Train Epoch[{epoch}/{self.config.END_EPOCH}]:time:epoch({epoch_time}),batch({batch_time})"
                f"lr({opt.get_lr()})"
            )
            self.tb_recoder.add_scalar(f"Train/EpochTime", epoch_time, epoch)
            self.tb_recoder.add_scalar(f"Train/BatchTime", batch_time, epoch)
            self.tb_recoder.add_scalar(f"Train/LR", opt.get_lr(), epoch)
            for name, meter in losses_meter.items():
                info(f"    loss:    {name}: {meter.avg}")
                self.tb_recoder.add_scalar(f"Train/{name}", meter.avg, epoch)
            for name, measure in metric_meter.items():
                info(f"    measure: {name}: {measure.avg}")
                self.tb_recoder.add_scalar(f"Train/{name}", measure.avg, epoch)
        opt.lr_schedule()

    def valid(self, valid_loader, model, criterion, metrics, epoch):
        model.eval()
        length = len(valid_loader)
        info(f"Valid Epoch[{epoch}/{self.config.END_EPOCH}] starting: length({length})")
        losses_meter = {"total": AverageMeter(f"Valid/TotalLoss")}
        for config in self.config.LOSS:
            losses_meter[config.NAME] = AverageMeter(f"Valid/{config.NAME}")
        metric_meter = {}
        for config in self.config.METRICS:
            metric_meter[config.NAME] = AverageMeter(f"Valid/{config.NAME}")
        batch_time_meter = AverageMeter("Valid/BatchTime")
        time_recoder = time.time()
        start_time = time_recoder
        
        with torch.no_grad():
            for index, batch in enumerate(valid_loader):
                device_id = self.rank
                batch = move_tensors_to_cuda(batch, device_id)
                if self.config.MIX_PRECISION:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(batch)
                        losses, name_to_loss = criterion(outputs)
                        name_to_measure = metrics(outputs)
                else:
                    outputs = model(batch)
                    losses, name_to_loss = criterion(outputs)
                    name_to_measure = metrics(outputs)
                
                # 可視化処理はランク0のみで実行
                if self.config.VISUALIZATION.TEST_VIS and self.rank == 0:
                    save_folder = self.config.VISUALIZATION.folder
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    pred_frames = outputs['global_sharp_pred_frames']
                    for i, pred_frame in enumerate(pred_frames):
                        pred_img = pred_frame[0].cpu().numpy()
                        pred_img = (pred_img * 255).astype(np.uint8)
                        save_path = os.path.join(
                            self.config.VISUALIZATION.folder,
                            f"output_image_{index:04d}_{i:04d}.png"
                        )
                        pred_img_bgr = cv2.cvtColor(pred_img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_path, pred_img_bgr)
                        info(f"推論結果画像を保存しました: {save_path}")

                now = time.time()
                batch_time_meter.update(now - time_recoder)
                time_recoder = now
                
                loss = losses.detach().item() if isinstance(losses, torch.Tensor) else losses
                losses_meter["total"].update(loss)
                for name, loss_item in name_to_loss:
                    loss_item = loss_item.detach().item() if isinstance(loss_item, torch.Tensor) else loss_item
                    losses_meter[name].update(loss_item)
                for name, measure_item in name_to_measure:
                    measure_item = measure_item.detach().item() if isinstance(measure_item, torch.Tensor) else measure_item
                    metric_meter[name].update(measure_item)
                
                if index % self.config.LOG_INTERVAL == 0 and self.rank == 0:
                    info(f"Valid Epoch[{epoch}/{self.config.END_EPOCH}, {index}/{length}]:")
                    info(f"    batch-time: {batch_time_meter.avg}")
                    for name, meter in losses_meter.items():
                        info(f"    loss:    {name}: {meter.avg}")
                    for name, measure in metric_meter.items():
                        info(f"    measure: {name}: {measure.avg}")

        dist.barrier() # 各プロセスで合計する前に同期
        # 各プロセスの結果をランク0に集約して平均を計算
        for meter in losses_meter.values():
            meter_sum_tensor = torch.tensor(meter.sum, device=f'cuda:{self.rank}')
            meter_count_tensor = torch.tensor(meter.count, device=f'cuda:{self.rank}')

            dist.all_reduce(meter_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(meter_count_tensor, op=dist.ReduceOp.SUM)

            meter.sum = meter_sum_tensor.item()
            meter.count = meter_count_tensor.item()
        
        if self.rank == 0:
            epoch_time = time.time() - start_time
            batch_time = batch_time_meter.avg
            info(f"Valid Epoch[{epoch}/{self.config.END_EPOCH}]:" f"time:epoch({epoch_time}),batch({batch_time})")
            self.tb_recoder.add_scalar(f"Valid/EpochTime", epoch_time, epoch)
            self.tb_recoder.add_scalar(f"Valid/BatchTime", batch_time, epoch)
            for name, meter in losses_meter.items():
                info(f"    loss:    {name}: {meter.avg}")
                self.tb_recoder.add_scalar(f"Valid/{name}", meter.avg, epoch)
            for name, measure in metric_meter.items():
                info(f"    measure: {name}: {measure.avg}")
                self.tb_recoder.add_scalar(f"Valid/{name}", meter.avg, epoch)
        
        return losses_meter["total"].avg