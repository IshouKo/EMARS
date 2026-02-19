# launch_real_test.py
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

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.testing._internal.common_quantization import AverageMeter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from optimizer import Optimizer
from dataloader.real_dataset import Sequence_Real # real_datasetをインポート
from loss import get_loss, get_metric
from get_model import get_model
from visualize import get_visulization

import torch.distributed as dist

FLAGS = flags.FLAGS

def move_tensors_to_cuda(data, device):
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

class ParallelLaunchRealTest:
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
        info("Initializing dataset for real test...")
        # SimualtedDatasetではなく、RealDatasetを使用
        test_dataset = Sequence_Real(self.config.DATASET)
        
        if self.config.DISTRIBUTED:
        # DDPに対応したデータサンプラーを使用
            val_sampler = DistributedSampler(test_dataset, shuffle=False)
            val_shuffle = False
        else:
            val_sampler = None
            val_shuffle = False

        test_loader = DataLoader(
            dataset=test_dataset,
            sampler=val_sampler,
            batch_size=self.config.VAL_BATCH_SIZE,
            shuffle=val_shuffle,
            num_workers=self.config.JOBS,
            pin_memory=True,
            drop_last=True,
        )

        model = get_model(self.config.MODEL)
        info(f"モデルのパラメータ総数: {get_model_parameters_count(model)}")
        
        opt = Optimizer(self.config.OPTIMIZER, model)  # オプティマイザ初期化

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

            if self.config.RESUME.SET_EPOCH:
                self.config.START_EPOCH = checkpoint["epoch"]  # エポック復元
                opt.optimizer.load_state_dict(checkpoint["optimizer"])  # オプティマイザ復元
                opt.scheduler.load_state_dict(checkpoint["scheduler"])  # スケジューラ復元

            if self.config.RESUME_STRICT:
                model.load_state_dict(checkpoint["state_dict"])  # 厳密に復元
            else:
                model.load_state_dict(checkpoint["state_dict"], strict=False)  # 厳密でなく復元

        self.test(test_loader, model)

    def test(self, test_loader, model):
        model.eval()
        length = len(test_loader)
        info(f"Test starting: length({length})")
        
        for index, batch in enumerate(test_loader):
            device_id = self.rank
            batch = move_tensors_to_cuda(batch, device_id)
            
            with torch.no_grad():
                outputs = model(batch)
                print("モデル推論完了")
            
            # visualization(全てのフレーム)
            if self.config.VISUALIZATION.TEST_VIS and self.rank == 0:
                # 新しい画像を保存するための、このラウンドの個別フォルダを作成
                # 既存のコードがバッチごとにインデックスを持つと仮定
                save_folder = self.config.VISUALIZATION.folder

                # フォルダが存在しない場合は作成
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                pred_frames = outputs['global_sharp_pred_frames']
                print("推論結果のフレーム取得完了")
                # リスト内の各フレームをループで処理
                for i, pred_frame in enumerate(pred_frames):
                    # テンソルをNumPy配列に変換
                    # shapeは (1, C, H, W) と仮定
                    pred_img = pred_frame[0].cpu().numpy()
                    
                    # [0, 1]の範囲を[0, 255]に変換し、uint8型に
                    pred_img = (pred_img * 255).astype(np.uint8)
                    
                    # 画像のパスを設定
                    # 例: visulization/round_0001/frame_0000.png
                    save_path = os.path.join(
                        self.config.VISUALIZATION.folder,
                        f"output_image_{i:04d}.png"
                    )
                    
                    # BGR形式に変換（OpenCVのデフォルト）
                    pred_img_bgr = cv2.cvtColor(pred_img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                    
                    # 画像保存
                    cv2.imwrite(save_path, pred_img_bgr)
                    
                    print("推論結果画像を保存しました")