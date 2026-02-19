import os  # OS操作用
import random  # 乱数生成用
import shutil  # ファイルコピー用
import time  # 時間計測用
from collections import OrderedDict  # 順序付き辞書
from os.path import isfile, join  # ファイルパス操作

import cv2  # 画像処理用
import numpy as np  # 数値計算用
import torch  # PyTorch本体
import torch.nn as nn  # ニューラルネットワークモジュール
from absl.logging import debug, flags, info  # ロギング
from pudb import set_trace  # デバッグ用

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.testing._internal.common_quantization import AverageMeter  # メーターツール
from torch.utils.data import DataLoader  # データローダ
from torch.utils.tensorboard import SummaryWriter  # TensorBoard記録用

from optimizer import Optimizer  # オプティマイザ
from dataloader.evunroll_simulated_dataset import get_evunroll_simulated_dataset_with_config # データセット取得
from loss import get_loss, get_metric  # 損失・評価指標取得
from get_model import get_model  # モデル取得
from visualize import get_visulization  # 可視化ツール取得

FLAGS = flags.FLAGS  # コマンドラインフラグ

def move_tensors_to_cuda(dictionary_of_tensors):
    # テンソルや辞書をCUDAに転送
    if isinstance(dictionary_of_tensors, dict):
        return {key: move_tensors_to_cuda(value) for key, value in dictionary_of_tensors.items()} 
    if isinstance(dictionary_of_tensors, torch.Tensor):
        return dictionary_of_tensors.cuda(non_blocking=True)
    else:
        return dictionary_of_tensors


# 新しく追加した関数：モデルのパラメータ数を計算
def get_model_parameters_count(model):
    """モデルのパラメータ数を計算する関数"""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

"""
# 並列学習の管理クラス
・このクラスは、学習の設定、データセット、モデル、損失関数、評価指標、オプティマイザを管理し、学習と検証のループを実行する
・TensorBoardを使用して学習の進捗を記録する
・学習中は、各エポックごとにモデルの状態を保存し、検証を行い、最良のモデルを保存
"""
class ParallelLaunch:
    def __init__(self, config):
        """並列学習のメインクラス。runメソッドがエントリポイント。

        Args:
            config (EasyDict): 学習実験の設定
        """
        os.environ["MASTER_ADDR"] = "localhost"  # 分散学習用アドレス
        os.environ["MASTER_PORT"] = "6666"  # 分散学習用ポート
        info(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")  # ログ出力
        info(f"MASTER_PORT: {os.environ['MASTER_PORT']}")  # ログ出力
        # 0. config
        self.config = config  # 設定保存
        # # 1. init environment
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True
        # 1.1 init global random seed
        torch.manual_seed(config.SEED)  # PyTorch乱数シード
        torch.cuda.manual_seed(config.SEED)  # CUDA乱数シード
        random.seed(config.SEED)  # Python乱数シード
        np.random.seed(config.SEED)  # NumPy乱数シード
        # 1.2 init the tensorboard log dir
        self.tb_recoder = SummaryWriter(FLAGS.log_dir)  # TensorBoard記録用
        self.start_time = time.time() # 訓練全体の開始時刻
        # 2. device
        self.visualizer = None  # 可視化ツール初期化
        # if config.VISUALIZE:
            # self.visualizer = get_visulization(config.VISUALIZATION)  # 可視化ツール取得

    # このクラスの`run`メソッドがエントリポイントとなり、学習を開始する
    def run(self):
        # 0. Init
        train_dataset, val_dataset = get_evunroll_simulated_dataset_with_config(self.config.DATASET)  # データセット取得
        model = get_model(self.config.MODEL)  # モデル取得

        # ここから追加: モデルのパラメータ数を計算して表示
        num_params = get_model_parameters_count(model)
        info(f"モデルのパラメータ総数: {num_params}")

        criterion = get_loss(self.config.LOSS)  # 損失関数取得
        metrics = get_metric(self.config.METRICS)  # 評価指標取得
        opt = Optimizer(self.config.OPTIMIZER, model)  # オプティマイザ初期化
        # 1. Build model
        if self.config.IS_CUDA:
            model = nn.DataParallel(model)  # 複数GPU対応
            model = model.cuda()  # CUDAへ転送
            # model = model.to(torch.distributed.get_rank())
            # model = DDP(model, device_ids=[torch.distributed.get_rank()])
            # model = model.cuda()

        if self.config.RESUME.PATH:
            # チェックポイントから再開
            if not isfile(self.config.RESUME.PATH):
                raise ValueError(f"File not found, {self.config.RESUME.PATH}")
            if self.config.IS_CUDA:
                checkpoint = torch.load(
                    self.config.RESUME.PATH,
                    map_location=lambda storage, loc: storage.cuda(0),
                )
            else:
                checkpoint = torch.load(self.config.RESUME.PATH, map_location=torch.device("cpu"))
                new_state_dict = OrderedDict()
                for k, v in checkpoint["state_dict"].items():
                    name = k[7:]  # 'module.'を除去
                    new_state_dict[name] = v
                checkpoint["state_dict"] = new_state_dict

            if self.config.RESUME.SET_EPOCH:
                self.config.START_EPOCH = checkpoint["epoch"]  # エポック復元
                opt.optimizer.load_state_dict(checkpoint["optimizer"])  # オプティマイザ復元
                opt.scheduler.load_state_dict(checkpoint["scheduler"])  # スケジューラ復元

            if self.config.RESUME_STRICT:
                model.load_state_dict(checkpoint["state_dict"])  # 厳密に復元
            else:
                model.load_state_dict(checkpoint["state_dict"], strict=False)  # 厳密でなく復元

        # 2. Build Dataloader
        # train_sampler = DistributedSampler(train_dataset)
        # val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_loader = DataLoader(
            dataset=train_dataset,
            # sampler=train_sampler,
            batch_size=self.config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.JOBS,
            pin_memory=True,
            drop_last=True,
        )  # 学習用データローダ
        val_loader = DataLoader(
            dataset=val_dataset,
            # sampler=val_sampler,
            batch_size=self.config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.JOBS,
            pin_memory=True,
            drop_last=True,
        )  # 検証用データローダ
        # 3. if test only
        if self.config.TEST_ONLY:
            self.valid(val_loader, model, criterion, metrics, 0)  # テストのみ
            return
        # 4. train
        min_loss = 123456789.0  # 最小損失初期化
        for epoch in range(self.config.START_EPOCH, self.config.END_EPOCH):
            # train_sampler.set_epoch(epoch)
            # val_sampler.set_epoch(epoch)
            # info(f"Epoch[{epoch}/{self.config.END_EPOCH}] starting...")
            self.train(train_loader, model, criterion, metrics, opt, epoch)  # 学習
            
            # ここから追加
            # 訓練全体の経過時間と残り時間を計算
            total_epochs = self.config.END_EPOCH - self.config.START_EPOCH
            elapsed_epochs = epoch - self.config.START_EPOCH + 1

            # エポック平均時間を計算
            elapsed_time_total = time.time() - self.start_time
            avg_time_per_epoch = elapsed_time_total / elapsed_epochs

            # 残り時間を計算
            remaining_epochs = total_epochs - elapsed_epochs
            remaining_time = remaining_epochs * avg_time_per_epoch

            # 時間を秒から時/分/秒に変換
            rem_time_h = int(remaining_time // 3600)
            rem_time_m = int((remaining_time % 3600) // 60)
            rem_time_s = int(remaining_time % 60)

            info(f"残り時間推定: {rem_time_h}h {rem_time_m}m {rem_time_s}s")
            self.tb_recoder.add_scalar("Train/EstimatedRemainingTime", remaining_time, epoch) # TensorBoard記録
            # ここまで追加

            # save checkpoint
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": opt.optimizer.state_dict(),
                "scheduler": opt.scheduler.state_dict(),
            }  # チェックポイント作成
            path = join(self.config.SAVE_DIR, "checkpoint.pth.tar")  # 保存パス
            time.sleep(1)  # 少し待つ
            # valid
            if epoch % self.config.VAL_INTERVAL == 0:
                torch.save(checkpoint, path)  # チェックポイント保存
                val_loss = self.valid(val_loader, model, criterion, metrics, epoch)  # 検証
                if val_loss < min_loss:
                    min_loss = val_loss  # 最小損失更新
                    copy_path = join(self.config.SAVE_DIR, "model_best.pth.tar")
                    shutil.copy(path, copy_path)  # ベストモデル保存
            # train
            if epoch % self.config.MODEL_SANING_INTERVAL == 0:
                path = join(
                    self.config.SAVE_DIR,
                    f"checkpoint-{str(epoch).zfill(3)}.pth.tar",
                )
                torch.save(checkpoint, path)  # 定期的に保存

    # 学習ループ
    def train(self, train_loader, model, criterion, metrics, opt, epoch):
        model.train()  # 学習モード
        info(f"Train Epoch[{epoch}/{self.config.END_EPOCH}]:len({len(train_loader)})")  # ログ
        length = len(train_loader)  # バッチ数
        # 1. init meter
        losses_meter = {"TotalLoss": AverageMeter(f"Valid/TotalLoss")}  # 損失メータ初期化
        for config in self.config.LOSS:
            losses_meter[config.NAME] = AverageMeter(f"Train/{config.NAME}")  # 各損失メータ
        metric_meter = {}
        for config in self.config.METRICS:
            metric_meter[config.NAME] = AverageMeter(f"Train/{config.NAME}")  # 各評価メータ
        batch_time_meter = AverageMeter("Train/BatchTime")  # バッチ時間メータ
        # 2. start a training epoch
        start_time = time.time()  # エポック開始時刻
        time_recoder = time.time()  # バッチ開始時刻
        scaler = torch.cuda.amp.GradScaler()  # AMP用スケーラ
        for index, batch in enumerate(train_loader):
            if self.config.IS_CUDA:
                batch = move_tensors_to_cuda(batch)  # CUDA転送
            if self.config.MIX_PRECISION:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # 混合精度
                    outputs = model(batch)  # 順伝播
                    losses, name_to_loss = criterion(outputs)  # 損失計算
                    # 2.1 forward
                    name_to_measure = metrics(outputs)  # 評価指標計算
                scaler.scale(losses).backward()  # 逆伝播
                scaler.step(opt)  # パラメータ更新
                scaler.update()  # スケーラ更新
                opt.zero_grad()  # 勾配初期化
            else:
                outputs = model(batch)  # 順伝播
                losses, name_to_loss = criterion(outputs)  # 損失計算
                # 2.1 forward
                name_to_measure = metrics(outputs)  # 評価指標計算
                # 2.2 backward
                opt.zero_grad()  # 勾配初期化
                losses.backward()  # 逆伝播
                # 2.3 update weights
                # clip the grad
                # clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                opt.step()  # パラメータ更新
            # 2.4 update measure
            # 2.4.1 time update
            now = time.time()  # 現在時刻
            batch_time_meter.update(now - time_recoder)  # バッチ時間更新
            time_recoder = now  # バッチ開始時刻更新
            # 2.4.2 loss update
            losses_meter["TotalLoss"].update(losses.detach().item())  # 総損失更新
            for name, loss_item in name_to_loss:
                loss_item = loss_item.detach().item()
                losses_meter[name].update(loss_item)  # 各損失更新
            # 2.4.3 measure update
            for name, measure_item in name_to_measure:
                measure_item = measure_item.detach().item()
                metric_meter[name].update(measure_item)  # 各評価指標更新
            # 2.5 log
            if index % self.config.LOG_INTERVAL == 0:
                info(f"Train Epoch[{epoch}/{self.config.END_EPOCH}, {index}/{length}]:")
                for name, meter in losses_meter.items():
                    info(f"    loss:    {name}: {meter.avg}")  # 損失ログ
                for name, measure in metric_meter.items():
                    info(f"    measure: {name}: {measure.avg}")  # 評価指標ログ
        # 3. record a training epoch
        # 3.1 record epoch time
        epoch_time = time.time() - start_time  # エポック時間
        batch_time = batch_time_meter.avg  # 平均バッチ時間
        info(
            f"Train Epoch[{epoch}/{self.config.END_EPOCH}]:time:epoch({epoch_time}),batch({batch_time})"
            f"lr({opt.get_lr()})"
        )
        self.tb_recoder.add_scalar(f"Train/EpochTime", epoch_time, epoch)  # TensorBoard記録
        self.tb_recoder.add_scalar(f"Train/BatchTime", batch_time, epoch)
        self.tb_recoder.add_scalar(f"Train/LR", opt.get_lr(), epoch)
        for name, meter in losses_meter.items():
            info(f"    loss:    {name}: {meter.avg}")
            self.tb_recoder.add_scalar(f"Train/{name}", meter.avg, epoch)
        for name, measure in metric_meter.items():
            info(f"    measure: {name}: {measure.avg}")
            self.tb_recoder.add_scalar(f"Train/{name}", measure.avg, epoch)
        # adjust learning rate
        opt.lr_schedule()  # 学習率調整

    """
    # 検証ループ
    ・このメソッドは、検証データセットを使用してモデルの性能を評価
    ・各エポックの検証結果をTensorBoardに記録し、平均損失を返す
    # 引数:
    ・valid_loader: 検証用データローダ
    ・model: 検証するモデル
    ・criterion: 損失関数
    ・metrics: 評価指標
    ・epoch: 現在のエポック番号
    # 戻り値:平均損失 (float): 検証データセットに対するモデルの平均損失
    """
    def valid(self, valid_loader, model, criterion, metrics, epoch):
        model.eval()  # 評価モード
        length = len(valid_loader)  # バッチ数
        info(f"Valid Epoch[{epoch}/{self.config.END_EPOCH}] starting: length({length})")
        # 1. init meter
        losses_meter = {"total": AverageMeter(f"Valid/TotalLoss")}  # 総損失メータ
        for config in self.config.LOSS:
            losses_meter[config.NAME] = AverageMeter(f"Valid/{config.NAME}")  # 各損失メータ
        metric_meter = {}
        for config in self.config.METRICS:
            metric_meter[config.NAME] = AverageMeter(f"Valid/{config.NAME}")  # 各評価メータ
        batch_time_meter = AverageMeter("Valid/BatchTime")  # バッチ時間メータ
        # 2. start a validating epoch
        time_recoder = time.time()  # バッチ開始時刻
        start_time = time_recoder  # エポック開始時刻
        for index, batch in enumerate(valid_loader):
            if self.config.IS_CUDA:
                batch = move_tensors_to_cuda(batch)  # CUDA転送
            with torch.no_grad():  # 勾配計算なし
                if self.config.MIX_PRECISION:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # 混合精度
                        outputs = model(batch)  # 順伝播
                        losses, name_to_loss = criterion(outputs)  # 損失計算
                        # 2.2. recorder
                        name_to_measure = metrics(outputs)  # 評価指標計算
                else:
                    outputs = model(batch)  # 順伝播
                    losses, name_to_loss = criterion(outputs)  # 損失計算
                    # 2.2. recorder
                    name_to_measure = metrics(outputs)  # 評価指標計算
            
            
            # 2.3 visualization
            #if self.visualizer:
            #    self.visualizer.visualize(outputs)  # 過程可視化
            if self.config.VISUALIZATION.TEST_VIS:
                # 新しい画像を保存するための、このラウンドの個別フォルダを作成
                # 既存のコードがバッチごとにインデックスを持つと仮定
                save_folder = self.config.VISUALIZATION.folder

                # フォルダが存在しない場合は作成
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                pred_frames = outputs['global_sharp_pred_frames']
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
                        f"output_image_{index:04d}_{i:04d}.png"
                    )
                    
                    # BGR形式に変換（OpenCVのデフォルト）
                    pred_img_bgr = cv2.cvtColor(pred_img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                    
                    # 画像保存
                    cv2.imwrite(save_path, pred_img_bgr)
                    
                    info(f"推論結果画像を保存しました: {save_path}")
                 
                    
            # 2.4. update measure
            now = time.time()  # 現在時刻
            batch_time_meter.update(now - time_recoder)  # バッチ時間更新
            time_recoder = now  # バッチ開始時刻更新
            loss = losses.detach().item() if isinstance(losses, torch.Tensor) else losses
            losses_meter["total"].update(loss)  # 総損失更新
            for name, loss_item in name_to_loss:
                loss_item = loss_item.detach().item() if isinstance(loss_item, torch.Tensor) else loss_item
                losses_meter[name].update(loss_item)  # 各損失更新
            for name, measure_item in name_to_measure:
                measure_item = measure_item.detach().item() if isinstance(measure_item, torch.Tensor) else measure_item
                metric_meter[name].update(measure_item)  # 各評価指標更新
            if index % self.config.LOG_INTERVAL == 0:
                info(f"Valid Epoch[{epoch}/{self.config.END_EPOCH}, {index}/{length}]:")
                info(f"    batch-time: {batch_time_meter.avg}")
                for name, meter in losses_meter.items():
                    info(f"    loss:    {name}: {meter.avg}")
                for name, measure in metric_meter.items():
                    info(f"    measure: {name}: {measure.avg}")
        # 3. record a training epoch
        # 3.1 record epoch time
        epoch_time = time.time() - start_time  # エポック時間
        batch_time = batch_time_meter.avg  # 平均バッチ時間
        info(f"Valid Epoch[{epoch}/{self.config.END_EPOCH}]:" f"time:epoch({epoch_time}),batch({batch_time})")
        self.tb_recoder.add_scalar(f"Valid/EpochTime", epoch_time, epoch)  # TensorBoard記録
        self.tb_recoder.add_scalar(f"Valid/BatchTime", batch_time, epoch)
        for name, meter in losses_meter.items():
            info(f"    loss:    {name}: {meter.avg}")
            self.tb_recoder.add_scalar(f"Valid/{name}", meter.avg, epoch)
        for name, measure in metric_meter.items():
            info(f"    measure: {name}: {measure.avg}")
            self.tb_recoder.add_scalar(f"Valid/{name}", measure.avg, epoch)
        return losses_meter["total"].avg  # 平均損失を返す
