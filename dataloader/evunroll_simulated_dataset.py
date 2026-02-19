## シミュレーションデータ（合成データ）を使ったローリングシャッター画像とイベントカメラデータのPyTorchデータセット前処理
import logging
import os
import weakref
from os import listdir
from os.path import join

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from absl.logging import info
from einops import rearrange, repeat
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from dataloader.basic_batch import get_rs_deblur_inference_batch
from dataloader.events_to_frame import event_stream_to_frames

logging.getLogger("PIL").setLevel(logging.WARNING)


def get_evunroll_simulated_dataset_with_config(config):
    return get_evunroll_simulated_dataset(
        root=config.root,
        moments=config.events_moment,
        crop_size=config.crop_size,
    )


def get_evunroll_simulated_dataset(root, moments, crop_size):
    # Load train dataset
    
    train_image_root = join(root, "Gev-RS-360", "train")
    train_event_root = join(root, "Gev-RS-DVS", "train")
    train_dataset = _get_dataset(
        train_image_root, train_event_root, moments=moments, is_training=True, crop_size=crop_size
    )
    
    # Load test dataset
    test_image_root = join(root, "Gev-RS-360", "test")
    test_event_root = join(root, "Gev-RS-DVS", "test")
    test_dataset = _get_dataset(
        test_image_root, test_event_root, moments=moments, is_training=False, crop_size=crop_size
    )
    # Return
    #return test_dataset # test only
    return train_dataset, test_dataset


def _get_dataset(img_root, event_root, moments, is_training, crop_size):
    all_seqs = listdir(img_root)
    all_seqs.sort()
    seq_dataset_list = []
    for seq in all_seqs:
        video_dataset = EvUnRollSimulatedVideoDataset(img_root, event_root, seq, moments, is_training, crop_size)
        seq_dataset_list.append(video_dataset)
    return ConcatDataset(seq_dataset_list)


class EvUnRollSimulatedVideoDataset(Dataset):
    def __init__(self, img_root, event_root, seq_name, moments, is_training, crop_size):
        super().__init__()
        self.seq_name = seq_name
        self.seq = seq_name[: seq_name.rfind("_", 0, seq_name.rfind("_"))]
        self.is_training = is_training
        self.gt_fps = 5000

        self.blur_length = int(seq_name.split("_")[-1])
        self.rs_delay_length = int(seq_name.split("_")[-2])
        self.rs_length = self.rs_delay_length + self.blur_length - 1
        self.interval_length = 100
        self.moments = moments

        self.delay_time = int(self.rs_delay_length * 1e6 / self.gt_fps)
        self.whole_time = int((self.rs_length - 1) * 1e6 / self.gt_fps)
        self.exposure_time = int((self.blur_length - 1) * 1e6 / self.gt_fps)
        self.interval_time = int(self.interval_length * 1e6 / self.gt_fps)
        self.img_folder = join(img_root, seq_name, "rs_blur")
        self.mid_gt_folder = join(img_root, seq_name, "gtx3_2")
        self.rs_sharp_folder = join(img_root, seq_name, "rs_sharp")

        self.event_file = join(event_root, self.seq, f"{self.seq}.h5")

        self.input_list = sorted(os.listdir(self.img_folder))
        self.mid_gt_list = sorted(os.listdir(self.mid_gt_folder))
        self.num_input = len(self.input_list)

        self.crop_size = crop_size
        im0 = cv2.imread(join(self.img_folder, self.input_list[0]))
        # 360, 640
        self.height, self.width, _ = im0.shape
        self.outsize = self.crop_size if self.is_training else (self.width, self.height)

        self.frame_count = 3 # global sharp frame count

        self.h5_file = None
        with h5py.File(self.event_file, "r") as f:
            img_to_idx = f["img_to_idx"]
            self.ev_idx = np.stack(
                [
                    img_to_idx[:: self.interval_length][: self.num_input],
                    img_to_idx[self.rs_length - 1 :: self.interval_length][: self.num_input],
                ],
                axis=1,
            )
        # When the object is closed, the close_callback function will be called.
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5_file)

        # Image to Tensor
        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def close_callback(h5f):
        if h5f is not None:
            h5f.close()

    def __len__(self):
        return self.num_input
    
    def get_rs_sharp_timemap(self):
        rs_sharp_row_stamp = torch.arange(self.height, dtype=torch.float32)/(self.height - 1)*self.rs_delay_length/self.rs_length + self.blur_length /(2 * self.rs_length)
        rs_sharp_timemap = repeat(rs_sharp_row_stamp, 'h -> h w', w = self.width)
        rs_sharp_timemap = rs_sharp_timemap.unsqueeze(0)
        return rs_sharp_timemap
    
    """
    def generate_gs_timemaps(self):
        # 1. まず、基準となる値のシーケンスを作成します
        # [1/(F+1), 2/(F+1), ..., F/(F+1)] というテンソルを生成
        # 形状: (frame_count,)
        base_values = torch.arange(1, 1 + 1, dtype=torch.float32) / (1 + 1) # 2個目と4個目の1はframe_count
        # 2. 形状を (frame_count, height, width) に拡張します
        # .viewで形状を(frame_count, 1, 1)に変形し、.expandでHとWの次元に値を引き伸ばす
        gs_timemaps = base_values.view(1, 1, 1).expand(1, self.height, self.width) # 1個目と4個目の1はframe_count
        
        return gs_timemaps
    """
    """
    def generate_gs_timemaps(self):
    # 特定のframe_count（1, 3, 5）の場合
        if self.frame_count in [1, 3, 5]:
            # まず、基準となる全9フレームのgs_timemapsを生成します。
            gs_timemaps_9 = torch.arange(1, 9 + 1, dtype=torch.float32).view(9, 1, 1).expand(9, self.height, self.width) / (9 + 1)
            
            if self.frame_count == 1:
                gs_timemaps = gs_timemaps_9[4].unsqueeze(0)
            elif self.frame_count == 3:
                gs_timemaps = gs_timemaps_9[[2, 4, 6]]
            elif self.frame_count == 5:
                # フレーム0, 2, 4, 6, 8を選択（インデックス0, 2, 4, 6, 8）
                gs_timemaps = gs_timemaps_9[[0, 2, 4, 6, 8]]
        # その他のframe_countの場合
        else:
            # ご希望の計算式を適用し、新しいgs_timemapsを生成します。
            gs_timemaps = torch.arange(1, self.frame_count + 1, dtype=torch.float32).view(self.frame_count, 1, 1).expand(self.frame_count, self.height, self.width) / (self.frame_count + 1)
        return gs_timemaps
    """
    def generate_gs_timemaps(self):
        if self.frame_count in [1, 3, 5, 9]:
            if self.frame_count == 1:
                timestamps = [0.5]
            elif self.frame_count == 3:
                timestamps = [0.1, 0.5, 0.9]
            elif self.frame_count == 5:
                timestamps = [0.1, 0.15, 0.5, 0.85, 0.9]
            elif self.frame_count == 9:
                timestamps = [0.1, 0.125, 0.15, 0.175, 0.5, 0.825, 0.85, 0.875, 0.9]
            # --- 定義されたタイムスタンプからテンソルを生成 ---
            # Pythonリストをtorchテンソルに変換
            gs_timemaps = torch.tensor(timestamps, dtype=torch.float32)
            # (frame_count) -> (frame_count, 1, 1) -> (frame_count, height, width) に拡張
            gs_timemaps = gs_timemaps.view(self.frame_count, 1, 1).expand(self.frame_count, self.height, self.width)
        # その他のframe_countの場合
        else:
            # 従来の等間隔な計算式を適用します
            gs_timemaps = torch.arange(1, self.frame_count + 1, dtype=torch.float32).view(self.frame_count, 1, 1).expand(self.frame_count, self.height, self.width) / (self.frame_count + 1)
            
        return gs_timemaps
    

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.event_file, "r")
            self.events = self.h5_file["events"]
            self.img_to_idx = self.h5_file["img_to_idx"]
        # load image
        image_input = join(self.img_folder, self.input_list[idx])
        image_input = Image.open(image_input).convert("RGB")
        image_input = self.to_tensor(image_input)

       # 修正箇所: mid_gt_folderから単一の画像、または複数画像を読み込む
        gt_path = join(self.mid_gt_folder, self.mid_gt_list[idx])
        # パスがディレクトリかどうかを判定
        if os.path.isdir(gt_path):
            # パスがディレクトリの場合（中身が複数枚の画像）
            gt_image_files = sorted(os.listdir(gt_path))
            gt_image_list = []
            for gt_file_name in gt_image_files:
                gt_image_path = join(gt_path, gt_file_name)
                gt_image = Image.open(gt_image_path).convert("RGB")
                gt_image_list.append(self.to_tensor(gt_image))
        else:
            # パスがファイルの場合（中身が1枚の画像）
            gt_image = Image.open(gt_path).convert("RGB")
            gt_image_list = [self.to_tensor(gt_image)]

        rs_sharp_image = join(self.rs_sharp_folder, self.input_list[idx])
        rs_sharp_image = Image.open(rs_sharp_image).convert("RGB")
        rs_sharp_image = self.to_tensor(rs_sharp_image)

        # load event
        # N x 4
        event_input = self.events[self.ev_idx[idx, 0] : self.ev_idx[idx, 1], :].copy().astype(np.int32)
        event_input[:, 0] = event_input[:, 0] - self.interval_time * idx
        event_moments = event_stream_to_frames(
            event_input,
            moments=self.moments,
            resolution=(self.height, self.width),
            positive=1,
            negative=0,
        )
        # M, 2, H, W
        event_moments = np.stack(event_moments, axis=0)
        event_moments = torch.from_numpy(event_moments).float()
        # M, 2, H, W -> M, H, W
        event_moments = event_moments[:, 0, :, :] - event_moments[:, 1, :, :]
        event_moments = torch.tanh(event_moments)

        # get time map
        rs_sharp_timemap = self.get_rs_sharp_timemap()
        gs_timemaps = self.generate_gs_timemaps()

        # training crop
        image_input, rs_sharp_image, gt_image_list, event_moments, rs_sharp_timemap, gs_timemaps = self._random_crop(image_input, rs_sharp_image, gt_image_list, event_moments, rs_sharp_timemap, gs_timemaps)

        # gt_image = gt_image.unsqueeze(0) # 3, H, W -> 1, 3, H, W
        # build a sample
        batch = get_rs_deblur_inference_batch()
        batch["video_name"] = self.seq_name
        batch["rolling_blur_frame_name"] = self.input_list[idx].split("/")[-1].split(".")[0]
        batch["rolling_blur_frame_color"] = image_input
        batch["rolling_sharp_frame_color"] = rs_sharp_image
        batch["events"] = event_moments
        batch["global_sharp_frames"] = gt_image_list
        batch["rs_sharp_timemap"] = rs_sharp_timemap
        batch["gs_timemaps"] = gs_timemaps
        return batch

    def _random_crop(self, image_input, rs_sharp_image, gt_image_list, event_moments, rs_sharp_timemap, gs_timemaps):
        # random crop
        # 360, 360
        h, w = self.crop_size
        # 360, 640
        th, tw = self.height, self.width
        if self.is_training:
            x1 = np.random.randint(0, tw - w + 1)
        else:
            x1 = (tw - w) // 2
        image_input = image_input[:, :, x1 : x1 + w]
        rs_sharp_image = rs_sharp_image[:, :, x1 : x1 + w]
       
        # 修正箇所: リスト内の各画像にクロップを適用
        cropped_gt_list = []
        for gt_image in gt_image_list:
            cropped_gt_list.append(gt_image[:, :, x1 : x1 + w])

        event_moments = event_moments[:, :, x1 : x1 + w]
        rs_sharp_timemap = rs_sharp_timemap[:, :, x1 : x1 + w]
        gs_timemaps = gs_timemaps[:, :, x1 : x1 + w]
        return image_input, rs_sharp_image, cropped_gt_list, event_moments, rs_sharp_timemap, gs_timemaps
