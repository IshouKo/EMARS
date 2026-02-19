import torch
import h5py
from torch.utils.data import Dataset, DataLoader,ConcatDataset
import os
import weakref
import cv2
import numpy as np
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange, repeat
import logging
import weakref
from os import listdir
from os.path import join
from absl.logging import info
from PIL import Image
from torchvision.transforms import transforms

from dataloader.basic_batch import get_rs_deblur_inference_batch
from dataloader.events_to_frame import event_stream_to_frames

logging.getLogger("PIL").setLevel(logging.WARNING)

class Sequence_Real(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.fps = cfg.fps
        self.idx = cfg.idx
        
        self.blur_length = 0
        self.rs_delay_length = 100
        self.rs_length = self.rs_delay_length + self.blur_length - 1
        self.interval_length = 100

        self.exposure_time = 1435
        self.delay_time = 34000
        self.whole_time = self.exposure_time + self.delay_time 
        self.interval_time = int(1e6/self.fps)

        self.img_folder = os.path.join(cfg.root, 'image')
        self.img_list = sorted(os.listdir(self.img_folder))
        self.event_file = os.path.join(cfg.root, f'04.npz')
        
        self.num_input= len(self.img_list)

        im0 = cv2.imread(os.path.join(self.img_folder, self.img_list[0]))
        self.height,self.width,_ = im0.shape

        self.ev_idx = None
        self.events = None

        self.crop_size = cfg.crop_size

        self.moments = cfg.events_moment

        self.to_tensor = transforms.ToTensor()
        self.frame_count = 1
        self.H = 720
        self.W = 1280

    def __len__(self):
        return 1

    def get_rs_sharp_timemap(self):
        rs_sharp_row_stamp = torch.arange(self.height, dtype=torch.float32)/(self.height - 1)*self.rs_delay_length/self.rs_length + self.blur_length /(2 * self.rs_length)
        rs_sharp_timemap = repeat(rs_sharp_row_stamp, 'h -> h w', w = self.width)
        rs_sharp_timemap = rs_sharp_timemap.unsqueeze(0)
        return rs_sharp_timemap

    def generate_gs_timemaps(self):
        if self.frame_count in [1, 3, 5, 9]:
            if self.frame_count == 1:
                timestamps = [0.5]
            elif self.frame_count == 3:
                timestamps = [0.1, 0.5, 0.9]
            elif self.frame_count == 5:
                timestamps = [0.1, 0.125, 0.5, 0.875, 0.9]
            elif self.frame_count == 9:
                timestamps = [0.1, 0.125, 0.15, 0.175, 0.5, 0.825, 0.85, 0.875, 0.9]
            # --- 定義されたタイムスタンプからテンソルを生成 ---
            # Pythonリストをtorchテンソルに変換
            gs_timemaps = torch.tensor(timestamps, dtype=torch.float32)
            # (frame_count) -> (frame_count, 1, 1) -> (frame_count, height, width) に拡張
            gs_timemaps = gs_timemaps.view(self.frame_count, 1, 1).expand(self.frame_count, self.H, self.W)
        # その他のframe_countの場合
        else:
            # 従来の等間隔な計算式を適用します
            gs_timemaps = torch.arange(1, self.frame_count + 1, dtype=torch.float32).view(self.frame_count, 1, 1).expand(self.frame_count, self.H, self.W) / (self.frame_count + 1)
            
        return gs_timemaps

    def get_event(self,idx):
        if self.ev_idx is None:
            if self.events.ndim == 1:
                et = self.events['t']
                ex = self.events['x']
                ey = self.events['y']
                ep = self.events['p']
                self.events = np.stack([et,ex,ey,ep], axis=1) # このコードは、各フィールド（t, x, y, p）を抽出し、np.stack を使って通常のNumPy配列に変換。この変換後の self.events の形状は、(N, 4) となり。ここで N はイベントの総数
            self.ev_idx = []
            ev_start_idx = 0
            ev_end_idx = 0
            for i in range(self.num_input):
                start_t = self.interval_time * i
                end_t = start_t + self.whole_time
                
                ev_start_idx = ev_end_idx
                while self.events[ev_start_idx,0] < start_t:
                    ev_start_idx += 1
                ev_end_idx = ev_start_idx
                while self.events[ev_end_idx,0] < end_t:
                    ev_end_idx += 1
                self.ev_idx.append((ev_start_idx, ev_end_idx))
        
        start_idx, end_idx = self.ev_idx[self.idx]
        event = self.events[start_idx:end_idx].copy() # 元の (N, 4) の配列から、特定の時間範囲内のイベントだけが抜き出されます。
        event[:,0] = event[:,0] - self.interval_time * self.idx # 切り出されたイベントデータの時間軸の値を変更し,これにより、各サンプル（idx）のイベント時間がゼロから始まるように正規化され
        return event
    
    def __getitem__(self,idx):
        if self.events is None:
            self.events = np.load(self.event_file)['event']
        # load image
        image_input = join(self.img_folder, self.img_list[self.idx])
        image_input = Image.open(image_input).convert("RGB")
        image_input = self.to_tensor(image_input)

        event_input = self.get_event(self.idx)

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

        gs_timemaps = self.generate_gs_timemaps()
        rs_sharp_timemap = self.get_rs_sharp_timemap()

        # training crop
        image_input, event_moments, rs_sharp_timemap, gs_timemaps = self._random_crop(image_input, event_moments, rs_sharp_timemap, gs_timemaps)

        # gt_image = gt_image.unsqueeze(0) # 3, H, W -> 1, 3, H, W
        # build a sample
        batch = get_rs_deblur_inference_batch()
        batch["rolling_blur_frame_color"] = image_input
        batch["events"] = event_moments
        batch["rs_sharp_timemap"] = rs_sharp_timemap
        batch["gs_timemaps"] = gs_timemaps
        return batch

    def _random_crop(self, image_input, event_moments, rs_sharp_timemap, gs_timemaps):
        # random crop
        # 360, 360
        h, w = self.crop_size
        # 360, 640
        th, tw = self.height, self.width
        x1 = (tw - w) // 2
        image_input = image_input[:, :, x1 : x1 + w]
        event_moments = event_moments[:, :, x1 : x1 + w]
        rs_sharp_timemap = rs_sharp_timemap[:, :, x1 : x1 + w]
        gs_timemaps = gs_timemaps[:, :, x1 : x1 + w]
        return image_input, event_moments,rs_sharp_timemap, gs_timemaps

