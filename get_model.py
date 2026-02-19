import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# アップロードされたファイルから必要なクラスをインポート
from encoder.frame_event_multi_scale_fusion_v2 import FrameEventsMultiScaleEncoderV2
from motionflow.motionflow import FlowModel
from decoder.frame_interpolator import FrameInterpolator

class MyFrameInterpolationModel(nn.Module):
    def __init__(self, flow_params, interp_params):
        super(MyFrameInterpolationModel, self).__init__()

        # EFNetの初期化
        self.FrameEventsMultiScaleEncoderV2 = FrameEventsMultiScaleEncoderV2()
        
        # FlowModelの初期化
        flow_params['input_dim'] = 256
        self.flow_model = FlowModel(**flow_params)
        
        # FrameInterpolatorの初期化
        interp_params['input_dim'] = 256
        self.frame_interpolator = FrameInterpolator(**interp_params)
    

    def forward(self, batch):
        """
        モデルのフォワードパス
        
        Args:
            events (torch.Tensor): イベントデータ。
            frames (torch.Tensor): RSブレフレーム。
        
        Returns:
            output_frame (torch.Tensor): 補間されたフレーム。
        """
        # 1. EFNetを使用して特徴マップを抽出
        features = self.FrameEventsMultiScaleEncoderV2(batch["rolling_blur_frame_color"], batch["events"])
        
        B, C, H, W = features.shape

        # 2. FlowModelを使用してオプティカルフローを生成して、FrameInterpolatorを使用して最終的な補間フレームを生成
        # rsの場合
        optical_flow_rs = self.flow_model(
            features=features,
            query_timemap=batch["rs_sharp_timemap"]
        )
        rs_sharp_pred_frame = self.frame_interpolator(
            features=features,
            optical_flow=optical_flow_rs,
            query_timemap=batch["rs_sharp_timemap"]
        )       

        # gsの場合
        global_sharp_pred_frames = []
        for i in range(batch["gs_timemaps"].shape[1]):
            gs_timemap = batch["gs_timemaps"][:, i].unsqueeze(1)

            optical_flow_gs = self.flow_model(
                features=features,
                query_timemap=gs_timemap
            )        

            global_sharp_pred_frame = self.frame_interpolator(
                features=features,
                optical_flow=optical_flow_gs,
                query_timemap=gs_timemap
            )

            global_sharp_pred_frames.append(global_sharp_pred_frame)

        batch["global_sharp_pred_frames"] = global_sharp_pred_frames
        batch["rs_sharp_pred_frame"] = rs_sharp_pred_frame

        return batch

def get_model(config):
    return MyFrameInterpolationModel(
        flow_params=config.flow_params,
        interp_params=config.interp_params,
        )

