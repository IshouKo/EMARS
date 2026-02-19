import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# SIRENクラスはSIREN.pyからインポートされることを想定しています。
from decoder.SIREN import Siren
# warpgrid関数はwarplayer.pyからインポートされることを想定しています。
from decoder.warplayer import Warper

class FrameInterpolator(nn.Module):
    def __init__(self, input_dim, hidden_features, hidden_layers, out_features):
        super(FrameInterpolator, self).__init__()

        # フレーム生成のためのSIRENネットワーク
        # 入力: ワープされた特徴量 + タイムスタンプ
        frame_imnet_in_features = input_dim + 1
        self.frame_imnet = Siren(
            in_features=frame_imnet_in_features,
            out_features=out_features,  # 例: RGBの場合は3
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            outermost_linear=True
        )
        self.warper = Warper()

    def forward(self, features, optical_flow, query_timemap):
        """
        ワープとSirenネットワークを使用して、任意時刻のフレームを生成します。

        入力:
            features (torch.Tensor): 特徴マップ。形状: (B, C, H, W)
            optical_flow (torch.Tensor): 開始フレームからターゲットフレームへのオプティカルフロー。形状: (B, 2, H, W)
            query_timestamp (torch.Tensor): 目的の時刻のタイムスタンプ。形状: (B, 1, H, W)

        Returns:
            torch.Tensor: 生成されたフレーム。形状: (B, num_channels, H, W)
        """
        B, C, H, W = features.shape
        
        # warpgrid関数を使い、ワープ後の座標グリッドを取得
        grid = self.warper.warpgrid(optical_flow)
        
        # ワープされたグリッドを使って、特徴マップから値をサンプリング
        warped_features = F.grid_sample(
            input=features, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        # Sirenへの入力を作成: ワープされた特徴量 + タイムスタンプ
        device = features.device
        query_timemap = query_timemap.to(device)
        input_for_siren = torch.cat([warped_features, query_timemap], dim=1)
        input_for_siren = input_for_siren.view(B, C + 1, H * W)
        input_for_siren = input_for_siren.permute(0, 2, 1)

        # 中間フレームの生成
        output_frame = self.frame_imnet(input_for_siren.reshape(B * H * W, -1)).reshape(B, H * W, -1)

        # 形状を画像形式に戻す
        output_frame = output_frame.permute(0, 2, 1).view(B, 3, H, W)
        
        return output_frame