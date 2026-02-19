import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from motionflow.SIREN import Siren

class FlowModel(nn.Module):
    def __init__(self, input_dim, hidden_features, hidden_layers, out_features):
        super(FlowModel, self).__init__()

        flow_imnet_in_features = input_dim + 1

        self.flow_imnet = Siren(
            in_features=flow_imnet_in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            outermost_linear=True
        )

    def forward(self, features, query_timemap):
        """
        オプティカルフローを生成します。

        入力:
            features (torch.Tensor): 空間時間情報を保持する特徴マップ。形状: (B, C, H, W)
            query_timestamp (torch.Tensor): オプティカルフローを生成したい任意の時刻のタイムスタンプ。形状: (B, 1, H, W)

        出力:
            torch.Tensor: 各タイムスタンプに対応するオプティカルフロー。
                          形状: (B, out_features, H, W)
        """
        B, C, H, W = features.shape

        # タイムスタンプと特徴マップを統合
        device = features.device

        query_timemap = query_timemap.to(device)
        inp = torch.cat([features, query_timemap], dim=1) 
        inp = inp.view(B, C + 1, H * W)
        input_for_siren = inp.permute(0, 2, 1)

        # inp.view(B*H, -1)は、H*Wの全ピクセルをフラット化
        flow_pred = self.flow_imnet(input_for_siren.reshape(B * H * W, -1)).reshape(B, H * W, -1)
        flow_pred = flow_pred.permute(0, 2, 1).reshape(B, 2, H, W)

        return flow_pred