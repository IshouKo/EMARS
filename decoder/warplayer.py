import torch
import torch.nn as nn
from pdb import set_trace as bp

# グローバルなキャッシュを削除し、クラス内で管理する
# backwarp_tenGrid = {}

class Warper(nn.Module):
    def __init__(self):
        super(Warper, self).__init__()
        self.backwarp_tenGrid = {} # 各インスタンスが独自のキャッシュを持つ

    def warpgrid(self, tenFlow):
        # tenFlowと同じデバイス上で動作させる
        k = (str(tenFlow.device), str(tenFlow.size()))

        # tenFlowと同じデバイスにテンソルがキャッシュされているか確認
        if k not in self.backwarp_tenGrid:
            # tenFlowと同じデバイスを使用
            device = tenFlow.device
            
            tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
                1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
            tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
                1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
            
            # テンソルを結合してキャッシュに保存
            self.backwarp_tenGrid[k] = torch.cat(
                [tenHorizontal, tenVertical], 1).to(device)

        tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenFlow.shape[3] - 1.0) / 2.0),
                             tenFlow[:, 1:2, :, :] / ((tenFlow.shape[2] - 1.0) / 2.0)], 1)
        
        # tenFlowと同じデバイス上のキャッシュされたテンソルを使用
        g = (self.backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)

        return g