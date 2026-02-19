import torch
from torch.nn.modules.loss import _Loss

from lpips import LPIPS
from loss.ms_ssim import MS_SSIM_Loss

class L1CharbonnierLossColor(_Loss):
    def __init__(self):
        super(L1CharbonnierLossColor, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = torch.add(x, -y)
        diff_sq = diff * diff
        error = torch.sqrt(diff_sq + self.eps)
        loss = torch.mean(error)
        return loss


class RollingShutterSharpReconstructedLoss(_Loss):
    def __init__(self):
        super(RollingShutterSharpReconstructedLoss, self).__init__()
        self.loss = L1CharbonnierLossColor()

    def forward(self, batch):
        rs_sharp_image = batch["rolling_sharp_frame_color"]
        rs_sharp_reconstructed_image = batch["rs_sharp_pred_frame"]
        return self.loss(rs_sharp_image, rs_sharp_reconstructed_image)


class GlobalShutterReconstructedLoss(_Loss):
    def __init__(self, loss_type):
        super(GlobalShutterReconstructedLoss, self).__init__()
        if loss_type == "charbonnier":
            self.loss = L1CharbonnierLossColor()
        elif loss_type == "mse":
            self.loss = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, batch):
        # B, N, C, H, W
        global_sharp_frames = batch["global_sharp_frames"]
        # B, N, C, H, W
        global_sharp_pred_frames = batch["global_sharp_pred_frames"]

        N = len(global_sharp_frames)
        
        loss = 0
        for i in range(N):
            gt = global_sharp_frames[i]
            pred = global_sharp_pred_frames[i]
            loss = loss + self.loss(gt, pred).float()
        loss = loss / N
        return loss

class GlobalShutterSharpMSSSIM(_Loss):
    def __init__(self):
        super(GlobalShutterSharpMSSSIM, self).__init__()
        self.loss = MS_SSIM_Loss()

    def forward(self, batch):
        # B, N, C, H, W
        global_sharp_frames = batch["global_sharp_frames"]
        # B, N, C, H, W
        global_sharp_pred_frames = batch["global_sharp_pred_frames"]

        N = len(global_sharp_frames)
        
        loss = 0
        for i in range(N):
            gt = global_sharp_frames[i]
            pred = global_sharp_pred_frames[i]
            loss = loss + self.loss(gt, pred).float()
        loss = loss / N
        return loss

