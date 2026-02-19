from math import exp
import torch
import torch.nn.functional as F
from absl.logging import info
from torch import nn
# torch.autograd.Variable は PyTorch 0.4.0 以降では不要ですが、参照元に合わせて残します。
from torch.autograd import Variable 


# SSIM計算に必要な定数 (グローバル定数として定義)
C1 = 0.01**2
C2 = 0.03**2

def gaussian(window_size, sigma):
    """ガウシアンカーネルを生成"""
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """2Dガウシアンウィンドウ（カーネル）を作成し、nn.Moduleのバッファ用にVariableに変換"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # カーネルをチャネル数に合わせて拡張
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim_map(img1, img2, window, window_size, channel):
    """
    単一スケールでSSIMの輝度 (L) と構造 (S) の要素を計算します。
    _ssim関数からLuminanceとStructureを分離して返します。
    """
    # windowをimg1と同じデバイスに移動 (単一SSIM関数から踏襲)
    if img1.is_cuda:
        # VariableからTensorに変換し、cudaに移動
        window = window.cuda(img1.get_device())
    
    # 局所平均 (mu)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 局所分散 (sigma_sq) と共分散 (sigma12)
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # 輝度 (Luminance) の類似度項
    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    
    # コントラストと構造 (Contrast and Structure) の類似度項
    contrast_structure = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    
    return luminance, contrast_structure


class MS_SSIM(nn.Module):
    def __init__(
        self,
        value_range=1.0,
        window_size=11,
        size_average=True,
        # MS-SSIMの標準的な重み (M=5)
        weights=(0.0448, 0.2856, 0.6600, 0.003, 0.0066), 
        scales=5
    ):
        super(MS_SSIM, self).__init__()
        self.value_range = value_range
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3 # 初期チャネル数
        self.window = create_window(self.window_size, self.channel)
        
        # MS-SSIMのパラメータ
        self.scales = scales
        # 重みを正規化してTensorとして登録 (デバイス移動を容易にするため)
        self.weights = torch.tensor(weights[:self.scales]).float()
        self.weights = self.weights / self.weights.sum()
        self.register_buffer('weights_tensor', self.weights.view(self.scales, 1))
        
        info(f"Init MS-SSIM with {self.scales} scales:")
        info(f"  weights        : {self.weights.tolist()}")


    def forward(self, img1, img2):
        if img1.dim() == 5:
            # 時系列データなどのためにテンソルをフラット化
            img1 = torch.flatten(img1, start_dim=0, end_dim=1)
            img2 = torch.flatten(img2, start_dim=0, end_dim=1)

        # テンソル値を0から1の範囲に正規化
        img1 = img1 / self.value_range
        img2 = img2 / self.value_range
        
        (_, channel, _, _) = img1.size()
        
        # チャネル数が変更された場合、windowを再作成
        if self.channel != channel:
            self.channel = channel
            self.window = create_window(self.window_size, self.channel)
            
        # MS-SSIMの計算を開始
        msssim = Variable(torch.zeros(img1.size(0)).cuda(img1.get_device()) if img1.is_cuda else torch.zeros(img1.size(0)))
        
        # 各スケールでの構造類似度 (S) の積を計算 (対数空間で和を計算)
        # 論文ではSの積だが、重み付きSの累積和として実装する
        for i in range(self.scales):
            
            # スケールiでのLuminance (L) と Contrast/Structure (S) を計算
            L, S = _ssim_map(img1, img2, self.window, self.window_size, channel)
            
            # LとSの平均 (バッチ、チャネル以外の次元で平均)
            # 論文の重み定義に合わせるため、バッチ次元のみ残して平均
            L_mean = L.mean(dim=[1, 2, 3])
            S_mean = S.mean(dim=[1, 2, 3])

            # 構造項 S_i^(beta_i) を累積和として計算 (ここではbeta_i=1)
            # msssim には重み付きのS_meanを累積していく
            msssim = msssim + self.weights_tensor[i].to(msssim.device) * S_mean
            
            # 最後のスケールで輝度項 L_M^(alpha_M) を使用 (ここではalpha_M=1)
            if i == self.scales - 1:
                # 最後に輝度項を乗算
                msssim = msssim * L_mean

            # 次のスケールのために画像をダウンサンプリング (2x2平均プーリング)
            if i < self.scales - 1:
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)

        # 最終的なMS-SSIMスコアは msssim です (形状: バッチサイズ)
        
        if self.size_average:
            return msssim.mean()
        else:
            return msssim

class MS_SSIM_Loss(nn.Module):
    """
    MS-SSIM スコアを損失関数 (1 - MS-SSIM) としてラップするクラス
    """
    def __init__(self, **kwargs):
        super(MS_SSIM_Loss, self).__init__()
        # MS_SSIMクラスを内部に保持
        self.ms_ssim_module = MS_SSIM(**kwargs)

    def forward(self, img1, img2):
        # MS-SSIMスコア (類似度) を計算
        score = self.ms_ssim_module(img1, img2)
        
        # 損失関数 L = 1 - score を返す
        # scoreは最大値1.0なので、損失は最小値0.0
        return 1.0 - score