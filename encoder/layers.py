import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

# --- ユーティリティ関数 ---
def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

def window_partition_2d(x, window_size):
    """ (B, H, W, C) -> (B*nW, Wh, Ww, C) """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows

def window_reverse_2d(windows, window_size, B, H, W):
    """ (B*nW, Wh, Ww, C) -> (B, H, W, C) """
    C = windows.shape[-1]
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# --- 2D Transformerモジュール (完全版) ---

class WindowAttention2D(nn.Module):
    """ 2D Window based multi-head self/cross attention """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, kv=None, mask=None):
        kv = q if kv is None else kv
        B_, N_q, C = q.shape
        _, N_kv, _ = kv.shape

        q = self.q(q).reshape(B_, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv_val = self.kv(kv).reshape(B_, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv_val[0], kv_val[1]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N_q, N_kv) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N_q, N_kv)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class EncoderTransformerBlock2D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(8, 8), shift_size=(0, 0), mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention2D(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix):
        B, H, W, C = x.shape
        
        shortcut = x
        x = self.norm1(x)

        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        if any(i > 0 for i in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        
        x_windows = window_partition_2d(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)

        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)

        shifted_x = window_reverse_2d(attn_windows, self.window_size, B, Hp, Wp)
        
        if any(i > 0 for i in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x[:, :H, :W, :].contiguous()
        
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderTransformerBlock2D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(8, 8), shift_size=(0, 0), mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = norm_layer(dim)
        self.attn1 = WindowAttention2D(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop) # Self-Attention
        
        self.norm2 = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.attn2 = WindowAttention2D(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop) # Cross-Attention
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm3 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, attn_kv, mask_matrix):
        B, H, W, C = x.shape
        
        # 1. Self-Attention
        shortcut = x
        x = self.norm1(x)
        
        # パディング処理
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        x_pad = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x_pad.shape

        # ウィンドウ分割と3次元への変形
        x_windows = window_partition_2d(x_pad, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)

        # アテンション計算
        attn_windows = self.attn1(x_windows, mask=None)

        # 4次元に戻す処理
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        x_sa = window_reverse_2d(attn_windows, self.window_size, B, Hp, Wp)
        
        # パディングを元に戻す
        x_sa = x_sa[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x_sa)
        
        # 2. Cross-Attention
        shortcut = x
        x = self.norm2(x)
        attn_kv = self.norm_kv(attn_kv)
        
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        attn_kv = F.pad(attn_kv, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape
        
        if any(i > 0 for i in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            shifted_attn_kv = torch.roll(attn_kv, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            shifted_attn_kv = attn_kv
            attn_mask = None

        q_windows = window_partition_2d(shifted_x, self.window_size).view(-1, self.window_size[0] * self.window_size[1], C)
        kv_windows = window_partition_2d(shifted_attn_kv, self.window_size).view(-1, self.window_size[0] * self.window_size[1], C)

        attn_windows = self.attn2(q_windows, kv_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse_2d(attn_windows, self.window_size, B, Hp, Wp)
        
        if any(i > 0 for i in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x[:, :H, :W, :].contiguous()
        x = shortcut + self.drop_path(x)
        
        # 3. FFN
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x

class EncoderLayer2D(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        
        self.blocks = nn.ModuleList([
            EncoderTransformerBlock2D(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=(0,0) if (i%2==0) else self.shift_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1) # B, H, W, C

        # ▼▼▼▼▼ マスク計算ロジック（完全版） ▼▼▼▼▼
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)
        Hp = int(np.ceil(H / window_size[0])) * window_size[0]
        Wp = int(np.ceil(W / window_size[1])) * window_size[1]
        
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None))
        w_slices = (slice(0, -window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition_2d(img_mask, window_size)
        mask_windows = mask_windows.view(-1, window_size[0] * window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # ▲▲▲▲▲ マスク計算ロジック（完全版） ▲▲▲▲▲

        for blk in self.blocks:
            x = blk(x, attn_mask)
            
        x = x.permute(0, 3, 1, 2) # B, C, H, W
        return x

class DecoderLayer2D(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        
        self.blocks = nn.ModuleList([
            DecoderTransformerBlock2D(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=(0,0) if (i%2==0) else self.shift_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)
        ])

    def forward(self, x, attn_kv):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1) # B, H, W, C
        attn_kv = attn_kv.permute(0, 2, 3, 1)

        # マスク計算はEncoderLayer2Dと同様
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)
        Hp = int(np.ceil(H / window_size[0])) * window_size[0]
        Wp = int(np.ceil(W / window_size[1])) * window_size[1]
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None))
        w_slices = (slice(0, -window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition_2d(img_mask, window_size)
        mask_windows = mask_windows.view(-1, window_size[0] * window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            x = blk(x, attn_kv, attn_mask)
            
        x = x.permute(0, 3, 1, 2) # B, C, H, W
        return x

# --- IO Modules (2D Modified) ---
class InputProj2D(nn.Module):
    def __init__(self, in_channels=3, embed_dim=32, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1),
            act_layer(inplace=True),
        )
    def forward(self, x):
        return self.proj(x)

class Downsample2D(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class Upsample2D(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
    def forward(self, x):
        return self.deconv(x)