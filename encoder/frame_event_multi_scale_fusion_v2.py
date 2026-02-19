"""
Real-time Spatial Transformer (2D Modified Version for Frame/Event Fusion).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import numpy as np

# 修正版のlayersモジュールをインポート
from encoder.layers import DecoderLayer2D
from encoder.layers import Downsample2D
from encoder.layers import EncoderLayer2D
from encoder.layers import InputProj2D
from encoder.layers import Upsample2D


class FrameEventFusion(nn.Module):
    """2D Feature Fusion Module"""
    def __init__(self, embed_dim=96):
        super().__init__()
        self.mlp = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0)

    def forward(self, x, e):
        """
        Args:
            x: Frame features (B, C, H, W)
            e: Event features (B, C, H, W)
        return: Fused features (B, C, H, W)
        """
        xe = torch.cat([x, e], dim=1)
        return self.mlp(xe)


class EventHead(nn.Module):
    """2D Event Encoder Head"""
    def __init__(self, moments, embed_dim):
        super(EventHead, self).__init__()
        self.increase_dim = nn.Sequential(
            nn.Conv2d(moments, embed_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
        )

    def forward(self, e):
        return self.increase_dim(e)


class FrameEventsMultiScaleEncoderV2(nn.Module):
    def __init__(
        self,
        in_chans=3,
        moments=52,
        embed_dim=96,
        depths=[8, 8, 8, 8, 8, 8, 8, 8],
        num_heads=[2, 4, 8, 16, 16, 8, 4, 2],
        window_sizes=[(4, 4), (4, 4), (4, 4), (4, 4), (4, 4), (4, 4), (4, 4), (4, 4)],
        final_inr_dim=256,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    ):
        super().__init__()

        self.num_layers = len(depths)
        self.num_enc_layers = self.num_layers // 2
        self.num_dec_layers = self.num_layers // 2
        self.scale = 2 ** (self.num_enc_layers) # 修正: enc_layersの数だけスケール
        dec_depths = depths[self.num_enc_layers :]
        self.embed_dim = embed_dim

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.input_proj = InputProj2D(in_channels=in_chans, embed_dim=embed_dim, act_layer=nn.LeakyReLU)
        self.event_proj = EventHead(moments=moments, embed_dim=embed_dim)

        # Encoder (Frame & Event)
        self.encoder_layers = nn.ModuleList()
        self.e_encoder_layers = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        for i_layer in range(self.num_enc_layers):
            self.encoder_layers.append(
                EncoderLayer2D(
                    dim=embed_dim, depth=depths[i_layer], num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                    norm_layer=norm_layer
                )
            )
            self.downsample.append(Downsample2D(embed_dim, embed_dim))
            
            self.e_encoder_layers.append(
                EncoderLayer2D(
                    dim=embed_dim, depth=depths[i_layer], num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                    norm_layer=norm_layer
                )
            )

        # Fusion and Decoder
        self.frame_event_fusions = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        self.bottleneck_fusion = FrameEventFusion(embed_dim=embed_dim)

        for i_layer in range(self.num_dec_layers):
            self.frame_event_fusions.append(FrameEventFusion(embed_dim=embed_dim))
            self.decoder_layers.append(
                DecoderLayer2D(
                    dim=embed_dim, depth=depths[i_layer + self.num_enc_layers], num_heads=num_heads[i_layer + self.num_enc_layers],
                    window_size=window_sizes[i_layer + self.num_enc_layers], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:self.num_enc_layers+i_layer]) : sum(depths[: self.num_enc_layers+i_layer + 1])],
                    norm_layer=norm_layer
                )
            )
            self.upsample.append(Upsample2D(embed_dim, embed_dim))

        self.final_inr_adapter = nn.Conv2d(embed_dim, final_inr_dim, 1, 1, 0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, e):
        B, C, H, W = x.size()

        x = self.input_proj(x)
        e = self.event_proj(e)

        Hp = int(np.ceil(H / self.scale)) * self.scale
        Wp = int(np.ceil(W / self.scale)) * self.scale
        x = F.pad(x, (0, Wp - W, 0, Hp - H))
        e = F.pad(e, (0, Wp - W, 0, Hp - H))

        # Encoder
        encoder_features = []
        e_encoder_features = []
        for i_layer in range(self.num_enc_layers):
            x = self.encoder_layers[i_layer](x)
            e = self.e_encoder_layers[i_layer](e)
            
            encoder_features.append(x)
            e_encoder_features.append(e)

            x = self.downsample[i_layer](x)
            # イベント側はダウンサンプルを共有するか、個別にするか設計によるが、ここでは共有
            e = self.downsample[i_layer](e) 
        
        # Bottleneck
        y = self.bottleneck_fusion(x, e)

        # Decoder
        for i_layer in range(self.num_dec_layers):
            y = self.upsample[i_layer](y)
            
            x_skip = encoder_features[-i_layer - 1]
            e_skip = e_encoder_features[-i_layer - 1]
            
            skip_connection = self.frame_event_fusions[i_layer](x_skip, e_skip)
            
            y = self.decoder_layers[i_layer](y, skip_connection)

        # Final Output
        y = y[..., :H, :W].contiguous()
        outs = self.final_inr_adapter(y)
        
        return outs