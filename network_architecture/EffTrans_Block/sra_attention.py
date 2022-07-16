import torch
import torch.nn as nn
import torch.nn.functional as F


class SRA_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim*2)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
          #  self.sr = nn.Conv3d(dim, dim, kernel_size=3, stride=sr_ratio) 
            self.norm = nn.LayerNorm(dim)

    def forward(self, in_x):
        """
        in_x is (seq_len, batch, embed_dim), [6912, 2, 192] or [864, 2, 192]
        sra attention need (batch, seq_len, embed_dim)
        view is (bs, c, d, h, w)  # 12, 24, 24;   6, 12, 12
        @ is used for matrix multiplication
        """
        x = in_x.permute(1, 0, 2)  # (2, 6912, 192)
        B, N, C = x.shape
        
        if N==55296:
            D, H, W = 24, 48, 48 
        elif N==6912:
            D, H, W = 12, 24, 24
        elif N==864:
            D, H, W = 6, 12, 12
        else:
            print("the input of sra attention is wrong.")   
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # (2, 6912, 8, 24)--> (2, 8, 6912, 24)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)  # [2, 192, 6912]--> [2, 192, 12, 24, 24]
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # [2, 192, 6, 12, 12]--> [2, 192, 864]-->[2, 864, 192]
            x_ = self.norm(x_)                                   # [2, 864, 192]
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # [2, 864, 384]--> [2, 864, 2, 8, 24]-->[2, 2, 8, 864, 24]
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [2, 8, 864, 24]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [2, 8, 6912, 864]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)   # [2, 8, 6912, 24]-->[2, 6912, 8, 24]-->[2, 6912, 192]
        x = self.proj(x)          # [2, 6912, 384]
        x = self.proj_drop(x)     # [2, 6912, 384]
        out_x = x.permute(1, 0, 2) # [6912, 2, 384]

        return out_x
