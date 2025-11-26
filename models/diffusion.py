import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """把时间 t 转换成向量"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """基础卷积块 - 已修复通道计算逻辑"""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        
        if up:
            # 🔴 修复点1：这里不再写 2 * in_ch，而是直接信任传入的 in_ch
            # 因为我们在 ConditionalUNet 里手动计算了准确的通道数
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
            
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t, ):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class ConditionalUNet(nn.Module):
    """条件扩散模型的主干网络"""
    def __init__(self):
        super().__init__()
        image_channels = 1
        down_channels = (64, 128, 256)
        up_channels = (256, 128, 64)
        out_dim = 1 
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        self.label_emb = nn.Embedding(10, time_emb_dim)

        self.downs = nn.ModuleList([
            Block(image_channels, down_channels[0], time_emb_dim),
            Block(down_channels[0], down_channels[1], time_emb_dim),
            Block(down_channels[1], down_channels[2], time_emb_dim),
        ])

        # 🔴 修复点2：手动计算拼接后的通道数 (Concat Channels)
        # Up 1: 接收 D3(256) + Residual D3(256) = 512
        # Up 2: 接收 U0(256) + Residual D2(128) = 384  <-- 之前报错就是这里
        # Up 3: 接收 U1(128) + Residual D1(64)  = 192
        self.ups = nn.ModuleList([
            Block(down_channels[2] + down_channels[2], up_channels[0], time_emb_dim, up=True),
            Block(up_channels[0] + down_channels[1], up_channels[1], time_emb_dim, up=True),
            Block(up_channels[1] + down_channels[0], up_channels[2], time_emb_dim, up=True),
        ])
        
        self.output = nn.Conv2d(up_channels[2], out_dim, 1)

    def forward(self, x, t, y):
        t = self.time_mlp(t)
        l = self.label_emb(y)
        emb = t + l 

        residuals = []
        for down in self.downs:
            x = down(x, emb)
            residuals.append(x)
        
        for up in self.ups:
            residual = residuals.pop()
            
            # 🔴 修复点3：保留之前的尺寸对齐修复 (F.interpolate)
            if x.shape[2:] != residual.shape[2:]:
                x = F.interpolate(x, size=residual.shape[2:], mode='nearest')
            
            x = torch.cat((x, residual), dim=1)
            x = up(x, emb)
            
        return self.output(x)