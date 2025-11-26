import torch
from torch import nn
import torch.nn.functional as F  # 建议加上这行，方便调用损失函数

class ConditionalVAE(nn.Module):
    """
    条件变分自编码器 (CVAE)
    输入: 图像 x 和其对应的标签 y
    """
    def __init__(self, input_dim=784, latent_dim=20, num_classes=10):
        super(ConditionalVAE, self).__init__()
        
        # 编码器部分
        # 将图像 (784) 和 one-hot 标签 (10) 拼接在一起作为输入
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # 学习潜在空间的均值 (mu) 和对数方差 (logvar)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # 解码器部分
        # 将潜在向量 z (20) 和 one-hot 标签 (10) 拼接在一起作为输入
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid() # 使用 Sigmoid 将输出像素值压缩到 0-1 之间
        )

    def reparameterize(self, mu, logvar):
        """重参数化技巧：z = mu + epsilon * std"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # 从标准正态分布中采样 epsilon
        return mu + eps * std

    def encode(self, x, y):
        """编码过程"""
        # 将图像和 one-hot 编码的标签在维度 1 上拼接
        inputs = torch.cat([x, y], dim=1)
        h = self.encoder(inputs)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, y):
        """解码过程"""
        # 将潜在向量和 one-hot 编码的标签在维度 1 上拼接
        inputs = torch.cat([z, y], dim=1)
        return self.decoder(inputs)

    def forward(self, x, y):
        """前向传播"""
        # ⚠️ 新增：把 4维图片 [64, 1, 28, 28] 拍扁成 2维 [64, 784]
        x = x.view(x.size(0), -1)
        
        # 1. 编码
        mu, logvar = self.encode(x, y)
        
        # 2. 重参数化
        z = self.reparameterize(mu, logvar)
        
        # 3. 解码
        recon_x = self.decode(z, y)
        
        return recon_x, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    """
    计算损失函数
    """
    # ⚠️ 新增：计算 Loss 前，也要把原始图片 x 拍扁，以匹配 recon_x 的形状
    x = x.view(x.size(0), -1)

    # 1. 重构损失
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # 2. KL 散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD