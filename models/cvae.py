import torch
from torch import nn
import torch.nn.functional as F

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
        """
        前向传播
        x:[B,1,28,28] or [B,784]
        y:[B,num_classes] one-hot label
        return:
            - recon_x:reconstructed image
            - mu:latent sapce's mean [B, latent_dim]
            - logvar: latent space's logvar [B, latent_dim]
        """
        batch_size = x.size(0)
        if x.dim() == 4:
            x_flat = x.view(batch_size,-1)
        else:
            x_flat = x
        
        mu, logvar = self.encode(x_flat,y)
        z = self.reparameterize(mu=mu,logvar=logvar)
        recon_batch = self.decode(z,y)
        recon_batch = recon_batch.view(batch_size,1,28,28)

        return recon_batch, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    """
    计算损失函数 = 重构损失 + KL 散度
    - recon_x: 重构后的图像
    - x: 原始图像
    - mu, logvar: 潜在空间的分布参数
    """
    batch_size = x.size(0)
    if recon_x.dim() == 4:
        recon_x_flat = recon_x.view(batch_size,-1)
    else:
        recon_x_flat = recon_x
    if x.dim() == 4:
        x_flat = x.view(batch_size,-1)
    else:
        x_flat = x

    bce = F.binary_cross_entropy(recon_x_flat,x_flat,reduction='sum')
    kld = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    loss = (bce + kld) / batch_size
    return loss