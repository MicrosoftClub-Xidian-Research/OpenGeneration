import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import os

# 路径黑魔法
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.diffusion import ConditionalUNet

# ================= 配置 =================
# 强制使用 0 号显卡 (对应你任务管理器的 GPU 0 - RTX 4060)
DEVICE = "cuda:0"
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 10 # 扩散模型训练慢，先跑10轮试试
TIMESTEPS = 300 # 扩散步数，标准是1000，为了速度我们改小点

# ================= 扩散过程参数 =================
# 定义 beta schedule (噪声增加的速率)
betas = torch.linspace(0.0001, 0.02, TIMESTEPS).to(DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    前向扩散：给 x_0 加噪声，生成 x_t
    返回: x_t (加噪后的图), noise (加进去的噪声)
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t][:, None, None, None]
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    
    # 公式: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

# ================= 数据与模型 =================
# 扩散模型输入通常是 [-1, 1]，所以 normalize 是 (0.5, 0.5)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ConditionalUNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss() # 预测噪声用均方误差

# ================= 训练循环 =================
print("Start training Diffusion Model...")
model.train()

for epoch in range(EPOCHS):
    for step, (images, labels) in enumerate(dataloader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # 1. 随机采样时间步 t
        t = torch.randint(0, TIMESTEPS, (images.shape[0],), device=DEVICE).long()
        
        # 2. 加噪 (Forward Process)
        x_t, noise = forward_diffusion_sample(images, t, DEVICE)
        
        # 3. 预测噪声 (Reverse Process step prediction)
        noise_pred = model(x_t, t, labels)
        
        # 4. 计算 Loss：只要预测的噪声和真噪声越像越好
        loss = loss_fn(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch} finished.")

# 保存模型
torch.save(model.state_dict(), "diffusion_mnist.pth")
print("Model saved to diffusion_mnist.pth")
