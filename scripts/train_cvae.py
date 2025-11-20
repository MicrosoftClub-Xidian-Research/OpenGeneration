import torch
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from tqdm import tqdm
import os

# 导入我们定义的模型和损失函数
from models.cvae import ConditionalVAE, loss_function

# --- 超参数设置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
BATCH_SIZE = 128
LATENT_DIM = 20
INPUT_DIM = 28 * 28  # MNIST 图像大小
NUM_CLASSES = 10     # MNIST 类别数 (0-9)
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "cvae_mnist.pth"

print(f"Using device: {DEVICE}")

# --- 1. 数据加载 ---
# 定义数据预处理：将图像转换为 Tensor
transform = transforms.ToTensor()

# 下载并加载 MNIST 训练集
# root='../data' 表示将数据存放在项目根目录的上一级的 data 文件夹中，避免混乱
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
mnist_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
train_loader = DataLoader(mnist_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 2. 初始化模型和优化器 ---
model = ConditionalVAE(
    input_dim=INPUT_DIM,
    latent_dim=LATENT_DIM,
    num_classes=NUM_CLASSES
).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if os.path.exists(MODEL_SAVE_PATH):
    print('Train from the existing one!')
    state_dict = torch.load(MODEL_SAVE_PATH,map_location=DEVICE)
    model.load_state_dict(state_dict)
else:
    print("Train from scratch!")

# --- 3. 训练循环 ---
print("Start training CVAE...")
for epoch in range(1, EPOCHS + 1):
    model.train()  # 设置为训练模式
    train_loss = 0
    
    # 使用 tqdm 显示进度条
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    
    for batch_idx, (data, labels) in enumerate(pbar):
        data = data.to(DEVICE)
        # 将标签转换为 one-hot 编码
        labels_onehot = one_hot(labels, num_classes=NUM_CLASSES).float().to(DEVICE)
        
        optimizer.zero_grad()
        
        # 前向传播
        recon_batch, mu, logvar = model(data, labels_onehot)
        
        # 计算损失
        loss = loss_function(recon_batch, data, mu, logvar)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item() / len(data):.4f}")

    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')

# --- 4. 保存模型 ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")