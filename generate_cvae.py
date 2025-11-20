import torch
import matplotlib.pyplot as plt
from torch.nn.functional import one_hot

# 导入我们定义的模型
from models.cvae import ConditionalVAE

# --- 配置 ---
DEVICE = "cpu"  # 生成任务通常不需要 GPU
LATENT_DIM = 20
NUM_CLASSES = 10
MODEL_PATH = "cvae_mnist.pth"

# --- 加载模型 ---
print(f"Loading model from {MODEL_PATH}...")
model = ConditionalVAE(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'.")
    print("Please run 'python scripts/train_cvae.py' first to train and save the model.")
    exit()

model.eval() # 设置为评估模式

def generate_images(digit, num_images=10):
    """
    为指定的数字生成图像。
    - digit: 要生成的数字 (0-9)
    - num_images: 要生成的图像数量
    """
    with torch.no_grad():
        # 1. 随机从标准正态分布中采样潜在向量 z
        z = torch.randn(num_images, LATENT_DIM).to(DEVICE)
        
        # 2. 创建条件的 one-hot 编码
        label_tensor = torch.tensor([digit] * num_images).to(DEVICE)
        label_onehot = one_hot(label_tensor, num_classes=NUM_CLASSES).float()
        
        # 3. 解码器从 (z, label) 生成图像
        generated_images = model.decode(z, label_onehot)
        
        # 将图像尺寸变回 28x28
        return generated_images.view(num_images, 28, 28)

if __name__ == '__main__':
    # --- 生成并可视化 ---
    digit_to_generate = 3  # 你可以改成任何 0-9 的数字
    num_samples = 10       # 生成 10 张图片

    print(f"Generating {num_samples} images for the digit '{digit_to_generate}'...")
    images = generate_images(digit_to_generate, num_images=num_samples)

    # 使用 Matplotlib 展示结果
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 2))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].cpu().numpy(), cmap='gray')
        ax.axis('off')
    
    plt.suptitle(f'Generated Images for Digit: {digit_to_generate}', fontsize=16)
    plt.show()