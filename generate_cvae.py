import torch
import matplotlib.pyplot as plt
from models.cvae import CVAE
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("正在加载模型...")
model = CVAE().to(device)
model.load_state_dict(torch.load('cvae_mnist.pth', map_location=device))
model.eval()
print("模型加载完成！")

# 生成数字
def generate_digit(digit):
    with torch.no_grad():
        z = torch.randn(1, 20).to(device)
        c = torch.tensor([digit], dtype=torch.long).to(device)
        generated = model.decode(z, c)
        return generated.cpu().numpy()

# 生成0-9的数字
print("正在生成数字图片...")
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    row = i // 5
    col = i % 5
    digit_img = generate_digit(i).reshape(28, 28)
    axes[row, col].imshow(digit_img, cmap='gray')
    axes[row, col].set_title(f'Digit: {i}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('generated_digits.png')
print("生成完成！图片已保存为 generated_digits.png")
plt.show()