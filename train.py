import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from model import ImprovedUNetWithFourier  # 导入模型定义
from perceptual_loss import PerceptualLoss  # 导入感知损失函数
from torchmetrics.image import StructuralSimilarityIndexMeasure  ###结构相似性损失
from fft_loss import FrequencyDomainLoss  ##频域损失函数

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  ##更灵活的调度器，余弦退火


# 自定义数据集类
class PairedImageDataset(Dataset):
    def __init__(self, low_light_dir, enhanced_dir, transform=None):
        self.low_light_dir = low_light_dir
        self.enhanced_dir = enhanced_dir
        self.transform = transform
        self.low_light_folders = sorted(os.listdir(low_light_dir))
        self.enhanced_folders = sorted(os.listdir(enhanced_dir))

        # 检查文件夹结构是否一致
        if self.low_light_folders != self.enhanced_folders:
            raise ValueError("Low light and enhanced folders do not match!")

    def __len__(self):
        return len(self.low_light_folders)*60

    def __getitem__(self, idx):
        folder_idx = idx // 60
        image_idx = idx % 60
        low_light_folder = os.path.join(self.low_light_dir, self.low_light_folders[folder_idx])
        enhanced_folder = os.path.join(self.enhanced_dir, self.enhanced_folders[folder_idx])

        low_light_images = sorted(os.listdir(low_light_folder))
        enhanced_images = sorted(os.listdir(enhanced_folder))

        low_light_image = Image.open(os.path.join(low_light_folder, low_light_images[image_idx])).convert("RGB")
        enhanced_image = Image.open(os.path.join(enhanced_folder, enhanced_images[image_idx])).convert("RGB")

        if self.transform:
            low_light_image = self.transform(low_light_image)
            enhanced_image = self.transform(enhanced_image)

        return low_light_image, enhanced_image

# 定义复合损失函数
# 定义 L1 损失
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
l1_loss = nn.L1Loss().to(device)
# 定义感知损失
vgg_path = "./Perceptualmodels/vgg16_weights.pth"
perceptual_loss = PerceptualLoss(vgg_path=vgg_path).to(device)
# 定义频域损失
FreLoss = FrequencyDomainLoss().to(device)
# 定义 SSIM 损失
#ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0)
ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
# 定义 PSNR 损失
#psnr_loss = PSNRLoss(max_val=1.0).to(device)  # 假设图像像素值范围为 [0, 1]

def combined_loss(output, target, ssim_weight=0.1):
    l1 = l1_loss(output, target)
    percept = perceptual_loss(output, target)
    ffloss = FreLoss(output, target)
    ssim = 1 - ssim_loss(output, target)
    #psnr = 1-psnr_loss(output, target)  # 计算 PSNR 损失
    total_loss=l1 + ssim_weight * ssim + 0.5*ffloss + 0.1 * percept   # #+ psnr_weight * psnr 
    return total_loss

# 普通训练函数
def train(model, train_loader, val_loader, optimizer,scheduler, epochs=10):
    # 检查 GPU 是否可用

    # 将模型移动到 GPU 或 CPU
    model = model.to(device)

    # 检查保存文件夹是否存在，如果不存在则创建
    save_folder = "./pth_V1"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        train_losses = []

        for batch_idx, (low_light_images, enhanced_images) in enumerate(train_loader):
            # 将数据移动到 GPU 或 CPU
            low_light_images = low_light_images.to(device)
            enhanced_images = enhanced_images.to(device)

            # 前向传播
            output = model(low_light_images)
            loss = combined_loss(output, enhanced_images)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item()}")
        scheduler.step()  # 更新学习率

        # 验证阶段
        model.eval()  # 设置模型为评估模式
        val_losses = []
        with torch.no_grad():
            for low_light_images, enhanced_images in val_loader:
                low_light_images = low_light_images.to(device)
                enhanced_images = enhanced_images.to(device)

                output = model(low_light_images)
                val_loss = combined_loss(output, enhanced_images)
                val_losses.append(val_loss.item())

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {np.mean(train_losses)}, Validation Loss: {np.mean(val_losses)}")

        # 保存模型权重
        if (epoch + 1) % 1 == 0:
            save_path = os.path.join(save_folder, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model weights saved to '{save_path}'")
            

# 主函数
if __name__ == "__main__":
    # 数据集路径
    low_light_dir = "./LOLBlur Dataset/train/low_blur"
    enhanced_dir = "./LOLBlur Dataset/train/high_sharp_scaled"

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 将像素值从 [0, 255] 转换为 [0, 1]
    ])
    # 加载数据集
    train_dataset = PairedImageDataset(low_light_dir, enhanced_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=False, num_workers=8)

    val_dataset = PairedImageDataset(low_light_dir, enhanced_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=False, num_workers=8)

    # 初始化模型和优化器
    model = ImprovedUNetWithFourier()
    # 加载保存的参数
    #save_path = './pth_V8(接V6)/model_epoch_337.pth'  # 替换为你的保存路径
    #model.load_state_dict(torch.load(save_path))
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
    # 设置余弦退火调度器
    T_0 = 10  # 第一个周期的迭代次数
    T_mult = 2  # 周期增长倍数
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-6)

    # 普通训练
    train(model, train_loader, val_loader, optimizer, scheduler,epochs=100)