import os
import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, vgg_path="./Perceptualmodels/vgg16_weights.pth"):
        super(PerceptualLoss, self).__init__()
        self.vgg_path = vgg_path

        # 确保保存路径的目录存在
        os.makedirs(os.path.dirname(self.vgg_path), exist_ok=True)

        if os.path.exists(self.vgg_path):
            print(f"Loading VGG16 weights from {self.vgg_path}")
            vgg16 = models.vgg16(pretrained=False)
            vgg16.load_state_dict(torch.load(self.vgg_path))
        else:
            print("Downloading pre-trained VGG16 weights...")
            vgg16 = models.vgg16(pretrained=True)
            torch.save(vgg16.state_dict(), self.vgg_path)
            print(f"VGG16 weights saved to {self.vgg_path}")

        # 提取 VGG16 的前 22 层
        self.vgg = nn.Sequential(*list(vgg16.features.children())[:22])

        # 冻结权重
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        # 将输入和目标图像通过 VGG16 网络
        input_vgg = self.vgg(input)
        target_vgg = self.vgg(target)
        # 计算 L1 损失
        loss = nn.L1Loss()(input_vgg, target_vgg)
        return loss