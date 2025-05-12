import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch.fft as fft

import torch
import torch.nn as nn

class FourierFusion_abs(nn.Module):
    def __init__(self, in_channels):
        super(FourierFusion_abs, self).__init__()
        # 空间域卷积支路
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # 使用 3x3 卷积核
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # 使用 3x3 卷积核
        
        # 频域支路
        self.conv_freq1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)  # 输入通道数为 2 * in_channels
        self.conv_freq2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # 支路 1：直接使用输入 x
        spatial_feature = x

        # 支路 2：空间域卷积支路
        conv_feature = self.conv1(x)
        conv_feature = F.relu(conv_feature)  # 激活函数
        conv_feature = self.conv2(conv_feature)

        # 支路 3：频域支路
        # 使用 rfft2 进行傅里叶变换
        freq_feature = torch.fft.rfft2(x)
        freq_feature_shifted = torch.fft.fftshift(freq_feature)  # 将零频分量移到中心
        ####把取幅值信息的操作去掉了改成了下面的操作,由幅值和相位的组合变成了实部和虚部的结合。
        
      
        # 分离实部和虚部
        y_real = freq_feature_shifted.real
        y_imag = freq_feature_shifted.imag
    
        #分离相位和幅值
        #magnitude = torch.abs(freq_feature_shifted )
        #phase = torch.angle(freq_feature_shifted )


        # 拼接实部和虚部
        y_f = torch.cat([y_real, y_imag], dim=1)  # 通道维度拼接
        #拼接相位和幅值
        #y_f = torch.cat([magnitude,  phase], dim=1)  # 通道维度拼接

        # 频域卷积
        freq_feature = self.conv_freq1(y_f)
        freq_feature = F.relu(freq_feature)  # 激活函数
        freq_feature = self.conv_freq2(freq_feature)

        # 将频域特征反变换回空间域
        freq_feature = torch.fft.ifftshift(freq_feature)  # 反移位
        freq_feature = torch.fft.irfft2(freq_feature, s=x.shape[2:])  # 逆傅里叶变换
        freq_feature = torch.real(freq_feature)  # 取实部

        # 三条支路的输出进行元素级相加
        fused_feature = spatial_feature + conv_feature + freq_feature
        return fused_feature
    
class FourierFusion_angle(nn.Module):
    def __init__(self, in_channels):
        super(FourierFusion_angle, self).__init__()
        # 空间域卷积支路
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # 使用 3x3 卷积核
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # 使用 3x3 卷积核
        
        # 频域支路
        self.conv_freq1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)  # 输入通道数为 2 * in_channels
        self.conv_freq2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # 支路 1：直接使用输入 x
        spatial_feature = x

        # 支路 2：空间域卷积支路
        conv_feature = self.conv1(x)
        conv_feature = F.relu(conv_feature)  # 激活函数
        conv_feature = self.conv2(conv_feature)

        # 支路 3：频域支路
        # 使用 rfft2 进行傅里叶变换
        freq_feature = torch.fft.rfft2(x)
        freq_feature_shifted = torch.fft.fftshift(freq_feature)  # 将零频分量移到中心
        #freq_feature = torch.angle(freq_feature)  # 取相位信息
        #freq_feature = (freq_feature + np.pi) / (2 * np.pi)  # 将相位归一化到 [0, 1] 范围

        
        # 分离实部和虚部
        y_real = freq_feature_shifted.real
        y_imag = freq_feature_shifted.imag
        
        #分离相位和幅值
        #magnitude = torch.abs(freq_feature_shifted )
        #phase = torch.angle(freq_feature_shifted )


        # 拼接实部和虚部
        y_f = torch.cat([y_real, y_imag], dim=1)  # 通道维度拼接
        #拼接相位和幅值
        #y_f = torch.cat([magnitude,  phase], dim=1)  # 通道维度拼接

        # 频域卷积
        freq_feature = self.conv_freq1(y_f)
        freq_feature = F.relu(freq_feature)  # 激活函数
        freq_feature = self.conv_freq2(freq_feature)

        # 将频域特征反变换回空间域
        freq_feature = torch.fft.ifftshift(freq_feature)  # 反移位
        freq_feature = torch.fft.irfft2(freq_feature, s=x.shape[2:])  # 逆傅里叶变换
        freq_feature = torch.real(freq_feature)  # 取实部

        # 三条支路的输出进行元素级相加
        fused_feature = spatial_feature + conv_feature + freq_feature
        return fused_feature
 
class EncoderWithFourier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderWithFourier, self).__init__()
        self.downconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.fourier_fusion = FourierFusion_abs(out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        # 下采样
        x = self.downconv(x) 
        # 频域特征融合
        x = self.fourier_fusion(x)
        # 卷积层
        x = self.conv(x)
        return x
    
class DecoderWithFourier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderWithFourier, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fourier_fusion = FourierFusion_angle(out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connection):

        # 跳跃连接
        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)
        # 上采样
        x = self.upconv(x) 

        # 频域特征融合
        x = self.fourier_fusion(x)

        # 卷积层
        x = self.conv(x)
        return x
    
class ImprovedUNetWithFourier(nn.Module):
    def __init__(self):
        super(ImprovedUNetWithFourier, self).__init__()
        # 编码器
        self.encoder1 = EncoderWithFourier(3, 64)
        self.encoder2 = EncoderWithFourier(64, 128)
        self.encoder3 = EncoderWithFourier(128, 256)
        self.encoder4 = EncoderWithFourier(256, 512)

        # 光照增强分支
        #self.illumination_head = nn.Sequential(
        #    nn.Conv2d(512, 3, kernel_size=1),
        #    nn.Sigmoid()  # 输出反射分量
       # )

        # 解码器（嵌入傅里叶先验知识）
        self.decoder4 = DecoderWithFourier(512, 256)
        self.decoder3 = DecoderWithFourier(512, 128)
        self.decoder2 = DecoderWithFourier(256, 64)
        self.decoder1 = DecoderWithFourier(128, 64)

        # 最终输出层（移除 Sigmoid 激活函数）
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
        #self.final_conv = nn.Sequential(nn.Conv2d(64, 3, kernel_size=1),nn.Sigmoid()) # 或 nn.Tanh()  

    def forward(self, x):
        # 编码器
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        # 光照增强分支
        #illumination = self.illumination_head(e4)

        # 解码器
        d4 = self.decoder4(e4, skip_connection=None)
        d3 = self.decoder3(d4, skip_connection=e3)
        d2 = self.decoder2(d3, skip_connection=e2)
        d1 = self.decoder1(d2, skip_connection=e1)

        # 最终输出
        deblurred = self.final_conv(d1)+x

        ##加上输入的跳跃连接，白的更白，黑的更黑
        
        return  deblurred
    
       
    
if __name__ == "__main__":
    model=ImprovedUNetWithFourier()