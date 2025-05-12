import torch
import torch.nn as nn

class FrequencyDomainLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(FrequencyDomainLoss, self).__init__()
        self.alpha = alpha  # 相位损失的权重
        self.beta = beta    # 幅度损失的权重

    def forward(self, output, target):
        # 傅里叶变换
        output_freq = torch.fft.fft2(output)
        output_freq_shifted = torch.fft.fftshift(output_freq)
        output_angle = torch.angle(output_freq_shifted)  # 相位
        output_abs = torch.abs(output_freq_shifted)      # 幅度

        target_freq = torch.fft.fft2(target)
        target_freq_shifted = torch.fft.fftshift(target_freq)
        target_angle = torch.angle(target_freq_shifted)  # 相位
        target_abs = torch.abs(target_freq_shifted)      # 幅度

        # 计算相位损失
        # 使用周期性边界处理相位差，避免不连续性
        phase_diff = torch.atan2(torch.sin(output_angle - target_angle), torch.cos(output_angle - target_angle))
        loss_angle = torch.mean(torch.abs(phase_diff))

        # 计算幅度损失
        #output_abs_norm = output_abs / (output_abs.max() + 1e-8)
        #target_abs_norm = target_abs / (target_abs.max() + 1e-8)
        #loss_abs = torch.mean(torch.abs(output_abs_norm - target_abs_norm))
        # 计算幅度损失
        # 使用对数变换压缩动态范围
        epsilon = 1e-8
        output_abs_log = torch.log(1 + output_abs + epsilon)
        target_abs_log = torch.log(1 + target_abs + epsilon)

        loss_abs = torch.mean(torch.abs(output_abs_log - target_abs_log))

        # 总损失
        loss = self.alpha * loss_angle + self.beta * loss_abs
        return loss