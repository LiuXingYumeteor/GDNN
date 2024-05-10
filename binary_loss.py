import torch
from L1_MS_SSIM import MS_SSIM_L1_LOSS
from SSIM import SSIM

def improved_binary_loss(output, target, beta=0.8):
    # 将目标图像二值化：这里假设目标图像已经是二值化的（0或255）
    binary_target = target / 255

    # 使用sigmoid函数来平滑模型输出，使其范围在0到1之间
    sigmoid_output = torch.sigmoid(beta * (output - 127.5))  # beta为斜率控制参数

    # 计算平滑后的输出与二值化目标之间的二次损失
    #loss = MS_SSIM_L1_LOSS(sigmoid_output, binary_target)
    loss1 = SSIM()
    loss = 1 - loss1(sigmoid_output, binary_target)

    return loss