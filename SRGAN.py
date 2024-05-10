import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from L1_MS_SSIM import MS_SSIM_L1_LOSS
from binary_loss import improved_binary_loss
from SSIM import SSIM
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast



class DenoiseModule(nn.Module):
    def __init__(self, in_channels):
        super(DenoiseModule, self).__init__()
        self.denoise = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.denoise(x)


class ResBlock(nn.Module):
    """优化后的残差模块"""
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        residual = x
        x = self.block(x)
        x += residual
        return x

class SRResNet(nn.Module):
    """优化后的SRResNet模型(4x)"""
    def __init__(self):
        super(SRResNet, self).__init__()
        self.denoise_module = DenoiseModule(1)
        self.conv_input = nn.Conv2d(64, 64, kernel_size=9, padding=4, padding_mode='reflect')
        self.relu = nn.PReLU()

        # 定义4个残差块
        self.res_blocks = nn.Sequential(*[ResBlock(64) for _ in range(3)])#note:定义多少残差块

        self.conv_mid = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.BatchNorm2d(64)

        # 子像素卷积层实现上采样
        self.upsample1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        self.conv_output = nn.Conv2d(64, 1, kernel_size=9, padding=4, padding_mode='reflect')

    def forward(self, x):
        x = self.denoise_module(x)
        x = self.relu(self.conv_input(x))
        residual = x

        x = self.res_blocks(x)

        x = self.bn_mid(self.conv_mid(x))
        x += residual  # 应用残差连接

        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.conv_output(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            # 输入尺寸: [input_channels x 64 x 64]
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
            # 输出尺寸: [1]
        )

    def forward(self, x):
        return self.network(x)


class CustomDataset(Dataset):
    def __init__(self, target_folder, degraded_folder, transform=None):
        self.target_folder = target_folder
        self.degraded_folder = degraded_folder
        self.transform = transform
        self.filenames = os.listdir(target_folder)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        target_img_path = os.path.join(self.target_folder, self.filenames[idx])
        degraded_img_path = os.path.join(self.degraded_folder, self.filenames[idx])

        # Convert the images to grayscale
        target_img = Image.open(target_img_path).convert('L')
        degraded_img = Image.open(degraded_img_path).convert('L')

        if self.transform:
            target_img = self.transform(target_img)
            degraded_img = self.transform(degraded_img)

        return degraded_img, target_img

BATCH_SIZE = 64
EPOCHS = 400
LR = 0.001
WEIGHT_DECAY = 1e-5
SAVE_INTERVAL = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化生成器
model = SRResNet().to(device)

# 加载预训练权重
model.load_state_dict(torch.load("D:\\xyl\\LXY\\NEW\\S2WEIGHT900M\\w200.pth"))

# 假设Discriminator类已经定义
discriminator = Discriminator().to(device)


# 数据处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    #数据增强
    transforms.RandomHorizontalFlip(),
])

train_dataset = CustomDataset(
    degraded_folder=r"D:\\xyl\\LXY\\NEW\\DATASET\\900data\\ANW",#新结构
    target_folder=r"D:\\xyl\\LXY\\NEW\\DATASET\\FTAR",#新结构
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)




# 初始化生成器和判别器的优化器
optimizer_G = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# 为判别器和生成器定义损失函数
criterion_GAN = nn.BCELoss()
criterion_content = MS_SSIM_L1_LOSS()  # 或者SSIM损失,SSIM不行

# 训练
for epoch in range(EPOCHS):
    model.train()
    discriminator.train()
    running_loss_G = 0.0
    running_loss_D = 0.0

    for degraded_imgs, target_imgs in train_loader:
        degraded_imgs, target_imgs = degraded_imgs.to(device), target_imgs.to(device)
        valid = torch.ones((degraded_imgs.size(0), 1), requires_grad=False).to(device)
        fake = torch.zeros((degraded_imgs.size(0), 1), requires_grad=False).to(device)

        # -----------------
        #  训练判别器
        # -----------------
        optimizer_D.zero_grad()

        #训练真实图像
        real_output = discriminator(target_imgs)
        real_output_squeeze = torch.squeeze(real_output,-1)
        real_output_squeeze = torch.squeeze(real_output_squeeze,-1)
        real_loss = criterion_GAN(real_output_squeeze, valid)

        # 训练生成的图像
        fake_imgs = model(degraded_imgs).detach()# 防止梯度传播到生成器
        fake_output = discriminator(fake_imgs)
        fake_output_squeezed = torch.squeeze(fake_output,-1) # 去除多余的维度
        fake_output_squeezed = torch.squeeze(fake_output_squeezed,-1)
        fake_loss = criterion_GAN(fake_output_squeezed, fake)

        # 判别器总损失
        loss_D = (real_loss + fake_loss) / 2
        loss_D.backward()
        optimizer_D.step()

        running_loss_D += loss_D.item()

        # -----------------
        #  训练生成器
        # -----------------
        optimizer_G.zero_grad()

        # 生成器的目标是骗过判别器
        print(discriminator(fake_output_squeezed).size())
        fake_imgs = model(degraded_imgs)
        g_loss = criterion_GAN(discriminator(fake_output_squeezed), valid)

        # 内容损失
        content_loss = criterion_content(fake_imgs, target_imgs)

        # 生成器总损失
        loss_G = g_loss + content_loss
        loss_G.backward()
        optimizer_G.step()

        running_loss_G += loss_G.item()

    epoch_loss_G = running_loss_G / len(train_loader.dataset)
    epoch_loss_D = running_loss_D / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{EPOCHS} - Generator Loss: {epoch_loss_G:.4f} - Discriminator Loss: {epoch_loss_D:.4f}")

    # 每SAVE_INTERVAL次保存一次权重
    if (epoch + 1) % SAVE_INTERVAL == 0:
        torch.save(model.state_dict(), f"D:\\xyl\\LXY\\NEW\\S2WEIGHT900G\\G_w{epoch + 1}.pth")
        torch.save(discriminator.state_dict(), f"D:\\xyl\\LXY\\NEW\\S2WEIGHT900G\\D_w{epoch + 1}.pth")
        print(f"Saved model weights at epoch {epoch + 1}")

