import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from PSNR import psnr_loss
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
        self.res_blocks = nn.Sequential(*[ResBlock(64) for _ in range(2)])#note:定义多少残差块

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

# 参数
BATCH_SIZE = 32
EPOCHS = 1000
LR = 0.001
WEIGHT_DECAY = 1e-5
SAVE_INTERVAL = 500

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

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRResNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# 加载预训练权重
model.load_state_dict(torch.load("D:\\xyl\\LXY\\NEW\\S2WEIGHT900\\z1000.pth"))


#'''
#训练   
for epoch in range(EPOCHS):
    # 在每个epoch开始时更新注意力权重
    model.train()
    running_loss = 0.0
    for degraded_imgs, target_imgs in train_loader:
        degraded_imgs, target_imgs = degraded_imgs.to(device), target_imgs.to(device)

        optimizer.zero_grad()

        outputs = model(degraded_imgs)

        # 1 - ssim是为了将它转换为损失，因为ssim的完美相似度得分是1
        #loss1 = SSIM()
        loss1 = MS_SSIM_L1_LOSS() #第一阶段训练效果最好,权重为z1000
        #loss1 = nn.MSELoss()
        #loss = psnr_loss(outputs,target_imgs)
        loss = loss1(outputs,target_imgs)


        loss.backward()
        optimizer.step()

        running_loss += loss.item() * degraded_imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{EPOCHS} - Fusion Loss: {epoch_loss:.4f}")

    # 每SAVE_INTERVAL次保存一次权重
    if (epoch + 1) % SAVE_INTERVAL == 0:
        torch.save(model.state_dict(), f"D:\\xyl\\LXY\\NEW\\S2WEIGHT900M\\n{epoch + 1}.pth")
        print(f"Saved model weights at epoch {epoch + 1}")

print("Training Complete!")

'''
def infer_images(model_path, input_folder_path, output_folder_path):
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = SRResNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 确保输出文件夹存在
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # 遍历输入文件夹中的所有图像
    for filename in os.listdir(input_folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            input_image_path = os.path.join(input_folder_path, filename)
            output_image_path = os.path.join(output_folder_path, filename)

            # 加载和预处理图像
            image = Image.open(input_image_path).convert('L')  # 转换为灰度图像
            image = transform(image).unsqueeze(0).to(device)  # 添加batch维度

            # 推断
            with torch.no_grad():
                reconstructed_image = model(image)

            # 将输出转换为图像并保存
            save_image = transforms.ToPILImage()(reconstructed_image.squeeze().cpu())  # 移除batch维度
            save_image.save(output_image_path)
            print(f"Processed and saved: {output_image_path}")

# 使用方法
model_path = "D:\\xyl\\LXY\\NEW\\S2WEIGHT900M\\z1000.pth"
input_folder_path = "D:\\xyl\\LXY\\NEW\\DATASET\\900data\\ANW"
output_folder_path = "D:\\xyl\\LXY\\NEW\\S2\\ANW2"
infer_images(model_path, input_folder_path, output_folder_path)#
'''