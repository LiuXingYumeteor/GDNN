import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from SSIM import SSIM
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast



#空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        # 添加可训练的权重参数
        self.attention_weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv1(attention)
        # 使用权重调整注意力的强度
        attention = self.sigmoid(attention) * torch.sigmoid(self.attention_weight)
        return x * attention


#通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # 注意力权重
        self.attention_weight = nn.Parameter(torch.zeros(1),requires_grad=True)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        scale = self.sigmoid(out)*torch.sigmoid(self.attention_weight)
        return x*scale

#残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_attention=True):
        super(ResidualBlock, self).__init__()

        self.use_attention = use_attention
        if self.use_attention:
            self.channel_attention = ChannelAttention(out_channels)
            self.spatial_attention = SpatialAttention()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.1)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_attention:
            out = self.channel_attention(out)
            out = self.spatial_attention(out)

        out += residual
        out = self.relu(out)

        return out


class UpSampleBlock(nn.Module):
    """上采样块，用于解码器"""
    def __init__(self, in_channels, out_channels,use_attention=True):
        super(UpSampleBlock, self).__init__()
        self.use_attention = use_attention
        if self.use_attention:
            self.channel_attention = ChannelAttention(out_channels)
            self.spatial_attention = SpatialAttention()
        self.up_sample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

    def forward(self, x):
        out = self.up_sample(x)
        if self.use_attention:
            out = self.channel_attention(out)
            out = self.spatial_attention(out)
        return out

class SimpleResNet(nn.Module):
    """带有注意力机制的对称编解码器结构"""
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(1, 16),
            nn.MaxPool2d(2, 2),
            ResidualBlock(16, 32),
            nn.MaxPool2d(2, 2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2, 2),
        )

        self.decoder = nn.Sequential(
            UpSampleBlock(64, 32),
            UpSampleBlock(32, 16),
            UpSampleBlock(16, 8),
        )

        self.final_conv = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        x = self.activation(x)
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
BATCH_SIZE = 64
EPOCHS = 20000
LR = 0.001
WEIGHT_DECAY = 1e-5
SAVE_INTERVAL = 1000

# 数据处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    #数据增强
    transforms.RandomHorizontalFlip(),
])

train_dataset = CustomDataset(
    degraded_folder=r"D:\\xyl\\LXY\\NEW\\DATASET\\0data\\DEG",#新结构
    target_folder=r"D:\\xyl\\LXY\\NEW\\DATASET\\0data\\TAR",#新结构
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleResNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def update_attention_weights(epoch, model, init_weight=0.0, final_weight=1.0, total_epochs=2000):
    """
    根据当前epoch渐进式增加注意力权重。

    参数:
    - epoch: 当前的epoch。
    - model: PyTorch模型。
    - init_weight: 注意力权重的初始值。
    - final_weight: 注意力权重的最终值。
    - total_epochs: 总的训练epochs。
    """
    # 计算当前权重，线性从init_weight增加到final_weight
    current_weight = (final_weight - init_weight) * (epoch / total_epochs) + init_weight
    current_weight = min(current_weight, final_weight)  # 确保不超过final_weight

    # 遍历模型中的所有模块，更新注意力权重
    for module in model.modules():
        if isinstance(module, (ChannelAttention, SpatialAttention)):
            module.attention_weight.data.fill_(current_weight)


'''
#训练
for epoch in range(EPOCHS):
    # 在每个epoch开始时更新注意力权重
    update_attention_weights(epoch, model,init_weight=0.0, final_weight=1.0, total_epochs=EPOCHS)
    model.train()
    running_loss = 0.0
    for degraded_imgs, target_imgs in train_loader:
        degraded_imgs, target_imgs = degraded_imgs.to(device), target_imgs.to(device)

        optimizer.zero_grad()

        outputs = model(degraded_imgs)

        # 1 - ssim是为了将它转换为损失，因为ssim的完美相似度得分是1
        lloss = SSIM()
        loss = 1 - lloss(outputs, target_imgs)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * degraded_imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{EPOCHS} - SSIM Loss: {epoch_loss:.4f}")

    # 每SAVE_INTERVAL次保存一次权重
    if (epoch + 1) % SAVE_INTERVAL == 0:
        torch.save(model.state_dict(), f"D:\\xyl\\LXY\\NEW\\WEIGHT0\\w{epoch + 1}.pth")
        print(f"Saved model weights at epoch {epoch + 1}")

print("Training Complete!")
'''

def infer_images(model_path, input_folder_path, output_folder_path):
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = SimpleResNet().to(device)
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
model_path = "D:\\xyl\\LXY\\NEW\\WEIGHT0\\w20000.pth"
input_folder_path = "D:\\xyl\\LXY\\NEW\\DATASET\\0data\\DEG"
output_folder_path = "D:\\xyl\\LXY\\NEW\\DATASET\\0data\\ANW"
infer_images(model_path, input_folder_path, output_folder_path)
#'''