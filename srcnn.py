import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os



class SRCNN(nn.Module):  # 搭建SRCNN 3层卷积模型，Conve2d（输入层数，输出层数，卷积核大小，步长，填充层）
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
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
EPOCHS = 800
LR = 0.001
WEIGHT_DECAY = 1e-5
SAVE_INTERVAL = 200

# 数据处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    #数据增强
    transforms.RandomHorizontalFlip(),
])

train_dataset = CustomDataset(
    degraded_folder=r"D:\xyl\LXY\NEW\Comparison\SRCNN\SRCNNDEG",#新结构
    target_folder=r"D:\xyl\LXY\NEW\Comparison\SRCNN\FTAR",#新结构
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
model.load_state_dict(torch.load("D:\\xyl\\LXY\\NEW\\Comparison\\SRCNN\\500.pth"))

'''
# 训练
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for degraded_imgs, target_imgs in train_loader:
        degraded_imgs, target_imgs = degraded_imgs.to(device), target_imgs.to(device)

        optimizer.zero_grad()

        outputs = model(degraded_imgs)

        # 1 - ssim是为了将它转换为损失，因为ssim的完美相似度得分是1
        loss1 = nn.MSELoss()

        loss = loss1(outputs, target_imgs)
        loss.requires_grad_(True)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * degraded_imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{EPOCHS} - SSIM Loss: {epoch_loss:.4f}")

    # 每SAVE_INTERVAL次保存一次权重
    if (epoch + 1) % SAVE_INTERVAL == 0:
        torch.save(model.state_dict(), f"D:\\xyl\\LXY\\NEW\\Comparison\\SRCNN\\{epoch + 1}.pth")
        print(f"Saved model weights at epoch {epoch + 1}")
  
print("Training Complete!")


'''
def infer_images(model_path, input_folder_path, output_folder_path):
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = SRCNN().to(device)
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
model_path = "D:\\xyl\\LXY\\NEW\\Comparison\\SRCNN\\500.pth"
input_folder_path = "D:\\xyl\\LXY\\NEW\\Comparison\\SRCNN\\SRCNNDEG"
output_folder_path = "D:\\xyl\\LXY\\NEW\\Comparison\\deconv\\deconvF"
infer_images(model_path, input_folder_path, output_folder_path)
#'''