import cv2
import numpy as np


# 定义双三次插值核函数（Bicubic Kernel）
def bicubic_kernel(x):
    return (1 - abs(x)) ** 3 - (1 - abs(x)) ** 2 * abs(x) if abs(x) < 1 else (
            (abs(x) - 2) ** 3 + 1) if 1 <= abs(x) < 2 else 0


# 双三次插值函数
def bicubic_interpolate(image, x, y):
    # 获取原始图像尺寸
    height, width, channels = image.shape

    # 计算对应的浮点坐标
    x = min(max(0, x), width - 1)
    y = min(max(0, y), height - 1)

    # 向下取整得到整数坐标
    x_int = int(x)
    y_int = int(y)

    # 获取浮点坐标的小数部分
    x_frac = x - x_int
    y_frac = y - y_int

    # 初始化插值结果
    pixel_value = np.zeros((channels,))

    # 对周围的16个像素点进行双三次插值
    for i in range(-1, 3):
        for j in range(-1, 3):
            # 计算核函数值
            kernel_value = bicubic_kernel(i - x_frac) * bicubic_kernel(j - y_frac)
            # 获取像素值，注意边界处理
            if (y_int + i) >= 0 and (y_int + i) < height and (x_int + j) >= 0 and (x_int + j) < width:
                pixel_value += kernel_value * image[y_int + i, x_int + j]

    return pixel_value


# 两倍超分辨率函数
def bicubic_upscale(image):
    # 获取原始图像尺寸
    height, width, channels = image.shape

    # 创建新图像，大小为原始图像的2倍
    new_height = height * 2
    new_width = width * 2
    upscaled_image = np.zeros((new_height, new_width, channels), dtype=image.dtype)

    # 对新图像的每个像素进行双三次插值
    for i in range(new_height):
        for j in range(new_width):
            # 计算原始图像中对应的浮点坐标
            x = j / 2.0
            y = i / 2.0

            # 应用双三次插值函数
            pixel_value = bicubic_interpolate(image, x, y)

            # 将插值结果赋值给新图像
            upscaled_image[i, j] = pixel_value

    return upscaled_image


# 读取图像
image = cv2.imread('E:\\Common files\\Work\\GMDNN\\PICTURE\\pt\\3001.png')

# 将图像数据类型转换为浮点数，以便进行数学计算
image = image.astype(np.float32)

# 应用双三次插值进行两倍超分辨率
upscaled_image = bicubic_upscale(image)

# 将数据类型转换回uint8
upscaled_image = np.clip(upscaled_image, 0, 255).astype(np.uint8)

# 保存结果
cv2.imwrite('E:\\Common files\\Work\\GMDNN\\PICTURE\\pt\\c3001.png', upscaled_image)