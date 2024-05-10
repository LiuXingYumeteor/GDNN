import cv2
import numpy as np
#'''
# 读取图像
img = cv2.imread('E:\\Common files\\Work\\GMDNN\\PICTURE\\P\\cropped_image2.png')  # 请将'your_image.jpg'替换为你的图像文件名

# 获取图像的高度和宽度
height, width, channels = img.shape

# 创建一个全为255的20列宽的图像，高度与原图相同
white_strip = np.full((height, 1, channels), 255, dtype=np.uint8)

# 将这个白条添加到原图的右侧
new_img = np.hstack((img, white_strip))

# 保存新图像
cv2.imwrite('E:\\Common files\\Work\\GMDNN\\PICTURE\\P\\cropped_image2.png', new_img)  # 新图像将被保存为'modified_image.jpg'
'''

import cv2
import numpy as np

# 读取图像
image_path = "E:\\Common files\\Work\\GMDNN\\PICTURE\\P\\cropped_image4.png"  # 请替换为你的图像文件路径
img = cv2.imread(image_path)

# 检查图像是否成功读取
if img is None:
    print("Error: Could not read the image.")
    exit()

# 获取图像的高度、宽度和通道数
height, width, channels = img.shape

# 创建一个两行高，与原图像等宽，所有像素值为255的新图像
white_strip = np.full((1, width, channels), 255, dtype=np.uint8)

# 将这个白条添加到原图的下方
new_img = np.vstack((img, white_strip))

# 保存新图像
new_image_path = "E:\\Common files\\Work\\GMDNN\\PICTURE\\P\\cropped_image4.png"  # 新图像的文件路径
cv2.imwrite(new_image_path, new_img)

print(f"New image with additional white strip has been saved to {new_image_path}")
'''