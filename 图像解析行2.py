import cv2
import os
import numpy as np

# 设置你的图像文件夹路径
folder_path = 'E:\\Common files\\GAN\\MTF'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # 检查文件扩展名
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            # 提取中间行的像素值
            middle_row = image[image.shape[0] // 2, :]

            # 构造保存的txt文件名
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(folder_path, txt_filename)

            # 保存中间行的像素值到txt文件
            np.savetxt(txt_path, middle_row, fmt='%d')

            print(f"Processed and saved middle row of {filename} as {txt_filename}.")
        else:
            print(f"Error loading image: {filename}")
    else:
        print(f"Skipped non-image file: {filename}")
