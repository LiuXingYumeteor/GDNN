import os
import cv2

# 读取灰度图像
image_path = 'E:\\Common files\\Work\\GMDNN\\PICTURE\\TIAOZHENG\\image4.png' # 图像文件路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 定义文本文件路径
text_file_path = os.path.splitext(image_path)[0] + '.txt'

# 保存灰度矩阵为文本文件
with open(text_file_path, 'w') as file:
    for row in image:
        for pixel in row:
            file.write(str(pixel) + ' ')
        file.write('\n')

print(f"灰度矩阵已保存为文本文件：{text_file_path}")

