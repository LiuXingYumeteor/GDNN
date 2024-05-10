import cv2
import numpy as np

# 读取图像（这里假设它是灰度图像）
image_path = "E:\\Common files\\Work\\GMDNN\\dataset\\FIND\\F\\1.png"  # 请替换为你的图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像

# 检查图像是否成功读取
if image is None:
    print("Error: Could not read the image.")
else:
    # 创建一个图像副本，以便我们可以在不改变原始图像的情况下工作
    modified_image = image.copy()

    # 将0到262列（不包含263列）中灰度值不等于0的像素强制变为0
    modified_image[:, :253][modified_image[:, :253] != 0] = 0

    # 保存修改后的图像
    output_path = 'E:\\Common files\\Work\\GMDNN\\dataset\\FIND\\F\\1.png'  # 请替换为你想要保存的路径
    cv2.imwrite(output_path, modified_image)
    print(f"Modified image saved to {output_path}")

