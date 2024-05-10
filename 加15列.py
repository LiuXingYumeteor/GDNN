import cv2
import numpy as np

# 读取图像（确保是灰度图）
image_path = "E:\\Common files\\Work\\GMDNN\\DATA\\nogrant2\\nogrant2\\4.png"  # 请替换为你的图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像

# 检查图像是否成功读取
if image is None:
    print("Error: Could not read the image.")
else:
    # 获取图像的高度和宽度
    height, width = image.shape

    # 提取最后一列的灰度值
    last_column = image[:, width - 1].reshape((-1, 1))

    # 提取最后一行的灰度值
    #last_row = image[height - 1, :].reshape((1, -1))

    # 创建一个只包含最后一列灰度值的新数组，高度与原图像相同，宽度为50
    new_columns = np.repeat(last_column, 30, axis=1)

    # 创建一个只包含最后一行灰度值的新数组，高度为15，宽度与原图像相同
    #new_rows = np.repeat(last_row, 5, axis=0)

    # 将新列拼接到原图像的右侧
    extended_image = np.hstack((image, new_columns))

    # 将新行拼接到原图像（已扩展）的底部
    #extended_image = np.vstack((image, new_rows))

    # 保存扩展后的图像
    #output_path = "E:\\Common files\\Work\\GMDNN\\DATA\\900\\44.png"  # 请替换为你想要保存的路径
    output_path="E:\\Common files\\Work\\GMDNN\\DATA\\nogrant2\\nogrant2\\4.png"
    cv2.imwrite(output_path, extended_image)
    print(f"Extended image saved to {output_path}")
