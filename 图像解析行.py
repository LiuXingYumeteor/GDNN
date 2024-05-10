import cv2

# 读取图像，并将其转换为灰度图像
image_path = "E:\\Common files\\Work\\GMDNN\\PICTURE\\TIAOZHENG\\image2.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 检查图像是否正确读取
if image is None:
    print("Could not open or find the image.")
else:
    start_column = 73
    column_interval = 153
    num_columns = 8

    # 提取特定列
    for i in range(num_columns):
        #print(i)
        column = image[:, start_column + i * column_interval]

        # 打开一个txt文件用于写入，文件名基于列索引
        with open(f'E:\\Common files\\Work\\GMDNN\\PICTURE\\TIAOZHENG\\{i + 1}.txt', 'w') as file:
            for pixel_index, pixel_value in enumerate(column):
                file.write(f"Pixel {pixel_index}: {pixel_value}\n")

    print("Pixel values have been written to separate txt files.")