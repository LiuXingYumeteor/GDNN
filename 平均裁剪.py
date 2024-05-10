from PIL import Image

def crop_image_into_grid(image_path, grid_size=(8, 8)):
    """
    将图像裁剪成指定网格大小的多个图像块。
    :param image_path: 原始图像的路径
    :param grid_size: 裁剪网格的大小，格式为(行, 列)
    :return: 裁剪后的图像块列表
    """
    # 打开原始图像
    image = Image.open(image_path)

    # 获取图像的宽度和高度
    width, height = image.size

    # 计算每个图像块的尺寸
    block_width = width // grid_size[1]
    block_height = height // grid_size[0]

    # 存储裁剪后的图像块
    cropped_images = []

    # 遍历网格并裁剪图像块
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            # 计算裁剪区域的坐标
            x0 = col * block_width
            y0 = row * block_height
            x1 = x0 + block_width
            y1 = y0 + block_height

            # 裁剪图像块
            cropped_image = image.crop((x0, y0, x1, y1))

            # 将裁剪后的图像块添加到列表中
            cropped_images.append(cropped_image)

    return cropped_images


# 使用示例
# 假设你有一个名为"example.jpg"的图像，并且你想要将其裁剪成8x8的网格。
cropped_images = crop_image_into_grid("E:\\Common files\\Work\\GMDNN\\dataset\\example_with_borders3.png", grid_size=(8, 8))
# 如果你想要保存裁剪后的图像块，可以使用save方法：
for i, img in enumerate(cropped_images):
     img.save(f"E:\\Common files\\Work\\GMDNN\\dataset\\Dataset\\TARGET\\block_{i+105}.png")