from PIL import Image


def crop_image(image_path, start_x, start_y, end_x, end_y):
    """
    裁剪图像的一部分。

    :param image_path: 原始图像的路径
    :param start_x: 裁剪区域左上角的x坐标
    :param start_y: 裁剪区域左上角的y坐标
    :param end_x: 裁剪区域右下角的x坐标
    :param end_y: 裁剪区域右下角的y坐标
    :return: 裁剪后的图像
    """
    # 打开原始图像
    image = Image.open(image_path)

    # 裁剪图像
    cropped_image = image.crop((start_x, start_y, end_x, end_y))

    # 显示裁剪后的图像（可选）
    cropped_image.show()

    # 返回裁剪后的图像
    return cropped_image


# 使用示例
# 假设你有一个名为"example.jpg"的图像，并且你想要裁剪(100, 100)到(300, 300)的区域。
cropped = crop_image("E:\\Common files\\Work\\GMDNN\\dataset\\PICTURE\\4.png", 675,628,4978,5515)

# 如果你想要保存裁剪后的图像，可以使用save方法：
cropped.save("cropped_image4.png")