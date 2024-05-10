from PIL import Image


def crop_image_centered(image_path, center_x, center_y, crop_size):
    # 加载图像
    image = Image.open(image_path)

    # 获取图像尺寸
    width, height = image.size

    # 将小数坐标转换为整数坐标
    center_x = int(round(center_x))
    center_y = int(round(center_y))

    # 计算裁剪框的左上角和右下角坐标
    left = center_x - crop_size // 2
    top = center_y - crop_size // 2
    right = center_x + crop_size // 2
    bottom = center_y + crop_size // 2

    # 确保裁剪框在图像范围内
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)

    # 裁剪图像
    cropped_image = image.crop((left, top, right, bottom))

    return cropped_image


# 使用示例：裁剪一个图像，中心坐标为(100.5, 150.5)，裁剪尺寸为56x56
cropped_image = crop_image_centered('E:\Common files\Work\GMDNN\dataset\FIND\dataD\c3001.png', 29.5, 27.5, 56)
cropped_image.show()
cropped_image.save('E:\Common files\Work\GMDNN\dataset\FIND\dataD\c1.png')