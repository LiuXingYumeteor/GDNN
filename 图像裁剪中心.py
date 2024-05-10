from PIL import Image


def crop_center(img, crop_width=700, crop_height=700):
    # 获取图像的宽度和高度
    width, height = img.size

    # 计算裁剪的起始坐标
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = (width + crop_width) / 2
    bottom = (height + crop_height) / 2

    # 确保裁剪区域不超过图像边界
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)

    # 裁剪图像并返回
    return img.crop((left, top, right, bottom))


# 加载图像
img = Image.open('E:\\Common files\\mypaper\\GMDNN\\dataset\\target\\a11-01.tif')  # 请替换为你的图像路径

# 裁剪图像
cropped_img = crop_center(img)

# 显示裁剪后的图像（可选）
cropped_img.show()

# 保存裁剪后的图像（可选）
cropped_img.save('E:\\Common files\\mypaper\\GMDNN\\dataset\\target\\a11.tif')  # 保存为新的文件，可以自定义文件名和路径