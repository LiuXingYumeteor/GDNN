from PIL import Image


def crop_image(image_path, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    # 打开图像
    image = Image.open(image_path)

    # 获取图像的宽度和高度
    width, height = image.size

    # 计算裁剪区域的左上角和右下角坐标
    left = max(0, top_left_x)
    top = max(0, top_left_y)
    right = min(width, bottom_right_x)
    bottom = min(height, bottom_right_y)

    # 裁剪图像
    cropped_image = image.crop((left, top, right, bottom))

    # 显示裁剪后的图像
    cropped_image.show()
    cropped_image.save("E:\\Common files\\Work\\GMDNN\\dataset\PICTURE\cropped_image1.png")

# 调用函数进行裁剪
crop_image("E:\\Common files\\Work\\GMDNN\\dataset\data2\\3004.png", 70,67,492,547)
