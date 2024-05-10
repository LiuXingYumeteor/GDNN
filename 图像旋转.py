from PIL import Image


def rotate_image(image_path, degrees_to_rotate):
    # 打开图像
    image = Image.open(image_path)

    # 旋转图像
    rotated_image = image.rotate(degrees_to_rotate)

    # 显示旋转后的图像
    rotated_image.show()
    rotated_image.save("D:\\GMDNN\\dataset\\900nm\\4\\9004.tif")


# 调用函数进行旋转
rotate_image("D:\\GMDNN\\dataset\\900nm\\4\\4_MMStack_Default.ome.tif", 180)