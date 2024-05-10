from PIL import Image
import numpy as np


def convert_to_grayscale(image_path, output_path):
    # 打开图像文件
    with Image.open(image_path) as img:
        # 将图像转换为灰度模式
        gray_img = img.convert('L')

        # 保存灰度图像到指定路径
        gray_img.save(output_path)

    # 调用函数并传入图像路径及输出路径
def invert_image(image_path, output_path):
    # 打开图像
    with Image.open(image_path) as img:
        # 将图像转换为NumPy数组以便进行数学运算
        img_array = np.array(img)

        # 对图像进行反色处理：从255中减去每个像素值
        inverted_array = 255 - img_array

        # 将处理后的NumPy数组转换回Pillow图像对象
        inverted_img = Image.fromarray(inverted_array.astype(np.uint8))

        # 保存反色处理后的图像
        inverted_img.save(output_path)

convert_to_grayscale("E:\\Common files\\Work\\GMDNN\\DATA\\600\\cropped_image4.png", "E:\\Common files\\Work\\GMDNN\\DATA\\600\\cropped_image4.png")
invert_image("E:\\Common files\\Work\\GMDNN\\DATA\\600\\cropped_image4.png", "E:\\Common files\\Work\\GMDNN\\DATA\\600\\cropped_image4.png")