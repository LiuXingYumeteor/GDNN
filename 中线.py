from PIL import Image


def read_middle_row_gray_values(image_path):
    # 加载灰度图像
    image = Image.open(image_path).convert('L')

    # 获取图像尺寸
    width, height = image.size

    # 计算中间行的索引
    middle_row_index = height // 2

    # 读取中间行的灰度值
    pixels = image.load()
    gray_values = [pixels[x, middle_row_index] for x in range(width)]

    return gray_values


# 图像路径
image_path = 'G:\\lxy\\TU6c\\6\\O1691.png'

# 读取灰度图中间水平行的灰度值
gray_values = read_middle_row_gray_values(image_path)

# 打印灰度值
print(gray_values)