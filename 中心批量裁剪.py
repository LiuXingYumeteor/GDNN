import os
from PIL import Image


def read_center_coordinates(file_path):
    """
    读取txt文件中的中心坐标数据。
    :param file_path: txt文件路径。
    :return: 中心坐标列表。
    """
    centers = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去除行尾的换行符，并提取中心坐标
            center_str = line.strip().strip('()')  # 假设坐标格式被括号包围，如(15, 15)
            x, y = map(int, center_str.split(','))
            centers.append((x, y))
    return centers


def crop_images_at_centers(image_path, centers_file_path, output_dir, crop_size=(128, 64)):
    """
    根据中心坐标裁剪图像并保存到指定路径。
    :param image_path: 图像路径。
    :param centers_file_path: 中心坐标txt文件路径。
    :param output_dir: 输出目录路径。
    :param crop_size: 裁剪尺寸（宽，高）。
    """
    # 读取中心坐标
    centers = read_center_coordinates(centers_file_path)

    # 打开图像
    img = Image.open(image_path)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历中心坐标并进行裁剪
    for i, (center_x, center_y) in enumerate(centers):
        # 计算裁剪框的左上角和右下角坐标
        left = center_x - crop_size[0] // 2
        top = center_y - crop_size[1] // 2
        right = left + crop_size[0]
        bottom = top + crop_size[1]

        # 确保裁剪坐标在图像范围内
        left = max(0, left)
        top = max(0, top)
        right = min(img.width, right)
        bottom = min(img.height, bottom)

        # 如果裁剪区域无效（即宽或高小于所需尺寸），则跳过此次裁剪
        if right - left < crop_size[0] or bottom - top < crop_size[1]:
            print(f"Skipping center {i + 1}: ({center_x}, {center_y}) - insufficient image area.")
            continue

            # 裁剪图像
        cropped_img = img.crop((left, top, right, bottom))

        # 如果裁剪出的图像尺寸大于所需尺寸，则进行中心裁剪至所需尺寸
        if cropped_img.width > crop_size[0] or cropped_img.height > crop_size[1]:
            new_left = (cropped_img.width - crop_size[0]) // 2
            new_top = (cropped_img.height - crop_size[1]) // 2
            cropped_img = cropped_img.crop((new_left, new_top, new_left + crop_size[0], new_top + crop_size[1]))

            # 保存裁剪后的图像
        output_path = os.path.join(output_dir, f"{i + 169 }.png")#1,65,105,169
        cropped_img.save(output_path)
        print(f"Saved cropped image {i + 1} to {output_path}")

    # 使用示例


image_path = 'E:\\Common files\\Work\\GMDNN\\DATA\\nogrant2\\nogrant2\\4.png'  # 你的图像路径
centers_file_path = 'E:\\Common files\\Work\\GMDNN\\DATA\\Center\\4.txt'  # 你的中心坐标txt文件路径
output_directory = r'E:\\Common files\\Work\\GMDNN\\DATA\\nogrant2\\DEG'  # 输出目录路

crop_images_at_centers(image_path, centers_file_path, output_directory)