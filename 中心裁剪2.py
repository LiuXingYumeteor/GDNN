import os
from PIL import Image


def center_crop(image, target_width, target_height):
    # 获取图像的原始宽度和高度
    width, height = image.size

    # 计算裁剪的起始位置
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    # 进行中心裁剪
    return image.crop((left, top, right, bottom))


def crop_images_in_folder(folder_path, target_width, target_height):
    # 确保目标文件夹存在
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

        # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)

            try:
                # 打开图像
                with Image.open(file_path) as img:
                    # 进行中心裁剪
                    cropped_image = center_crop(img, target_width, target_height)

                    # 保存裁剪后的图像（可以选择覆盖原文件或保存到新文件夹）
                    # 这里我们选择保存到原文件夹，并覆盖原文件
                    cropped_image.save(file_path)
                    print(f"Cropped {filename} successfully.")
            except Exception as e:
                print(f"An error occurred while cropping {filename}: {e}")

            # 使用示例


# 假设我们有一个名为'images'的文件夹，其中包含要进行裁剪的图像
crop_images_in_folder('E:\\Common files\\GAN\\MTF', 64, 64)