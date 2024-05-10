import os
from PIL import Image


def downscale_image(input_image_path, output_image_path, target_width, target_height):
    # 打开原始图像
    with Image.open(input_image_path) as img:
        # 使用ANTIALIAS滤波器进行高质量的下采样
        resized_img = img.resize((target_width, target_height), Image.ANTIALIAS)
        # 保存调整大小后的图像
        resized_img.save(output_image_path)


def downscale_images_in_folder(folder_path, target_width, target_height):
    # 确保目标文件夹存在
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

        # 创建新文件夹来存储降分辨率后的图像
    resized_folder_path = os.path.join(folder_path, 'resized_images')
    os.makedirs(resized_folder_path, exist_ok=True)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为图像文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 构建完整的原始文件路径
            original_image_path = os.path.join(folder_path, filename)

            # 构建输出文件路径，在新文件夹中保存降分辨率后的图像
            resized_filename = filename  # 可以选择改变文件名，但在这里我们保持原文件名
            resized_image_path = os.path.join(resized_folder_path, resized_filename)

            # 调用降分辨率函数
            try:
                downscale_image(original_image_path, resized_image_path, target_width, target_height)
                print(f"Downscaled {filename} to {resized_filename} successfully and saved in {resized_folder_path}.")
            except Exception as e:
                print(f"An error occurred while downscaling {filename}: {e}")

            # 使用示例


# 假设我们有一个名为'images'的文件夹，其中包含要进行降分辨率处理的图像
downscale_images_in_folder("E:\\Common files\\Work\\GMDNN\\PICTURE\\p", 1076, 1222)