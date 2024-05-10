import os
from PIL import Image


def rotate_images(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件是否是图像
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        
            # 构建完整的文件路径
            input_path = os.path.join(input_folder, filename)
            # 打开图像文件
            with Image.open(input_path) as img:
                # 提取文件名和扩展名
                base_name, ext = os.path.splitext(filename)

                # 旋转90度
                rotated_90 = img.rotate(90, expand=True)
                new_name_90 = str(int(base_name) + 205) + ext
                output_path_90 = os.path.join(output_folder, new_name_90)
                rotated_90.save(output_path_90)

                # 旋转180度
                rotated_180 = img.rotate(180, expand=True)
                new_name_180 = str(int(base_name) + 410) + ext
                output_path_180 = os.path.join(output_folder, new_name_180)
                rotated_180.save(output_path_180)

                # 旋转270度
                rotated_270 = img.rotate(270, expand=True)
                new_name_270 = str(int(base_name) + 615) + ext
                output_path_270 = os.path.join(output_folder, new_name_270)
                rotated_270.save(output_path_270)

                print(f"Processed {filename} -> {new_name_90}, {new_name_180}, {new_name_270}")

            # 设置输入和输出文件夹路径


input_folder = 'E:\\Common files\\Work\\GMDNN\\dataset\\Dataset\\TARGET'  # 请替换为实际的输入文件夹路径
output_folder = 'E:\\Common files\\Work\\GMDNN\\dataset\\Dataset\\TARGET'  # 请替换为实际的输出文件夹路径

# 调用函数处理图像
rotate_images(input_folder, output_folder)