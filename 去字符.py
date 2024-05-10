import os
import re


def remove_pixel_from_txt_files(directory):
    # 遍历指定文件夹中的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名是否为.txt
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)

                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # 使用正则表达式删除"Pixel"字样（不区分大小写）
                content_without_pixel = re.sub(r'Pixel ', '', content, flags=re.IGNORECASE)

                # 将处理后的内容写回文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content_without_pixel)

                # 调用函数，传入要处理的文件夹路径


# 请将'your_directory_path'替换为您要处理的文件夹的实际路径
remove_pixel_from_txt_files('E:\\Common files\\Work\\GMDNN\\dataset')