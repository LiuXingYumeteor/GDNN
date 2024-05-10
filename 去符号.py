import os
import re


def remove_character_from_txt_files(directory, character):
    # 遍历指定文件夹中的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名是否为.txt
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)

                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # 使用正则表达式删除指定字符
                # 注意：这里我们使用re.escape来确保字符被正确处理，即使它是一个正则表达式特殊字符
                content_without_character = re.sub(re.escape(character), '', content)

                # 将处理后的内容写回文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content_without_character)

                # 调用函数，传入要处理的文件夹路径和要删除的字符


# 请将'your_directory_path'替换为您要处理的文件夹的实际路径
remove_character_from_txt_files('E:\\Common files\\Work\\GMDNN\\dataset\\FIND', "]")