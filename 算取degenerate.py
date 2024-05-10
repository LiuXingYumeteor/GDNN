import os
import re


def process_numbers_in_file(file_path):
    # 打开原始文件并读取内容
    with open(file_path, 'r') as file:
        content = file.read()

        # 使用正则表达式找到所有的数字（可以是整数或浮点数）
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', content)

    # 对找到的数字进行处理
    modified_content = content
    for number in numbers:
        # 将字符串转换为浮点数，进行处理，然后四舍五入取整
        modified_number = round(float(number) / 10.18125)
        # 将处理后的数字替换回文本中
        modified_content = modified_content.replace(number, str(modified_number), 1)

        # 构造新文件的路径
    dir_name, file_name = os.path.split(file_path)
    file_base_name, file_ext = os.path.splitext(file_name)
    new_file_path = os.path.join(dir_name, f"{file_base_name}.modified{file_ext}")

    # 将处理后的内容写入新文件
    with open(new_file_path, 'w') as file:
        file.write(modified_content)

    print(f"Processed file saved as: {new_file_path}")


# 调用函数，传入要处理的文件夹路径
# 请将'your_directory_path'替换为您要处理的文件夹的实际路径
for filename in os.listdir('E:\\Common files\\Work\\GMDNN\\dataset\\FIND\\dataD'):
    if filename.endswith('.txt'):
        file_path = os.path.join('E:\\Common files\\Work\\GMDNN\\dataset\\FIND\\dataD', filename)
        process_numbers_in_file(file_path)