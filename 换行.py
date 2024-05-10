import os


def insert_newlines_if_gap(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)

                # 读取文件内容
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                    # 假设整个文件内容在一行上，我们将其按逗号分割
                if lines:
                    numbers_str = lines[0].replace('\n', '').replace(' ', '')  # 移除空白行和空格
                    numbers = [int(n) for n in numbers_str.split(',') if n]  # 转换为整数列表，忽略空字符串

                    # 处理数字列表，如果相邻数字差值大于150，则插入换行符
                    processed_content = ''
                    for i in range(len(numbers) - 1):
                        processed_content += str(numbers[i]) + ','
                        if numbers[i + 1] - numbers[i] > 150:
                            processed_content += '\n'
                            # 添加最后一个数字（没有后续的逗号或换行符）
                    processed_content += str(numbers[-1])

                    # 将处理后的内容写回文件
                with open(file_path, 'w') as f:
                    f.write(processed_content)

                # 调用函数，传入要处理的文件夹路径


# 请将'your_directory_path'替换为您要处理的文件夹的实际路径
insert_newlines_if_gap('E:\\Common files\\Work\\GMDNN\\dataset\\FIND')