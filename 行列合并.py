import os


def append_to_lines(directory):
    # 定义一个映射，根据文件名的最后一个字符确定要追加的字符串
    append_map = {
        '1': ',292',
        '2': ',907',
        '3': ',1521',
        '4': ',2136',
        '5': ',2751',
        '6': ',3366',
        '7': ',3980',
        '8': ',4595'
    }

    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):  # 确保文件是.txt格式
            last_char = filename[-5]  # 获取文件名倒数第二个字符（因为.txt占据了最后三个字符中的两个）
            print(last_char)
            if last_char.isdigit() and last_char in append_map:  # 确保是数字且在映射中
                file_path = os.path.join(directory, filename)
                new_content = []

                # 读取文件内容，修改后存入new_content列表
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        new_line = line.rstrip('\n')  # 移除行尾的换行符
                        new_line += append_map[last_char] + '\n'  # 追加相应的字符串并添加换行符
                        new_content.append(new_line)

                        # 将修改后的内容写回文件
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.writelines(new_content)

                # 调用函数，传入要处理的文件夹路径


# 请将'your_directory_path'替换为您要处理的文件夹的实际路径
append_to_lines('E:\\Common files\\Work\\GMDNN\\dataset\\FIND\\data0')