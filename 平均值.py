import os


def calculate_first_last_avg(directory):
    # 遍历文件夹中的所有txt文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                new_file_path = os.path.join(root, 'z' + file)  # 新文件名加上前缀z

                # 读取文件内容
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                    # 存储每一行第一个数与最后一个数的平均值
                averages = []

                # 处理每一行内容
                for line in lines:
                    # 去除行尾的换行符，并按逗号分割字符串得到数字列表
                    numbers_str = line.rstrip('\n').split(',')
                    # 确保至少有两个数字才计算平均值
                    if len(numbers_str) >= 2:
                        # 将字符串转换为整数并计算第一个数与最后一个数的平均值
                        first_num = int(numbers_str[0])
                        last_num = int(numbers_str[-1])  # 最后一个数字，注意索引是-1
                        average = (first_num + last_num) / 2
                        averages.append(average)

                        # 将平均值写入新的txt文件
                with open(new_file_path, 'w') as f:
                    for avg in averages:
                        f.write(f"{avg}\n")  # 每个平均值后加一个换行符


# 调用函数，传入要处理的文件夹路径
# 请将'your_directory_path'替换为您要处理的文件夹的实际路径
calculate_first_last_avg('E:\\Common files\\Work\\GMDNN\\dataset\\PICTURE\\PICTURE\\data')