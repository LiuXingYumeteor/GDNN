# 打开原始文件和新文件
with open('E:\\Common files\\Work\\GMDNN\\dataset\\FIND\\dataD\\4.modified.txt', 'r') as original_file, open('E:\\Common files\\Work\\GMDNN\\dataset\\FIND\\dataD\\4.txt', 'w') as modified_file:
    # 逐行读取原始文件
    for line in original_file:
        # 移除行尾的换行符，并以逗号为分隔符拆分字符串
        numbers = line.strip().split(',')

        # 确保每行有两个数字
        if len(numbers) == 2:
            try:
                # 将字符串转换为整数，进行加减操作
                first_number = int(numbers[0]) - 1
                second_number = int(numbers[1]) +1

                # 将结果转换回字符串，并用逗号连接，然后写入新文件
                modified_line = f"{first_number},{second_number}\n"
                modified_file.write(modified_line)
            except ValueError:
                # 如果转换失败（即不是整数），则跳过该行或进行其他处理
                print(f"Error: Unable to process line '{line.strip()}'")
        else:
            # 如果每行的数字数量不是2，则跳过该行或进行其他处理
            print(f"Error: Unexpected number of values in line '{line.strip()}'")