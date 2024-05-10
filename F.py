def find_keys_with_value_zero(file_path):
    # 初始化一个列表来存储键值为0的键
    keys_with_zero_value = []

    # 打开文件以读取内容
    with open(file_path, 'r') as file:
        # 逐行读取文件
        for line in file:
            # 去除行尾的换行符，并分割行以获取键和值
            parts = line.strip().split(': ')
            if len(parts) == 2:
                key, value = parts
                # 尝试将值转换为整数，并检查是否为0
                try:
                    if int(value) != 255:
                        # 如果值为0，则将该键添加到列表中
                        keys_with_zero_value.append(key)
                except ValueError:
                    # 如果值无法转换为整数，则忽略该行并继续
                    continue

                    # 返回所有键值为0的键的列表
    return keys_with_zero_value


# ToDo 输入路径：指定txt文件的路径
txt_file_path = 'E:\\Common files\\Work\\GMDNN\\dataset\\L.txt'
# 调用函数并打印结果
keys_with_zero_value = find_keys_with_value_zero(txt_file_path)
if keys_with_zero_value:
    print(f"Found keys with value 0: {keys_with_zero_value}")
else:
    print("No keys with value 0 found in the file.")