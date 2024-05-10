import os
def read_and_calculate_centers(file_path):
    """
    读取txt文件中的坐标数据并计算中心坐标。
    :param file_path: txt文件路径。
    :return: 中心坐标列表。
    """
    centers = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去除行尾的换行符并分割坐标
            coords = line.strip().split(',')
            if len(coords) == 4:
                # 将字符串转换为整数
                top_left = (int(coords[0]), int(coords[1]))
                bottom_right = (int(coords[2]), int(coords[3]))

                # 计算中心坐标
                center_x = (top_left[0] + bottom_right[0]) // 2
                center_y = (top_left[1] + bottom_right[1]) // 2

                # 将中心坐标添加到列表中
                centers.append((center_x, center_y))
    return centers


# 使用示例
coordinates_file_path = 'E:\\Common files\\Work\\GMDNN\\PICTURE\\TIAOZHENG\\4.txt'  # 你的坐标txt文件路径
centers = read_and_calculate_centers(coordinates_file_path)

# 打印所有中心坐标
for i, center in enumerate(centers):
    print(f"{center}")



