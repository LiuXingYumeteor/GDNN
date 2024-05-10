import math

# 初始化最小小数部分为1（因为小数部分不可能超过1，所以这是一个安全的初始值）
# 和对应的x值为None
min_fraction = 1.0
best_x = None

# 遍历1到200之间的所有整数
for x in range(113, 150):
    # 计算4x/10.18125的值
    value = 4 * x / 10.18125

    # 提取小数部分
    fraction = value - math.floor(value)

    # 如果当前小数部分小于之前记录的最小小数部分，则更新最小小数部分和对应的x值
    if fraction < min_fraction:
        min_fraction = fraction
        best_x = x

    # 输出结果
print(f"找到的最优x值为：{best_x}")
print(f"对应的小数部分为：{min_fraction}")