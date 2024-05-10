import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 原始数据点
x = np.array([1250, 1333, 1428, 1538, 1667, 1818, 2000, 2083, 2222, 2500, 2631, 2857])
y = np.array([0.366, 0.334, 0.310, 0.275, 0.262, 0.223, 0.213, 0.181, 0.178, 0.150, 0.134, 0.129])

# 使用插值使曲线更光滑
X_Y_Spline = make_interp_spline(x, y)
X_ = np.linspace(x.min(), x.max(), 500)
Y_ = X_Y_Spline(X_)

# 绘制原始数据点
#plt.scatter(x, y, color='black', label='Original data')

# 绘制光滑曲线
plt.plot(X_, Y_, color='black',label='Smooth Curve')

# 添加图例和标签
plt.legend()
plt.title('Smooth Curve Fitting to Data Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(False)

# 显示图形
plt.show()