import cv2
import numpy as np

# 读取图像
image_path = 'E:\\Common files\\mypaper\\GMDNN\\dataset\\target\\a11.png'  # 请替换为你的图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像

# 应用高斯模糊来减少图像噪声
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# 应用Canny边缘检测
edges = cv2.Canny(blurred, 50, 150)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 筛选轮廓（可选，根据具体情况调整）
# 假设电极轮廓具有较大的面积，可以根据面积进行筛选
min_area = 50  # 最小面积阈值，需要根据实际情况调整
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# 计算轮廓的中心点
centers = []
for cnt in filtered_contours:
    # 计算轮廓的矩形边界框
    x, y, w, h = cv2.boundingRect(cnt)
    # 计算矩形框的中心点
    center = (x + w // 2, y + h // 2)
    centers.append(center)

# 绘制中心点（可选，用于可视化）
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # 转换为彩色图像以便绘制
for center in centers:
    cv2.circle(output, center, 5, (0, 255, 0), -1)  # 绘制绿色圆点表示中心点

# 显示结果（可选）
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)
cv2.imshow('Centers', output)
cv2.waitKey(0)  # 等待按键
cv2.destroyAllWindows()  # 关闭窗口

# 输出中心点坐标
print("Electrode Centers:")
for center in centers:
    print(center)