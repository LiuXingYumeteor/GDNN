import cv2
import numpy as np

def locate_and_draw_bbox(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    original_image = image.copy()

    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 轮廓查找
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)

    # 近似轮廓
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    # 透视变换
    rect = cv2.minAreaRect(approx)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 画出边界框
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Located Object with Bounding Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用函数并传入图像路径
locate_and_draw_bbox('E:\\Common files\\mypaper\\GMDNN\\dataset\\target\\a11.png')
