import cv2
import numpy as np


def jaccard_similarity(img1, img2):
    # 确保两张图像大小相同
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions.")

        # 计算交集：两个图像对应像素都为1的点
    intersection = np.logical_and(img1, img2)

    # 计算并集：两个图像对应像素至少有一个为1的点
    union = np.logical_or(img1, img2)

    # 计算交集和并集中为True的像素个数
    intersection_count = np.sum(intersection)
    union_count = np.sum(union)

    # 避免除以0的情况
    if union_count == 0:
        return 0

        # 计算并返回杰卡德相似系数
    jaccard_sim = intersection_count / union_count
    return jaccard_sim


# 读取两张二值化图像
image_path1 = 'G:\\TU6c\\169.png'
image_path2 = 'G:\\TU6c\\F169.png'

img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
img1 = (img1 > 128).astype(np.uint8)  # 假设阈值为128，将图像二值化

img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
img2 = (img2 > 128).astype(np.uint8)  # 假设阈值为128，将图像二值化

# 计算并打印杰卡德相似系数
similarity = jaccard_similarity(img1, img2)
print(f'Jaccard Similarity: {similarity}')