from PIL import Image  
import numpy as np  
  
def calculate_mse(image1_path, image2_path):  
    # 加载图像  
    image1 = Image.open(image1_path).convert('L')  # 转换为灰度图像以便比较  
    image2 = Image.open(image2_path).convert('L')  # 转换为灰度图像以便比较  
      
    # 确保两张图像尺寸相同  
    if image1.size != image2.size:  
        raise ValueError("Images must have the same dimensions.")  
      
    # 将图像转换为numpy数组  
    image1_array = np.array(image1)  
    image2_array = np.array(image2)  
      
    # 计算MSE  
    mse = np.mean((image1_array - image2_array) ** 2)  
      
    return mse  
  
# 使用示例  
image1_path = 'G:\\TU5\\1732.png'  # 替换为你的图像路径
image2_path = 'G:\\TU5\\9-173.png'  # 替换为你的图像路径
mse_value = calculate_mse(image1_path, image2_path)  
print(f"The Mean Squared Error (MSE) between the two images is: {mse_value}")