import numpy as np
import matplotlib.pyplot as plt

# 假设的入射角度范围，从0到90度
incident_angles = np.linspace(0, 90, 100)  # 100个点

# 简化模型下，假设衍射效率随入射角线性变化（仅为演示，非真实物理模型）
# 这里我们假设效率从0度的0.1线性增加到90度的0.9
diffraction_efficiency = 0.1 + (incident_angles / 90) * 0.8

# 绘制衍射效率随入射角（透射波束）的变化图
plt.figure(figsize=(10, 6))
plt.plot(incident_angles, diffraction_efficiency, label='Diffraction Efficiency')
plt.xlabel('Incident Angle (degrees)')
plt.ylabel('Diffraction Efficiency')
plt.title('Diffraction Efficiency vs. Incident Angle')
plt.legend()
plt.grid(True)
plt.show()
