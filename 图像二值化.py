import cv2
import numpy as np

# Load the image in grayscale
image_path = 'E:\\Common files\\GAN\\t\\f9.png'  # Replace this with the path to your image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error loading image.")
else:
    # Apply threshold
    _, thresholded_image = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)

    # Save the resulting image
    cv2.imwrite('E:\\Common files\\GAN\\t\\f91.png', thresholded_image)

    print("Image processed and saved as 'thresholded_image.png'.")
