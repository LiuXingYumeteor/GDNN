import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image in grayscale
image_path = 'E:\\Common files\\Work\\Aberration\\1.png'  # Replace 'path_to_your_image.png' with the actual image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error loading image")
else:
    # Extract the middle row of pixels
    middle_row = image[image.shape[0] // 2, :]

    # Generate x coordinates for each pixel in the middle row
    x_coords = np.arange(len(middle_row))

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, middle_row, label='Pixel Intensity')

    # Mark every 32nd pixel on the x-axis
    plt.xticks(np.arange(0, len(middle_row), 32))

    # Adding labels and title
    plt.xlabel('Pixel Position')
    plt.ylabel('Grayscale Intensity(a.u.)')
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()
    plt.savefig('contrast_line_graph.png', dpi=300)
