import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def roberts_operator(image):
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])

    gradient_x = convolve(image, kernel_x)
    gradient_y = convolve(image, kernel_y)

    return np.sqrt(gradient_x**2 + gradient_y**2)

def sobel_operator(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gradient_x = convolve(image, kernel_x)
    gradient_y = convolve(image, kernel_y)

    return np.sqrt(gradient_x**2 + gradient_y**2)

def main():
    image_path = "D:\\Perkuliahan\\S5\\Pengolahan Citra Digital\\s9.2\\tiger.jpg"
    image = imageio.imread(image_path, mode="F") / 255.0

    roberts_edges = roberts_operator(image)
    sobel_edges = sobel_operator(image)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Robert's Operator")
    plt.imshow(roberts_edges, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Sobel Operator")
    plt.imshow(sobel_edges, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
