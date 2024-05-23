import numpy as np
import matplotlib.pyplot as plt

def gaussian_function(size, sigma):
    gaussian_kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(
            -((x - (size - 1) / 2)**2 + (y - (size - 1) / 2)**2) / (2 * sigma**2)
        ), (size, size)
    )
    return gaussian_kernel / np.sum(gaussian_kernel)

def comp_sobel_filter(image):
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gradient_x = convolution(image, sobel_kernel_x)
    gradient_y = convolution(image, sobel_kernel_y)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    direction = np.arctan2(gradient_y, gradient_x)
    return magnitude, direction

def convolution(image, kernel):
    kernel_size = kernel.shape[0]
    padding = kernel_size // 2
    padded_image = np.pad(image, padding, mode='edge')
    convolved_image = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            convolved_image[y, x] = np.sum(padded_image[y:y+kernel_size, x:x+kernel_size] * kernel)
    return convolved_image

def perform_non_maximum_suppression(mag, dir):
    rows, cols = mag.shape
    suppressed_image = np.zeros_like(mag)
    quantized_angle = (np.round(dir * (4.0 / np.pi)) % 4).astype(int)
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if quantized_angle[i, j] == 0:
                suppressed_image[i, j] = mag[i, j] if (mag[i, j] > mag[i, j+1] and mag[i, j] > mag[i, j-1]) else 0
            elif quantized_angle[i, j] == 1:
                suppressed_image[i, j] = mag[i, j] if (mag[i, j] > mag[i-1, j+1] and mag[i, j] > mag[i+1, j-1]) else 0
            elif quantized_angle[i, j] == 2:
                suppressed_image[i, j] = mag[i, j] if (mag[i, j] > mag[i-1, j] and mag[i, j] > mag[i+1, j]) else 0
            elif quantized_angle[i, j] == 3:
                suppressed_image[i, j] = mag[i, j] if (mag[i, j] > mag[i-1, j-1] and mag[i, j] > mag[i+1, j+1]) else 0
    return suppressed_image

def double_threshold(image, low_ratio=0.05, high_ratio=0.09):
    high_threshold = image.max() * high_ratio
    low_threshold = high_threshold * low_ratio
    strong_edges = (image >= high_threshold)
    weak_edges = (image >= low_threshold) & (image < high_threshold)
    return strong_edges, weak_edges

def track_edges(image, strong_edges, weak_edges):
    rows, cols = image.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if weak_edges[i, j]:
                if strong_edges[i-1:i+2, j-1:j+2].any():
                    strong_edges[i, j] = True
                else:
                    weak_edges[i, j] = False
    return strong_edges

def detect_edges_canny(image):
    kernel_size = 7
    sigma = 1.4
    gaussian_kernel = gaussian_function(kernel_size, sigma)
    smoothed_image = convolution(image, gaussian_kernel)
    gradient_magnitude, gradient_direction = comp_sobel_filter(smoothed_image)
    suppressed_image = perform_non_maximum_suppression(gradient_magnitude, gradient_direction)
    strong_edges, weak_edges = double_threshold(suppressed_image)
    final_edges = track_edges(suppressed_image, strong_edges, weak_edges)
    return final_edges

inp_image = plt.imread('pic.jpg')

grayscale_image = np.mean(inp_image, axis=2)

edges = detect_edges_canny(grayscale_image)

plt.imshow(edges, cmap='gray')
plt.title('Canny')
plt.axis('off')
plt.show()