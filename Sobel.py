import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def edge_sobel(img):
    gray_scale = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    
    sobel_kernel_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    sobel_kernel_y = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])
    grad_x = convolution(gray_scale, sobel_kernel_x)
    grad_y = convolution(gray_scale, sobel_kernel_y)
    combined_gradients = np.sqrt(grad_x**2 + grad_y**2)
    combined_gradients *= 255.0 / np.max(combined_gradients)
    return combined_gradients

def convolution(img, ker):
    ker_height, ker_width = ker.shape
    img_height, img_width = img.shape
    pad_height = ker_height // 2
    pad_width = ker_width // 2
    padded_img = np.zeros((img_height + 2 * pad_height, img_width + 2 * pad_width))
    padded_img[pad_height:img_height + pad_height, pad_width:img_width + pad_width] = img
    result_img = np.zeros_like(img)
    for x in range(img_height):
        for y in range(img_width):
            result_img[x, y] = np.sum(padded_img[x:x + ker_height, y:y + ker_width] * ker)
    
    return result_img

inp_image = mpimg.imread('pic.jpg')
detected_edges = edge_sobel(inp_image)
plt.imshow(detected_edges, cmap='gray')
plt.title('Sobel')
plt.axis('off')
plt.show()