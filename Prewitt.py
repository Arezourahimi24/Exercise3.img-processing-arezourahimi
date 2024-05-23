import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

def convolution(img, filt):
    img_height, img_width = img.shape
    filt_height, filt_width = filt.shape
    pad_h = filt_height // 2
    pad_w = filt_width // 2
    result = np.zeros_like(img)
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), 'constant')
    for y in range(img_height):
        for x in range(img_width):
            result[y, x] = np.sum(padded_img[y:y+filt_height, x:x+filt_width] * filt)
    return result

def edge_prewitt(img):
    gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    filt_x = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]])
    filt_y = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]])
    edge_x = convolution(gray_img, filt_x)
    edge_y = convolution(gray_img, filt_y)
    edge_result = np.sqrt(np.square(edge_x) + np.square(edge_y))
    edge_result = np.clip(edge_result, 0, 255) 
    edge_result = edge_result.astype(np.uint8) 
    return edge_result

inp_image = mpimg.imread('pic.jpg')
detected_edges = edge_prewitt(inp_image)
plt.imshow(detected_edges, cmap='gray')
plt.title('Prewitt')
plt.axis('off')
plt.show()