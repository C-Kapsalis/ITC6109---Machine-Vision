### This script takes care of implementing pre-processing on the images of various types
# and creating 'masks' that act as a map capturing the outline and body of each guitar.
# Thanks to the high quality and consistency of the raw images used, we were able to create
# masks of very high quality, suitable for template matching under (very) controlled conditions.

### Importing Libraries and Dependencies ###
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_histogram(img_obj, bins_no, intended_range, xlim_range=None, ylim_range=None, mode='bin'):
    hist, bins = np.histogram(img_obj.ravel(), bins=bins_no, range=intended_range)
    cdf = hist.cumsum()
    cdf_normal = cdf * hist.max() / cdf.max()

    # Finding and showing the most frequent value in the histogram
    most_frequent_bin = np.argmax(hist)
    most_frequent_value = bins[most_frequent_bin]

    if mode == 'bin':
        return int(most_frequent_value)
    elif mode == 'message':
        print(f'The most frequent value of the histogram is {int(most_frequent_value)}, with a frequency of {int(hist[most_frequent_bin])}')
    elif mode == 'plot':
        print(f'The most frequent value of the histogram is {int(most_frequent_value)}, with a frequency of {int(hist[most_frequent_bin])}')
        plt.plot(hist, color='r')
        if xlim_range:
            plt.xlim(xlim_range)
        if ylim_range:
            plt.ylim(ylim_range)
        plt.show()


def create_cross_kernel(size, sigma):
    """
    Create a Gaussian cross kernel.
    Args:
        size (int): Size of the kernel (must be odd, e.g., 5x5).
        sigma (float): Standard deviation for the Gaussian.
    Returns:
        cross_kernel (np.array): Cross-shaped Gaussian kernel.
    """
    gaussian_kernel = cv2.getGaussianKernel(size, sigma)
    gaussian_2d_kernel = np.outer(gaussian_kernel, gaussian_kernel)

    # Create a cross mask
    cross_mask = np.zeros_like(gaussian_2d_kernel, dtype=np.uint8)
    center = size // 2
    cross_mask[center, :] = 1  # Horizontal line
    cross_mask[:, center] = 1  # Vertical line

    # Apply the cross mask to the Gaussian kernel
    cross_kernel = gaussian_2d_kernel * cross_mask
    cross_kernel /= cross_kernel.sum()  # Normalize
    return cross_kernel

def apply_cross_blur(image, kernel):
    """
    Apply a cross Gaussian blur to an image.
    Args:
        image (np.array): Input image (grayscale or color).
        kernel (np.array): Cross Gaussian kernel.
    Returns:
        blurred (np.array): Blurred image.
    """
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

if __name__ == "__main__":
    w_bg_guitar_file_abs_path = '../st_guitars/details/images/schecter_c_1_fr_s_sls_evil_twin_sbk.jpg'
    b_bg_guitar_file_abs_path = '../st_guitars/details/images/fender_player_plus_strat_mn_3csb.jpg'
    
    # Load the images into cv2 in color mode
    w_bg_img = cv2.imread(w_bg_guitar_file_abs_path)
    b_bg_img = cv2.imread(b_bg_guitar_file_abs_path)

    # Define universal column boundaries
    col1 = 195 + 5
    col2 = 405 - 5

    # Convert images to grayscale
    w_bg_img_lum = cv2.cvtColor(w_bg_img, cv2.COLOR_BGR2GRAY)
    b_bg_img_lum = cv2.cvtColor(b_bg_img, cv2.COLOR_BGR2GRAY)

    # Draw histograms for the central parts
    #plot_histogram(img_obj=w_bg_img_lum[:, col1:col2], bins_no=256, intended_range=[0, 256], mode='plot')
    #plot_histogram(img_obj=b_bg_img_lum[:, col1:col2], bins_no=256, intended_range=[0, 256], mode='plot')

    # Histogram equalization, blurring, and edge detection
    random_b_bg_guitar_file_abs_path = '../st_guitars/details/images/fender_sq_cv_70s_strat_hss_lrl_wal.jpg'#fender_player_plus_strat_mn_tqs.jpg'
    random_img = cv2.imread(random_b_bg_guitar_file_abs_path, cv2.IMREAD_GRAYSCALE)
    random_img = random_img[:, col1:col2]
    
    # hist equalization
    random_img_eq = cv2.equalizeHist(random_img)
    
    # gaussian blurring - we would only do this on other images that do not have such a consistent background
    kernel = create_cross_kernel(size=5, sigma=2)
    blurred_image = apply_cross_blur(random_img_eq, kernel)
    
    # sobel edge detection
    sobel_x = cv2.Sobel(random_img_eq, cv2.CV_64F, dx=1, dy=0, ksize=5)
    sobel_y = cv2.Sobel(random_img_eq, cv2.CV_64F, dx=0, dy=1, ksize=5)
    # Compute Gradient Magnitude and Orientation
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_orientation = np.arctan2(sobel_y, sobel_x)  # Radians (-π to π)

    # Normalize Gradient Magnitude to [0, 1]
    normalized_magnitude = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)

    # Map Gradient Orientation to Hue (H)
    hue = ((gradient_orientation + np.pi) / (2 * np.pi)) * 180  # Map [-π, π] to [0, 180] (OpenCV Hue scale)

    # Create HSV Image
    hsv_image = np.zeros((*gradient_orientation.shape, 3), dtype=np.float32)
    hsv_image[..., 0] = hue  # Hue channel
    hsv_image[..., 1] = 1.0  # Saturation channel (constant for vivid colors)
    hsv_image[..., 2] = normalized_magnitude  # Value channel (normalized gradient magnitude)

    # Convert HSV to RGB for Visualization
    hsv_image_uint8 = (hsv_image * 255).astype(np.uint8)
    rgb_image = cv2.cvtColor(hsv_image_uint8, cv2.COLOR_HSV2RGB)


	# I keep the gradients in the [0,1] scale, not normalized !!! 
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(random_img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Sobel X")
    plt.imshow(sobel_x, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Sobel Y")
    plt.imshow(sobel_y, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Gradient Magnitude")
    plt.imshow(rgb_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    
    
#     # Perform connected component analysis on the Sobel gradient magnitude
#     sobel_magnitude_uint8 = np.uint8(sobel_magnitude)  # Convert to uint8 for connected components
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sobel_magnitude_uint8, connectivity=4)
# 
#     # Display the results of connected component analysis
#     print(f"Number of components found (including background): {num_labels}")
#     for i in range(1, num_labels):  # Start from 1 to skip the background
#         print(f"Component {i}:")
#         print(f"  Area: {stats[i, cv2.CC_STAT_AREA]}")
#         print(f"  Bounding Box: (x={stats[i, cv2.CC_STAT_LEFT]}, y={stats[i, cv2.CC_STAT_TOP]}, "
#         	  f"width={stats[i, cv2.CC_STAT_WIDTH]}, height={stats[i, cv2.CC_STAT_HEIGHT]})")
#         print(f"  Centroid: (x={centroids[i][0]}, y={centroids[i][1]})")
# 
#     # Visualize the connected components
#     label_img = cv2.normalize(labels, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     label_img_color = cv2.applyColorMap(label_img, cv2.COLORMAP_JET)
# 	
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Gradient Magnitude")
#     plt.imshow(sobel_magnitude, cmap="gray")
#     plt.axis("off")
# 
#     plt.subplot(1, 2, 2)
#     plt.title("Connected Components")
#     plt.imshow(label_img_color)
#     plt.axis("off")
# 
#     plt.tight_layout()
#     plt.show()


    