import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def compute_dominant_gradient_orientation(image, bins=8):
    """
    Compute gradient orientations per color channel and take the dominant channel.
    Quantize the orientations using the specified number of bins.
    
    Args:
        image (np.array): Input image.
        bins (int): Number of bins for quantization.
    
    Returns:
        quantized_orientations (np.array): Quantized gradient orientations.
    """
    gradients = []
    for channel in cv2.split(image):
        grad_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        orientation = np.arctan2(grad_y, grad_x) % (2 * np.pi)
        gradients.append((magnitude, orientation))
    
    magnitudes, orientations = zip(*gradients)
    max_channel_idx = np.argmax(np.array(magnitudes), axis=0)
    dominant_orientations = np.choose(max_channel_idx, orientations)

    # Quantize orientations to the specified number of bins
    bin_width = 2 * np.pi / bins
    quantized_orientations = np.floor(dominant_orientations / bin_width).astype(np.uint8)
    return quantized_orientations, magnitude


def create_template(images_folder, region_coordinates, bins=8):
    """
    Create a robust template based on dominant gradient orientations.
    
    Args:
        images_folder (str): Folder containing input images.
        region_coordinates (tuple): (row1, row2, col1, col2) coordinates of the region of interest.
        bins (int): Number of bins for gradient orientation quantization.
    
    Returns:
        template (np.array): The final template as a quantized orientation map.
    """
    row1, row2, col1, col2 = region_coordinates
    gradient_accumulator = None

    for file_name in os.listdir(images_folder):
        if not file_name.endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(images_folder, file_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load {file_name}")
            continue

        # Crop region of interest
        roi = image[row1:row2, col1:col2]
        if roi.size == 0:
            print(f"Warning: Empty ROI in {file_name}")
            continue

        # Preprocessing: Histogram Equalization and Gaussian Blurring
        #for i in range(roi.shape[2]):
        #    roi[:, :, i] = cv2.equalizeHist(roi[:, :, i])
        roi = cv2.GaussianBlur(roi, (5, 5), sigmaX=1.5, sigmaY=1.5)

        # Compute quantized dominant gradient orientations
        quantized_orientations, magnitude = compute_dominant_gradient_orientation(roi, bins=bins)

        if gradient_accumulator is None:
            gradient_accumulator = np.zeros_like(quantized_orientations, dtype=np.float64)

        gradient_accumulator += magnitude * quantized_orientations  # weighted avg based on normalized magnitude to avoid noise accumulation

    if gradient_accumulator is None:
        raise ValueError("No valid regions processed!")



    # Normalize the accumulated gradient orientations
    gradient_accumulator = cv2.normalize(gradient_accumulator, None, 0, 255, cv2.NORM_MINMAX)
    _, gradient_accumulator = cv2.threshold(gradient_accumulator, 70, 255, cv2.THRESH_BINARY)  
    gradient_accumulator = cv2.equalizeHist(gradient_accumulator.astype(np.uint8))


    template = gradient_accumulator / np.max(gradient_accumulator)#np.sum(gradient_accumulator)
    template[template <= 0.4] = 0 
	
    # equalization to make contrasts sharper
	
	
    cv2.imshow('template', template)
    cv2.waitKey(0)
    cv2.destroyWindow('template')
	
    return template


def save_template(template, output_path, title):
    """
    Save the template as an image for visualization.
    
    Args:
        template (np.array): Template array.
        output_path (str): Path to save the template.
        title (str): Title for the saved template image.
    """
    np.save(output_path, template)
    print(f"Template saved to {output_path}")



if __name__ == "__main__":
    # Define paths
    telecaster_folder = "../tele_guitars/details/images"
    head_template_output = "tele_head_template.png"
    upper_body_template_output = "tele_upper_body_template.png"

    # Define regions
    HEAD_COORDINATES = (2, 120, 263, 327)
    UPPER_BODY_COORDINATES = (322, 452, 208, 389)

    # Create templates for head and upper body
    print("Processing head region...")
    head_template = create_template(telecaster_folder, HEAD_COORDINATES, bins=8)

    print("Processing upper body region...")
    upper_body_template = create_template(telecaster_folder, UPPER_BODY_COORDINATES, bins=8)

    # Save templates
    save_template(head_template, head_template_output, "Telecaster Head Template")
    save_template(upper_body_template, upper_body_template_output, "Telecaster Upper Body Template")


