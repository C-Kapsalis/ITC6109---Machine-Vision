import cv2

import numpy as np

import matplotlib.pyplot as plt



#this is good but not perfect

def rotate_image_head(image, angle):

   (h, w) = image.shape[:2]

   center = (w // 2, h // 2)

   M = cv2.getRotationMatrix2D(center, angle, 1.0)

   rotated = cv2.warpAffine(image, M, (w, h))

   return rotated




def compute_dominant_gradient_orientation_head(image, bins=8):

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


    # Quantize orientations

    bin_width = 2 * np.pi / bins

    quantized_orientations = np.floor(dominant_orientations / bin_width).astype(np.uint8)

    return quantized_orientations


def preprocess_target_image_head(target_image, bins=8, edge_threshold1=100, edge_threshold2=200, morph_ksize=3, iterations=1):

    blur_ksize = 7

    blurred_image = cv2.GaussianBlur(target_image, (blur_ksize, blur_ksize), sigmaX=1.5, sigmaY=1.5)

    

    edges = cv2.Canny(blurred_image, threshold1=edge_threshold1, threshold2=edge_threshold2)

    

    quantized_orientations = compute_dominant_gradient_orientation_head(edges, bins)

    

    quantized_orientations = cv2.normalize(quantized_orientations, None, 0, 255, cv2.NORM_MINMAX)

    threshold_value = 147

    _, quantized_orientations = cv2.threshold(quantized_orientations, threshold_value, 255, cv2.THRESH_BINARY)

    

    morph_kernel = np.ones((morph_ksize, morph_ksize), np.uint8)

    quantized_orientations = cv2.dilate(quantized_orientations, morph_kernel, iterations=iterations)


    normalized_target = quantized_orientations / np.max(quantized_orientations)

    

    return normalized_target


def match_template_with_orientation_head(template, target, threshold=0.885, bins=8):

    preprocessed_target = preprocess_target_image_head(cv2.GaussianBlur(target, (5, 5), sigmaX=1.5, sigmaY=1.5), bins)

    

    template_h, template_w = template.shape

    target_h, target_w = preprocessed_target.shape


    result = np.zeros((target_h - template_h + 1, target_w - template_w + 1), dtype=np.float64)


    bin_width = 2 * np.pi / bins

    for y in range(target_h - template_h):

        for x in range(target_w - template_w):
			# region of interest - roi from the target/input image 
            temp = preprocessed_target[y:y + template_h, x:x + template_w]

            if np.all(temp == 0) or np.all(temp == 255) or np.all(temp == 254):

                cos_similarity = 0

            else:

                max_temp = np.max(temp)

                if max_temp == 0:

                    cos_similarity = 0

                else:

                    roi = temp #= (temp / max_temp) * np.max(template)
						
                    diff_ = np.abs(template-roi)
                    diff_ = np.minimum(diff_, 360-diff_)
                    cos_similarity = np.sum(np.cos(np.radians(diff_)))
                    #cos_similarity = np.mean(np.cos((template - roi) * bin_width))

            result[y, x] = cos_similarity


    match_mask = (result >= threshold).astype(np.uint8)

    return match_mask, result


def get_best_match_head(template_image, target_image, rotation_degrees, threshold=0.885, bins=8):

    best_match = None

    best_rotation = None

    best_result = None

    best_overlay = None


    for angle in rotation_degrees:

        rotated_template = rotate_image_head(template_image, angle)

        print(f"Matching with template rotated by {angle} degrees")

        match_mask, result = match_template_with_orientation_head(rotated_template, target_image, threshold=threshold, bins=bins)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)


        if best_result is None or max_val > best_result:

            best_result = max_val

            best_match = (max_loc, rotated_template)

            best_rotation = angle


    # Apply the best match to the target image

    if best_match is not None:

        max_loc, rotated_template = best_match

        template_h, template_w = rotated_template.shape

        roi = target_image[max_loc[1]:max_loc[1] + template_h, max_loc[0]:max_loc[0] + template_w]


        resized_template = cv2.resize(rotated_template, (roi.shape[1], roi.shape[0]))


        if resized_template.dtype != np.uint8:

            resized_template = cv2.normalize(resized_template, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


        if len(roi.shape) == 3 and len(resized_template.shape) == 2:

            resized_template = cv2.cvtColor(resized_template, cv2.COLOR_GRAY2BGR)


        overlay = cv2.addWeighted(roi, 0.7, resized_template, 0.3, 0)

        target_image[max_loc[1]:max_loc[1] + template_h, max_loc[0]:max_loc[0] + template_w] = overlay

        best_overlay = target_image


    return best_rotation, best_overlay

###########


def rotate_image_body(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
 
def compute_gradients_body(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x) % (2 * np.pi)
    return magnitude, orientation

def preprocess_image_body(image, magnitude_threshold=50):
    # Ensure grayscale for multi-channel images
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # Already single-channel
 
    magnitude, orientation = compute_gradients_body(gray)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    mask = magnitude > magnitude_threshold  # Filter weak gradients
    return orientation * mask, mask  # Masked orientations
 
def match_template_body(template, target, threshold=0.5):
    template_orientation, template_mask = preprocess_image_body(template)
    target_orientation, target_mask = preprocess_image_body(target)
    
    result = np.zeros((target_mask.shape[0] - template_mask.shape[0] + 1,
                       target_mask.shape[1] - template_mask.shape[1] + 1), dtype=np.float32)
 
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            roi = target_orientation[y:y+template_mask.shape[0], x:x+template_mask.shape[1]]
            roi_mask = target_mask[y:y+template_mask.shape[0], x:x+template_mask.shape[1]]
            if np.sum(roi_mask) == 0:  # Skip blank areas
                continue
            diff = np.abs(template_orientation - roi) * template_mask
            cos_similarity = np.sum(np.cos(diff)) / np.sum(template_mask)
            result[y, x] = cos_similarity
 
    result[result < threshold] = 0
    return result
 
def get_best_match_body(template, target, rotation_degrees, threshold=0.5):
    best_match = None
    best_rotation = None
    best_result = None
 
    for angle in rotation_degrees:
        rotated_template = rotate_image_body(template, angle)
        result = match_template_body(rotated_template, target, threshold=threshold)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
 
        if best_result is None or max_val > best_result:
            best_result = max_val
            best_match = (max_loc, rotated_template)
            best_rotation = angle
 
    if best_match:
        x, y = best_match[0]
        h, w = best_match[1].shape[:2]
        overlay = target.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return best_rotation, overlay
    return None, None
    
    
####
if __name__ == "__main__":

    template_path_head = "tele_head_template.png.npy"
    template_path_body = "tele_upper_body_template.png.npy"

    target_image_path = "../tele_guitars/details/images/fender_am_ultra_tele_mn_cobra_blue.jpg"


    template_head = np.load(template_path_head)
    template_body = np.load(template_path_body)



    target_image = cv2.imread(target_image_path)
    target_image = cv2.rotate(target_image, cv2.ROTATE_90_CLOCKWISE)


    if template_head is None or template_body is None or target_image is None:
        raise FileNotFoundError("Template or target image not found!")


    rotation_degrees = [0, 45, 90, 135, 180, 225, 270, 315]  # You can adjust the rotation degrees as needed


    # Find best match for the first template
    best_rotation_body, final_overlay_body = get_best_match_body(template_body, target_image.copy(), rotation_degrees, threshold=0.7)
    print(f"Best rotation for Upper Body template: {best_rotation_body} degrees")

    best_rotation_head, final_overlay_head = get_best_match_head(template_head, target_image.copy(), rotation_degrees)
    print(f"Best rotation for Head template: {best_rotation_head} degrees")

    if final_overlay_head is not None and final_overlay_body is not None:
        combined_overlay = cv2.addWeighted(final_overlay_head, 0.5, final_overlay_body, 0.5, 0)
        plt.imshow(cv2.cvtColor(combined_overlay, cv2.COLOR_BGR2RGB))
        plt.title("Both Templates Overlaid")
        plt.show()

    else:
        print("No overlay found for one of the templates.")
    
