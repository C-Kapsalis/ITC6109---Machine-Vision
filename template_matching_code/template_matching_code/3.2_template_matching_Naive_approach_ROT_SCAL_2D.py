import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def generate_pyramid(image, scale_factor=0.75, min_size=(100, 100)):
    """
    generating an image pyramid, which means we're creating progressively smaller versions
    of the original image. We do this so that we can perform template matching at different scales.
    The 'scale_factor' is the factor by which we reduce the image size, and 'min_size' is the smallest size
    we will allow. This is useful for matching templates that may appear at different scales in the test image.
    """
    pyramid = [image]  # Start with the original image at the top of the pyramid
    while True:
        # resizing the image by the scale factor. Each time, the image gets smaller.
        new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
        if new_size[0] < min_size[0] or new_size[1] < min_size[1]:
            # We stop resizing when the image gets smaller than the minimum size defined by 'min_size'
            break
        image = cv2.resize(image, new_size)  # Resize the image to the new size
        pyramid.append(image)  # Add the resized image to the pyramid list
    return pyramid

def rotate_image(image, angle):
    """
    rotating the image by a given angle. The 'angle' parameter specifies the angle
    in degrees to rotate the image counterclockwise. This is useful for template matching when 
    the object in the template may appear rotated in the test image.
    """
    height, width = image.shape[:2]  # getting the height and width of the image
    center = (width // 2, height // 2)  # calculate the center of the image to rotate around
    # getting the rotation matrix for the given angle
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # rotating the image using the warpAffine function with the rotation matrix
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def match_multiple_templates(test_image_path, template_folder, output_path, check_rotation=False, rotation_step=5, check_scale=False, scale_step=0.1):
    """
    matching a test image with multiple template images. The 'check_rotation' flag lets us 
    rotate the templates to check for matches at different angles, and the 'check_scale' flag lets us 
    scale the templates to check for matches at different sizes.
    
    Parameters:
    - test_image_path: The path to the test image we want to match against.
    - template_folder: The folder containing the template images we want to match with.
    - output_path: The path to save the result image with the best match.
    - check_rotation: If True, we will rotate the templates at different angles to check for a match.
    - rotation_step: The step size (in degrees) for each rotation, only used if 'check_rotation' is True.
    - check_scale: If True, we will scale the templates to different sizes to check for a match.
    - scale_step: The step size for scaling, only used if 'check_scale' is True.
    """
    # First, we are loading the test image in grayscale mode
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if test_image is None:
        raise ValueError("Test image not found!")

    # applying edge detection (Canny) on the test image to extract its edges
    edges = cv2.Canny(test_image, 50, 150)
    # defining a kernel to use for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    # using the kernel to close small gaps in the edges (morphological closing)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # finding the contours (outlines) of objects in the test image
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # selecting the largest contour (assuming the object we want is the largest one)
    largest_contour = max(contours, key=cv2.contourArea)
    # creating a mask that only includes the largest contour
    mask = np.zeros_like(test_image)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    # closing small gaps in the mask using morphological closing again
    filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # generating an image pyramid for the test image. This creates smaller versions of the test image.
    test_pyramid = generate_pyramid(filled_mask)

    # iterating over all the template images in the folder
    template_files = [os.path.join(template_folder, f) for f in os.listdir(template_folder) if f.endswith(('.jpg', '.png'))]
    if not template_files:
        raise ValueError("No templates found in the specified folder.")

    best_match = None  # initializing the best match location
    best_score = -np.inf  # Initialize the best score with negative infinity to ensure any score is better
    best_template = None  # initializing the best template file path
    best_rotation = 0  # initialize the best rotation angle
    best_scale = 1.0  # initialize the best scale factor

    for template_path in template_files:
        # loading each template image in grayscale mode
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Template {template_path} could not be loaded. Skipping.")
            continue

        # generating a pyramid for each template image (scaling it to smaller sizes)
        template_pyramid = generate_pyramid(template)

        for test_img in test_pyramid:  # iterating over all levels of the test image pyramid
            for template_img in template_pyramid:  # iterating over all levels of the template pyramid
                if template_img.shape[0] > test_img.shape[0] or template_img.shape[1] > test_img.shape[1]:
                    # If the template is larger than the test image at this level, we skip it
                    continue

                # If we are checking for rotation
                if check_rotation:
                    for angle in range(0, 360, rotation_step):  # rotating the template by 'rotation_step' degrees
                        # rotating the template using the specified angle
                        M = cv2.getRotationMatrix2D((template_img.shape[1] // 2, template_img.shape[0] // 2), -angle, 1)  # Clockwise rotation
                        rotated_template = cv2.warpAffine(template_img, M, (template_img.shape[1], template_img.shape[0]))

                        if rotated_template.shape[0] > test_img.shape[0] or rotated_template.shape[1] > test_img.shape[1]:
                            # If the rotated template is larger than the test image, we skip it
                            continue

                        # matching the rotated template with the test image using the TM_CCOEFF_NORMED method
                        result = cv2.matchTemplate(test_img, rotated_template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)

                        # updating the best match if the current score is better than the best score so far
                        if max_val > best_score:
                            best_score = max_val
                            best_match = max_loc  # Store the location of the best match
                            best_template = template_path  # Store the template path
                            best_rotation = angle  # Store the rotation angle
                            best_scale = 1.0  # No scaling applied for this rotation

                # If we are checking for scaling
                if check_scale:
                    scale = 1.0
                    while scale <= 2.0:  # Limit scaling up to 2 times the original size
                        # resizing the template image by the scale factor
                        scaled_template = cv2.resize(template_img, None, fx=scale, fy=scale)
                        if scaled_template.shape[0] > test_img.shape[0] or scaled_template.shape[1] > test_img.shape[1]:
                            break  # If the scaled template is larger than the test image, we stop scaling
                        
                        # matching the scaled template with the test image
                        result = cv2.matchTemplate(test_img, scaled_template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)

                        # Again, we update the best match if the current score is better than the best score so far
                        if max_val > best_score:
                            best_score = max_val
                            best_match = max_loc
                            best_template = template_path
                            best_rotation = 0  # No rotation applied for this scale
                            best_scale = scale  # Store the scaling factor

                        scale += scale_step  # Increment the scale factor

    # drawing a rectangle around the best match in the test image
    if best_match is not None:
        test_image_color = cv2.imread(test_image_path)  # Load the original test image in color
        h, w = template.shape[:2]
        top_left = best_match
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # Draw a red rectangle around the matched area
        cv2.rectangle(test_image_color, top_left, bottom_right, (0, 0, 255), 2)

        # saving the result image to the specified output path
        cv2.imwrite(output_path, test_image_color)
        print(f"Best match found at location {best_match} with score {best_score}")
        print(f"Template: {best_template}, Rotation: {best_rotation}, Scale: {best_scale}")
    else:
        print("No match found.")


def match_multiple_templates_for_all_tests(test_images_folder, template_folder, output_folder, check_rotation=False, rotation_step=5, check_scale=False, scale_step=0.1):
    """
    Iterate over all test images and match them with multiple templates.
    """
    test_images = [os.path.join(test_images_folder, f) for f in os.listdir(test_images_folder) if f.endswith(('.jpg', '.png'))]

    if not test_images:
        raise ValueError("No test images found in the specified folder.")

    for test_image_path in test_images:
        # Prepare output file path
        output_path = os.path.join(output_folder, os.path.basename(test_image_path))

        print(f"Processing test image: {test_image_path}")
        match_multiple_templates(test_image_path, template_folder, output_path, check_rotation, rotation_step, check_scale, scale_step)

def main():
    test_images_folder = "./irregulars/"  # ORRRRR with "./test_images/"  Replace with test images folder path
    template_folder = "./templates/"  
    output_folder = "./matched_results/"  

    # matching process for all test images
    match_multiple_templates_for_all_tests(test_images_folder, template_folder, output_folder, check_rotation=True, rotation_step=5, check_scale=True, scale_step=0.1)

if __name__ == "__main__":
    main()
