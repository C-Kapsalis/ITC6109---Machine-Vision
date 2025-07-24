import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

def generate_pyramid(image, scale_factor=0.75, min_size=(100, 100)):
    pyramid = [image]
    while True:
        new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
        if new_size[0] < min_size[0] or new_size[1] < min_size[1]:
            break
        image = cv2.resize(image, new_size)
        pyramid.append(image)
    return pyramid

def rotate_image_2d(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def rotate_image_3d(image, angle_x, angle_y):
    # Assuming a simple 3D rotation matrix in the x and y axes
    # OpenCV does not support 3D rotations directly, so we use approximation
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    rotation_matrix_x = cv2.getRotationMatrix2D(center, angle_x, 1.0)
    rotated_x = cv2.warpAffine(image, rotation_matrix_x, (width, height))
    
    rotation_matrix_y = cv2.getRotationMatrix2D(center, angle_y, 1.0)
    rotated_y = cv2.warpAffine(rotated_x, rotation_matrix_y, (width, height))
    
    return rotated_y

def match_multiple_templates(test_image_path, template_folder, output_path, apply_2d_rotation=False, apply_3d_rotation=False, apply_scaling=False, rotation_step=5, scale_step=0.1):
    # Load the test image
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if test_image is None:
        raise ValueError("Test image not found!")

    # Apply the pre-processing steps to the test image
    edges = cv2.Canny(test_image, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(test_image)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Generate pyramids for the test image
    test_pyramid = generate_pyramid(filled_mask)

    # Iterate over all templates in the folder
    template_files = [os.path.join(template_folder, f) for f in os.listdir(template_folder) if f.endswith(('.jpg', '.png'))]
    if not template_files:
        raise ValueError("No templates found in the specified folder.")

    best_match = None
    best_score = -np.inf  # Negative infinity to ensure any match score is better
    best_template = None
    best_rotation = 0
    best_scale = 1.0

    for template_path in template_files:
        # Load the template image
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Template {template_path} could not be loaded. Skipping.")
            continue

        # Generate pyramid for the template
        template_pyramid = generate_pyramid(template)

        for test_img in test_pyramid:
            for template_img in template_pyramid:
                if template_img.shape[0] > test_img.shape[0] or template_img.shape[1] > test_img.shape[1]:
                    continue  # Skip matching if the template is larger than the test image at this pyramid level

                # Optionally apply rotation in 2D (z-axis) and check best match
                if apply_2d_rotation:
                    for angle in range(0, 360, rotation_step):  # Iterate over rotations in 5-degree steps
                        rotated_template = rotate_image_2d(template_img, angle)

                        if rotated_template.shape[0] > test_img.shape[0] or rotated_template.shape[1] > test_img.shape[1]:
                            continue  # Skip if rotated template is larger than the test image

                        result = cv2.matchTemplate(test_img, rotated_template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)

                        # Update best match if this is the best score
                        if max_val > best_score:
                            best_score = max_val
                            best_match = max_loc
                            best_template = template_path
                            best_rotation = angle  # Store the best rotation angle
                            best_scale = 1.0  # Keep default scale (no scaling for this rotation)

                # Optionally apply rotation in 3D (x and y axes) and check best match
                if apply_3d_rotation:
                    for angle_x in [0, 45, 90, 180]:
                        for angle_y in [0, 45, 90, 180]:
                            rotated_template = rotate_image_3d(template_img, angle_x, angle_y)

                            if rotated_template.shape[0] > test_img.shape[0] or rotated_template.shape[1] > test_img.shape[1]:
                                continue  # Skip if rotated template is larger than the test image

                            result = cv2.matchTemplate(test_img, rotated_template, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, max_loc = cv2.minMaxLoc(result)

                            # Update best match if this is the best score
                            if max_val > best_score:
                                best_score = max_val
                                best_match = max_loc
                                best_template = template_path
                                best_rotation = (angle_x, angle_y)  # Store the best 3D rotation angles
                                best_scale = 1.0  # Keep default scale (no scaling for this rotation)

                # Optionally apply scaling to the template and check best match
                if apply_scaling:
                    scale = 1.0
                    while scale <= 2.0:  # Limit scale range
                        scaled_template = cv2.resize(template_img, None, fx=scale, fy=scale)
                        if scaled_template.shape[0] > test_img.shape[0] or scaled_template.shape[1] > test_img.shape[1]:
                            break  # Skip if scaled template is larger than the test image
                        
                        result = cv2.matchTemplate(test_img, scaled_template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)

                        # Update best match if this is the best score
                        if max_val > best_score:
                            best_score = max_val
                            best_match = max_loc
                            best_template = template_path
                            best_rotation = 0  # No rotation applied
                            best_scale = scale  # Store the best scale

                        scale += scale_step  # Increase scale

    # Print the best match information
    if best_match is not None:
        best_template_img = cv2.imread(best_template, cv2.IMREAD_GRAYSCALE)
        h, w = best_template_img.shape
        matched_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(matched_image, best_match, (best_match[0] + w, best_match[1] + h), (0, 255, 0), 2)

        match_text = f"Best Match: {os.path.basename(best_template)} with confidence {best_score:.4f}, rotation {best_rotation}, scale {best_scale:.2f}"
        cv2.putText(matched_image, match_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(output_path, matched_image)

        # Show the images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Test Image")
        plt.imshow(test_image, cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("Best Matching Template")
        plt.imshow(best_template_img, cmap="gray")
        plt.subplot(1, 3, 3)
        plt.title("Matched Result")
        plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        plt.show()

        print(f"Best matching template: {os.path.basename(best_template)} with confidence {best_score:.4f}, rotation {best_rotation}, scale {best_scale:.2f}")
    else:
        print("No suitable match found.")

def match_multiple_templates_for_all_tests(test_images_folder, template_folder, output_folder, apply_2d_rotation=False, apply_3d_rotation=False, apply_scaling=False):
    test_images = [os.path.join(test_images_folder, f) for f in os.listdir(test_images_folder) if f.endswith(('.jpg', '.png'))]

    if not test_images:
        raise ValueError("No test images found in the specified folder.")

    for test_image_path in test_images:
        output_path = os.path.join(output_folder, os.path.basename(test_image_path))

        print(f"Processing test image: {test_image_path}")
        match_multiple_templates(test_image_path, template_folder, output_path, apply_2d_rotation, apply_3d_rotation, apply_scaling)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply_2d_rotation', action='store_true', help="Apply rotation in 2D (z-axis)")
    parser.add_argument('--apply_3d_rotation', action='store_true', help="Apply rotation in 3D (x and y axes)")
    parser.add_argument('--apply_scaling', action='store_true', help="Apply scaling")
    args = parser.parse_args()

    test_images_folder = "./test_images/"
    template_folder = "./templates/"
    output_folder = "./matched_results/"

    match_multiple_templates_for_all_tests(test_images_folder, template_folder, output_folder, args.apply_2d_rotation, args.apply_3d_rotation, args.apply_scaling)

if __name__ == "__main__":
    main()
