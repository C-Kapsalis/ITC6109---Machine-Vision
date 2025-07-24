import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def generate_pyramid(image, scale_factor=0.75, min_size=(100, 100)):
    """
    Generate an image pyramid by resizing the image to progressively smaller sizes.
    Stops when the image reaches the minimum size.
    """
    pyramid = [image]
    while True:
        # Resize the image by the scale factor
        new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
        if new_size[0] < min_size[0] or new_size[1] < min_size[1]:
            break
        image = cv2.resize(image, new_size)
        pyramid.append(image)
    return pyramid

def rotate_image(image, angle):
    """
    Rotate an image by the given angle (counterclockwise).
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def match_multiple_templates(test_image_path, template_folder, output_path, check_rotation=False, rotation_step=5, check_scale=False, scale_step=0.1, keypoint_threshold=0.06):
    """
    Match a test image against multiple templates and determine the best match, with optional rotation and scale checking.
    If the difference between the top confidence scores is less than `keypoint_threshold`, perform keypoint matching.
    """
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

    best_matches = {}  # Dictionary to store best match for each template

    for template_path in template_files:
        # Load the template image
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Template {template_path} could not be loaded. Skipping.")
            continue

        # Generate pyramid for the template
        template_pyramid = generate_pyramid(template)
        best_score = -np.inf  # Initial best score is negative infinity
        best_match_info = None  # To store match details for best score

        for test_img in test_pyramid:
            for template_img in template_pyramid:
                if template_img.shape[0] > test_img.shape[0] or template_img.shape[1] > test_img.shape[1]:
                    continue  # Skip matching if the template is larger than the test image at this pyramid level

                # Optionally apply rotation to the template and check best match
                if check_rotation:
                    for angle in range(0, 360, rotation_step):  # Iterate over rotations
                        # Use OpenCV to rotate the template
                        M = cv2.getRotationMatrix2D((template_img.shape[1] // 2, template_img.shape[0] // 2), -angle, 1)  # Negative angle for clockwise rotation
                        rotated_template = cv2.warpAffine(template_img, M, (template_img.shape[1], template_img.shape[0]))

                        if rotated_template.shape[0] > test_img.shape[0] or rotated_template.shape[1] > test_img.shape[1]:
                            continue  # Skip if rotated template is larger than the test image

                        result = cv2.matchTemplate(test_img, rotated_template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)

                        # Update best match if we find a better score
                        if max_val > best_score:
                            best_score = max_val
                            best_match_info = (max_val, max_loc, template_path, angle, 1.0)  # Store match info

                # Optionally apply scaling to the template and check best match
                if check_scale:
                    scale = 1.0
                    while scale <= 2.0:  # Limit scale range
                        scaled_template = cv2.resize(template_img, None, fx=scale, fy=scale)
                        if scaled_template.shape[0] > test_img.shape[0] or scaled_template.shape[1] > test_img.shape[1]:
                            break  # Skip if scaled template is larger than the test image
                        
                        result = cv2.matchTemplate(test_img, scaled_template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)

                        # Update best match if we find a better score
                        if max_val > best_score:
                            best_score = max_val
                            best_match_info = (max_val, max_loc, template_path, 0, scale)  # Store match info

                        scale += scale_step  # Increase scale

        # After checking all levels, store the best match for the template
        if best_match_info:
            best_matches[template_path] = best_match_info

    # Now print the top match for each template and check if the score difference is less than keypoint_threshold
    previous_score = None
    for template, match_info in best_matches.items():
        score, location, template_path, rotation, scale = match_info
        print(f"Best match for {os.path.basename(template_path)}: Confidence {score:.4f}, Rotation {rotation}°, Scale {scale:.2f}")

        # If the difference in score is less than threshold, perform keypoint matching
        if previous_score is not None and abs(previous_score - score) < keypoint_threshold:
            print("Score difference is less than threshold, performing keypoint matching.")

            # Keypoint detection using ORB
            orb = cv2.ORB_create()
            test_keypoints, test_descriptors = orb.detectAndCompute(test_image, None)
            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            template_keypoints, template_descriptors = orb.detectAndCompute(template_img, None)

            # Use BFMatcher to find the best matches
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(test_descriptors, template_descriptors)

            # Sort matches by distance
            matches = sorted(matches, key = lambda x:x.distance)

            # Draw the matches on the images
            match_img = cv2.drawMatches(test_image, test_keypoints, template_img, template_keypoints, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(match_img)
            plt.title(f"Keypoint Matches for {os.path.basename(template_path)}")
            plt.show()

        previous_score = score

        # Visualize the best match result
        best_template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        h, w = best_template_img.shape
        matched_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(matched_image, location, (location[0] + w, location[1] + h), (0, 255, 0), 2)

        match_text = f"Best Match: {os.path.basename(template_path)}\nConfidence: {score:.4f}, Rotation: {rotation}°, Scale: {scale:.2f}"
        cv2.putText(matched_image, match_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        output_path = f"{output_path}_best_match_{os.path.basename(template_path)}.png"
        cv2.imwrite(output_path, matched_image)

        # Show the images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title(f"Test Image")
        plt.imshow(test_image, cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"Best Matching Template")
        plt.imshow(best_template_img, cmap="gray")
        plt.subplot(1, 3, 3)
        plt.title(f"Matched Result")
        plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        plt.show()


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
    template_folder = "./templates/"  # Replace with templates folder path
    output_folder = "./matched_results/"  # Replace with desired output folder path

    # Run the matching process for all test images
    match_multiple_templates_for_all_tests(test_images_folder, template_folder, output_folder, check_rotation=True, rotation_step=5, check_scale=True, scale_step=0.1)

if __name__ == "__main__":
    main()
