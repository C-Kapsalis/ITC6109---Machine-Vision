import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Generate an image pyramid
def generate_pyramid(image, scale_factor=0.75, min_size=(100, 100)):
    pyramid = [image]
    while True:
        new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
        if new_size[0] < min_size[0] or new_size[1] < min_size[1]:
            break
        image = cv2.resize(image, new_size)
        pyramid.append(image)
    return pyramid

# Rotate an image by a given angle
def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

# Keypoint Matching using ORB (can use SIFT if needed)
def keypoint_match(test_image, template_image, check_rotation=False, check_scale=False, rotation_step=5, scale_step=0.1):
    # Use ORB for detecting keypoints
    orb = cv2.ORB_create()
    
    # Detect keypoints and descriptors for both images
    kp_test, des_test = orb.detectAndCompute(test_image, None)
    kp_template, des_template = orb.detectAndCompute(template_image, None)

    # If there are no keypoints in either image, return a low match
    if des_test is None or des_template is None:
        return 0, []

    # Use Brute Force Matcher to find best matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_test, des_template)

    return len(matches), matches


def match_multiple_templates(test_image_path, template_folder, output_path, check_rotation=False, rotation_step=5, check_scale=False, scale_step=0.1):
    # Load the test image
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if test_image is None:
        raise ValueError("Test image not found!")

    # Generate pyramids for the test image
    test_pyramid = generate_pyramid(test_image)

    # List to store best matches based on template matching and keypoints
    best_matches = {}  # Template matching best scores
    best_keypoints_matches = {}  # Keypoint matches
    best_rotations = {}  # To store the best rotated template for each template

    # Iterate over all templates in the folder
    template_files = [os.path.join(template_folder, f) for f in os.listdir(template_folder) if f.endswith(('.jpg', '.png'))]
    if not template_files:
        raise ValueError("No templates found in the specified folder.")

    for template_path in template_files:
        # Load the template image
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Template {template_path} could not be loaded. Skipping.")
            continue

        # Generate pyramid for the template
        template_pyramid = generate_pyramid(template)
        
        # Best match score (Template Matching)
        best_score = -np.inf
        best_score_rotation = 0
        best_score_scale = 1.0
        best_rotated_template = template  # Default to original template if no rotation needed
        for test_img in test_pyramid:
            for template_img in template_pyramid:
                if template_img.shape[0] > test_img.shape[0] or template_img.shape[1] > test_img.shape[1]:
                    continue  # Skip if the template is larger than the test image
                
                # Optionally apply rotation and scale for template matching
                if check_rotation:
                    for angle in range(0, 360, rotation_step):
                        rotated_template = rotate_image(template_img, angle)
                        result = cv2.matchTemplate(test_img, rotated_template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(result)
                        if max_val > best_score:
                            best_score = max_val
                            best_score_rotation = angle
                            best_score_scale = 1.0
                            best_rotated_template = rotated_template

                if check_scale:
                    scale = 1.0
                    while scale <= 2.0:
                        scaled_template = cv2.resize(template_img, None, fx=scale, fy=scale)
                        if scaled_template.shape[0] > test_img.shape[0] or scaled_template.shape[1] > test_img.shape[1]:
                            break
                        result = cv2.matchTemplate(test_img, scaled_template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(result)
                        if max_val > best_score:
                            best_score = max_val
                            best_score_rotation = 0
                            best_score_scale = scale
                            best_rotated_template = scaled_template
                        scale += scale_step

        # Keypoint Matching (Best match based on keypoints)
        keypoints_score, _ = keypoint_match(test_image, template, check_rotation, check_scale, rotation_step, scale_step)

        # Store the best results for both template matching and keypoints
        best_matches[template_path] = (best_score, best_score_rotation, best_score_scale)
        best_keypoints_matches[template_path] = keypoints_score
        best_rotations[template_path] = best_rotated_template

    # Now calculate the final best match based on both template matching and keypoints
    print("\nTop Scores for Each Template:")  # For displaying all templates' top scores

    final_best_score = -np.inf
    final_best_template = None
    final_best_rotation = 0
    final_best_scale = 1.0
    final_best_keypoints = 0

    print("\nTop Scores for Each Template:")  # For displaying all templates' top scores
    for template_path, (score, rotation, scale) in best_matches.items():
        
        keypoints_score = best_keypoints_matches[template_path]

        # Here we combine both template matching score and keypoint matching score.
        # We calculate the final score as a weighted sum of both scores.
        final_score = (score + keypoints_score) / 2  # Adjust the weights as necessary

        # Print the top scores for each template
        print(f"\nTemplate: {os.path.basename(template_path)}")
        print(f"  Template Matching Score: {score:.4f}, Rotation: {rotation}°, Scale: {scale:.2f}")
        print(f"  Keypoint Matching Score: {keypoints_score:.4f}")
        print(f"  Final Combined Score (Template + Keypoints): {final_score:.4f}")
        print("-" * 40)

        # Update if this template has the highest combined score
        if final_score > final_best_score:
            final_best_score = final_score
            final_best_template = template_path
            final_best_rotation = rotation
            final_best_scale = scale
            final_best_keypoints = keypoints_score

    # Print the final best result
    print(f"\nBest Final Match for Test Image: {test_image_path}")
    print(f"  Best Template: {os.path.basename(final_best_template)}")
    print(f"  Template Matching: Confidence {best_matches[final_best_template][0]:.4f}, Rotation {360 - final_best_rotation}°, Scale {final_best_scale:.2f}")
    print(f"  Keypoint Matching: Points Matched: {final_best_keypoints}")
    print(f"  Final Combined Score: {final_best_score:.4f}")

    # Display the result with the best matching template and its score
    matched_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    best_template_img = cv2.imread(final_best_template, cv2.IMREAD_GRAYSCALE)
    best_rotated_template_img = rotate_image(best_rotated_template, 360 - final_best_rotation)  # Apply reverse rotation
    
    h, w = best_template_img.shape
    matched_image = cv2.cvtColor(matched_image, cv2.COLOR_GRAY2BGR)
    location = (0, 0)  # Placeholder for location (adjust this as needed)
    cv2.rectangle(matched_image, location, (location[0] + w, location[1] + h), (0, 255, 0), 2)
    
    match_text = f"Best Match: {os.path.basename(final_best_template)}\nConfidence: {best_matches[final_best_template][0]:.4f}, Rotation: {360 - final_best_rotation}°, Scale: {final_best_scale:.2f}"
    cv2.putText(matched_image, match_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    output_path = f"{output_path}_final_best_match_{os.path.basename(final_best_template)}.png"
    cv2.imwrite(output_path, matched_image)

    # Display images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 4, 1)
    plt.title(f"Test Image")
    plt.imshow(matched_image)
    plt.subplot(1, 4, 2)
    plt.title(f"Best Matching Template")
    plt.imshow(best_template_img, cmap="gray")
    plt.subplot(1, 4, 3)
    plt.title(f"Best Rotated Template")
    plt.imshow(best_rotated_template_img, cmap="gray")
    plt.subplot(1, 4, 4)
    plt.title(f"Matched Result")
    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()



# Main function to process all test images
def match_multiple_templates_for_all_tests(test_images_folder, template_folder, output_folder, check_rotation=False, rotation_step=5, check_scale=False, scale_step=0.1):
    test_images = [os.path.join(test_images_folder, f) for f in os.listdir(test_images_folder) if f.endswith(('.jpg', '.png'))]
    if not test_images:
        raise ValueError("No test images found in the specified folder.")

    for test_image_path in test_images:
        output_path = os.path.join(output_folder, os.path.basename(test_image_path))
        print(f"Processing test image: {test_image_path}")
        match_multiple_templates(test_image_path, template_folder, output_path, check_rotation, rotation_step, check_scale, scale_step)

if __name__ == "__main__":
    test_images_folder = "./irregulars/"  # Replace with test images folder path
    template_folder = "./templates/"  # Replace with templates folder path
    output_folder = "./matched_results/"  # Replace with desired output folder path

    match_multiple_templates_for_all_tests(test_images_folder, template_folder, output_folder, check_rotation=True, rotation_step=5, check_scale=True, scale_step=0.1)
