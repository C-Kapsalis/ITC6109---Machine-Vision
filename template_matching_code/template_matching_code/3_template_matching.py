import os
import numpy as np
import matplotlib.pyplot as plt 
import cv2



def match_multiple_templates(test_image_path, template_folder, output_path):
    """
    Function that matches every one of our 'generalized' guitar body templates against every guiven 
    # image (called a 'test' image) to determine the best match.

    Input:
        test_image_path (str): Path to the test image.
        template_folder (str): Path to the folder containing guitar templates.
        output_path (str): Path to save the best match visualization.

    Output: 
        A visualization showing the best matching mask, together with the confidence of the test image matching 
        every single one of our considered 'generalized' templates.
    """
    # Load the test image
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

    if test_image is None:
        raise ValueError("Test image not found!")

    # Apply to the test image the pre-processing we did in the exploratory phase
    edges = cv2.Canny(test_image, 50, 150)
    ###TODO: Also add the further preprocessing steps we've implemented
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(test_image)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    
    ###
    
    # Iterate over all templates in the folder
    template_files = [os.path.join(template_folder, f) for f in os.listdir(template_folder) if f.endswith(('.jpg', '.png'))]
    print(template_files)



    if not template_files:
        raise ValueError("No templates found in the specified folder.")

    best_match = None
    best_score = -np.inf  # Negative infinity to ensure any match score is better
    best_template = None

    for template_path in template_files:
        # Load each template
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

        if template is None:
            print(f"Template {template_path} could not be loaded. Skipping.")
            continue

        # Perform template matching
        #result = cv2.matchTemplate(edges, template, cv2.TM_CCOEFF_NORMED)
        result = cv2.matchTemplate(filled_mask, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Update the best match if this one is better
        if max_val > best_score:
            best_score = max_val
            best_match = max_loc
            best_template = template_path

    # Visualize the best match
    if best_match is not None:
        h, w = cv2.imread(best_template, cv2.IMREAD_GRAYSCALE).shape
        matched_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(matched_image, best_match, (best_match[0] + w, best_match[1] + h), (0, 255, 0), 2)
        cv2.putText(matched_image, f"Best Match: {os.path.basename(best_template)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save and display the result
        cv2.imwrite(output_path, matched_image)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Test Image")
        plt.imshow(test_image, cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("Best Matching Template")
        plt.imshow(cv2.imread(best_template, cv2.IMREAD_GRAYSCALE), cmap="gray")
        plt.subplot(1, 3, 3)
        plt.title("Matched Result")
        plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        plt.show()

        print(f"Best matching template: {os.path.basename(best_template)} with confidence {best_score:.4f}")
    else:
        print("No suitable match found.")




def match_multiple_templates_for_all_tests(test_images_folder, template_folder, output_folder):
    """
    Function that implements template matching on all images of a particular folder (which we a-priori know that they fall under) 
    a particular body type. 

    Input:
        test_images_folder (str): Path to the folder containing test images.
        template_folder (str): Path to the folder containing guitar templates.
        output_folder (str): Path to save the matched result images.
    Output:
        For each of these images, the function produces a visual showing the best matching 'generalized' template/mask.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all test images in the folder
    test_images = [f for f in os.listdir(test_images_folder) if f.endswith(('.jpg', '.png'))]
    
    if not test_images:
        raise ValueError("No test images found in the specified folder.")
    
    for test_image_name in test_images:
        test_image_path = os.path.join(test_images_folder, test_image_name)
        
        # Use the test image name to create an output path
        output_image_name = f"matched_{test_image_name}"
        output_path = os.path.join(output_folder, output_image_name)

        print(f"Processing {test_image_name}...")

        # Match the test image against all templates
        match_multiple_templates(test_image_path, template_folder, output_path)

    print("All test images processed.")


if __name__ == "__main__":
    # Paths to folders
    test_images_folder = "../test_images/"  # Replace with test images folder path
    template_folder = "../templates/"  # Replace with templates folder path
    output_folder = "../matched_results/"  # Replace with desired output folder path

    # Run the function for all test images
    match_multiple_templates_for_all_tests(test_images_folder, template_folder, output_folder)

