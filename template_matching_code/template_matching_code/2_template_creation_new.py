### This script takes care of implementing pre-processing on the images of the various types
# and creating 'masks' that essentially act as a map that captures the outline and body of each guitar considered.
# This was made very easy for us due to the high quality and consistency of the raw images we used. As a result, we have created
# masks of very high quality that can be used for template matching under (very) controlled conditions.  


### Importing Libraries and Dependencies ### 
import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt




def plot_histogram(img_obj, bins_no, intended_range, xlim_range=None, ylim_range=None, mode='bin'):
    """
    Function used for exploratory reasons to build the histogram of intensity values in an image's luninance channel.

        Input: raw image object (expects it in grayscale to have to work with only a single color channel), 
            number of bins and the x-axis range we want to cover, x- and y- axis ranges we want to contrain our visual 
            of the histogram into (for presentational - exploratory purposes), and a desired mode of execution of the script. 
            # There are 3 possible/acceptable values for this parameter: 'bin', 'message', and 'plot'.
    
        Output: The 'bin' option returns an integer representing the bin which is linked to the highest frequency in the histogram. 
            The 'message' option prints out a message containing the previous and the actual value of the frequency. 
            The 'plot' option prints out the said message and also prints the relevant histogram plot visual. 
    """


	hist, bins = np.histogram(img_obj.ravel(), bins=bins_no, range=intended_range)
	cdf = hist.cumsum()
	cdf_normal = cdf * hist.max() / cdf.max()
 
	most_frequent_bin = np.argmax(hist)  # retrieves the index of the corresponding value
	most_frequent_value = bins[most_frequent_bin]  # retrieves the actual value
 
	if mode == 'bin':
		return int(most_frequent_value)
	elif mode == 'message':
		print(f'The most frequent value of the histogram is {int(most_frequent_value)}, with a frequency of {int(hist[most_frequent_bin])}.')
	elif mode == 'plot':
		print(f'The most frequent value of the histogram is {int(most_frequent_value)}, with a frequency of {int(hist[most_frequent_bin])}.')

		plt.plot(hist, color='r')	
		
		if xlim_range:
			plt.xlim(xlim_range)  # else keep the default that will be
	
		if ylim_range:
			plt.ylim(ylim_range)  # else keep the default that will reach the maximum value of the histogram

		plt.show()




def light_back_mask_creation(img_obj):
    """
    Function operating on 'light background' electric guitar images retrieved from Thomann. The processing being implemented on these images has been decided on 
    by our exploratory analysis of images that led us to the conclusion that a uniform processing is enough to build a 'mask' of 
    the instrument depicted.

        Input: image object - we expect this image to have previously been loaded in grayscale so there is only a single color
            channel to work with, the luminance channel. 
                
        Output: a 'mask' which comes in the form of a binary image, where the white cells act as a map to capture the depicted instrument
            in as much detail as possible. 
    """


    print('Light bacground image detected')

    edges = cv2.Canny(img_obj, 20, 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(img_obj)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return filled_mask




def dark_back_mask_creation(img_obj):
   """
    This function has operates in the same manner and with the same goal as the previous one, but it is applied on sample photos of guitars taken from thomann.de
    that happen to have a 'dark background'.
   """

    print('Dark bacground image detected')

    edges = cv2.Canny(img_obj, 20, 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(img_obj)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return filled_mask




def guitar_mask_creation(path_to_image, path_to_mask_output):
    """
    This is our main and essential/fundamental script for the 'simple' approach of template matching. 

    Input: Paths to the guitar image considered on each iteration and the desired storage location 
        of the output mask image object. 
        
    Output: A mask capturing the (approximately) the totality of the guitar's surface, saved at our desired location. 
        The preprocessing leading to the creation of this mask is implemented based on the estimated 'type' of background of 
        each input image.
    """
    
    ## Importing the image with color 
    img_colored = cv2.imread(path_to_image)
    ## Turning the image to grayscale 
    img = cv2.cvtColor(img_colored, cv2.COLOR_BGR2GRAY)

	## see histogram - do semi-thresholding to make black background white 
	# this we will make out by trying it on a multitude of images (4-5) for each type
    bin_value = plot_histogram(img_obj=img[:, 195:405], bins_no=256, intended_range=[0,256], mode='bin')
    if bin_value in range(33-5, 33+5+1):
        return (True, dark_back_mask_creation(img))
    elif bin_value in range(250, 255+1):
        return (True, light_back_mask_creation(img))
    else:
        # we will actually ignore these cases - we have more than enough cases of guitar stock photos falling under the 
        # other two 'standard' cases to create our templates
        print(f'Histogram case we have not found detected @ {path_to_image}.')
        return (False, None)





# After creating sample masks for a large number of guitars, but also based on our experience,
# we understood that even though there might be slight differences in the exact guitar shapes and dimensions from
# brand to brand and even between a particular brand's different product lines, but not that significant that would
# render it irrelevant to have 'the' template for each particular guitar type. In other words, even with this variance 
# considered, we saw that the resulting templates were significantly distinct, so much that even a 'simple' template
# matching algorithm should be able to handle successfully on most 'simple' cases.
# Thus, we went on by creating 'generalized' templates for each guitar body type by 'averaging' the intensity values
# of every mask we created. 

def create_generalized_template(image_folder, output_path, gtype):
    """
    Create a generalized binary template for a particular guitar body type using multiple preprocessed masks of the corresponding category.
    
    Input:
        image_folder (str): Path to the folder containing Telecaster images.
        output_path (str): Path to save the resulting generalized template.
        gtype (str): The (known a-priori through the crawler algorithm that retrieved it)
    Output: 
        A single binary mask image (in png format) we can use to match against guitar objects of the relevant scale & size.
    """
    # List all image files in the folder
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

    if not image_files:
        raise ValueError("No images found in the specified folder.")

    # Load the first image to initialize the stack
    first_image = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    height, width = first_image.shape
    stack = np.zeros((height, width), dtype=np.float32)

    # Process each image, generate a mask, and add to the stack
    for image_path in image_files:
        temp = guitar_mask_creation(image_path, image_path.replace(".jpg", "_mask.jpg"))  # Save intermediate masks
        if temp[0] == True:
            mask = temp[1]
            stack += mask / 255.0  # Normalize to [0, 1] before stacking
        else:
            continue  # ignore cases that do not fall under the dominant ones on thomann.de

    # Average the stack
    avg_mask = stack / len(image_files)

    # Apply threshold to create a binary template
    _, binary_template = cv2.threshold(avg_mask, 0.17, 255, cv2.THRESH_BINARY)  # our threshold is expressed as 
    # a decimal number in [0,1] since we have normalized !!! 

    # Save the generalized template
    cv2.imwrite(output_path, binary_template.astype(np.uint8))

    # Display the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Averaged Mask")
    plt.imshow(avg_mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title(f"Generalized {gtype} Template")
    plt.imshow(binary_template, cmap="gray")
    plt.tight_layout()
    plt.show()


# Actual Usage
if __name__ == "__main__":
    # Folder containing Telecaster images
    for guit_type in ['tele','single_cut','double_cut','jazz','st']:
        image_folder = f"../{guit_type}_guitars/details/img_sample/"  
        output_path = f"../{guit_type}_guitars/details/{guit_type}_temp.jpg" 

        create_generalized_template(image_folder, output_path, guit_type)

