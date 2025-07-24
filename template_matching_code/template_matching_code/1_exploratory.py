### This script takes care of implementing pre-processing on the images of the various types
# and creating 'masks' that essentially act as a map that captures the outline and body of each guitar considered.
# This was made very easy for us due to the high quality and consistency of the raw images we used. As a result, we have create
# masks of very high quality that can be used for template matching under (very) controlled conditions.

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
    most_frequent_bin = np.argmax(hist)  # retrieves the index of the corresponding value 
    most_frequent_value = bins[most_frequent_bin]  # retrieves the actual value

    if mode == 'bin':
        return int(most_frequent_value)
    elif mode == 'message':
        print(f'The most frequent value of the histogram is {int(most_frequent_value)}, with a frequency of {int(hist[most_frequent_bin])}')
    elif mode == 'plot':
        print(f'The most frequent value of the histogram is {int(most_frequent_value)}, with a frequency of {int(hist[most_frequent_bin])}')

        # the gray scales are condences on the two edges, so a 127 threshold is ok
        # It would be better though to exclude those edges because there is info 
        # being missed in the initial plot 
        plt.plot(hist, color='r')
        if xlim_range:
            plt.xlim(xlim_range)  # else keep the default that will be 
        if ylim_range:
            plt.ylim(ylim_range)  # else keep the default that will reach the maximum value of the histogram

        plt.show()



def semi_thresholding(img_obj, bin_range):
    # img_obj is considered to already come as a grayscale/monochromatic image
    w_img_obj = img_obj.copy()  # we never modify the original image file in any step we implement
    for bin in bin_range:
        w_img_obj[w_img_obj == bin] = 255  

    return w_img_obj




def guitar_mask_creation(path_to_image, path_to_mask_output):
    """
        This is our main and essential/fundamental script for the 'simple' approach of template matching.
            Input: Paths to the guitar image considered on each iteration and the desired storage location of the output mask image
            Output: A mask capturing the (approximately) the totality of the guitar's surface, saved at our desired location.
    """
 

    ## Importing the image with color
    img_colored = cv2.imread(image_path)
    
    ## Turning the image to grayscale
    img = cv2.cvtColor(img_colored, cv2.COLOR_BGR2GRAY)
 
    ## see histogram - do semi-thresholding to make black background white
    # this we will make out by trying it on a multitude of images (4-5) for each type
    bins_no = 256
    intended_range = [0, 255]
    plot_histogram(img_obj=img, bins_no=bins_no, intended_range=intended_range)




def vertical_line_on_image(col_img_obj, col1, col2=None):
    if col2:
        cols = [col1, col2]
    else: 
        cols = [col1]

    for col in cols:
        for row in range(col_img_obj.shape[0]):
            col_img_obj[row, col] = [0, 0, 255]

    cv2.imshow('img with line(s)', col_img_obj)
    cv2.waitKey(0)
    cv2.destroyWindow('img with line(s)')



if __name__ == "__main__":

    # Hand Picking Particular images
    w_bg_guitar_file_abs_path = '../st_guitars/details/images/schecter_c_1_fr_s_sls_evil_twin_sbk.jpg'
    b_bg_guitar_file_abs_path = '../st_guitars/details/images/fender_player_plus_strat_mn_3csb.jpg'
    
    # Loading the images into cv2 in color mode (for visualization purposes)
    w_bg_img = cv2.imread(w_bg_guitar_file_abs_path)
    b_bg_img = cv2.imread(b_bg_guitar_file_abs_path)

    # Try various lines to capture only the 'central' part of the img, which is where they are contained.
    # I do this based on the black-background image, which gives more hints as to where these boundaries lie. 
    #vertical_line_on_image(b_bg_img, 195, 405)

    # After a trial & error process, the col1=195 and col2=405 boundaries seem most appropriate for the images.
    # We validate this based on testing on the white-background image as well.
    #vertical_line_on_image(w_bg_img, 195, 405)
    # These boundaries seem fine on this image as well, so, trusting the consistency of Thomann's in their product 
    # sample photoshooting processes, we will treat these boundaries as universal for all raw images retrieved from their website.
    col1 = 195
    col2 = 405


    ## From now on, we only work with the luminance channel of the image, to avoid the hassle of dealing with multiple color channels. 

    # Drawing the histogram of the black-background image
    # CAUTION: We only draw for the 'central' part of it, based on the previous boundaries. 
    w_bg_img_lum = cv2.cvtColor(w_bg_img, cv2.COLOR_BGR2GRAY)
    plot_histogram(img_obj=w_bg_img_lum[:, col1:col2], bins_no=256, intended_range=[0,256], mode='plot')

    # We do the same on the white-background image
    b_bg_img_lum = cv2.cvtColor(b_bg_img, cv2.COLOR_BGR2GRAY)
    plot_histogram(img_obj=b_bg_img_lum[:, col1:col2], bins_no=256, intended_range=[0,256], mode='plot')
     
    # Both images have a clear peak in a specific intensity value; 255 for the white-background image, and 33 for the black-background one 
    
    ## So, trusting again the consistency of Thomann's image photoshooting processes, for every incoming raw image, 
    # we will calculate their most frequent value, and if it is +-7 (ad-hoc number) 33 or 255, we will consider them as 
    # white-background or black-background images. 
    # We tried the following in the latter case: to implement semi-thresholding to throw away the background
    # (aka pixels with intensity value exactly the same as the background's hue), so that they represent the exact same conditions as
    # the other images and we can handle all with more uniform operations.
    # However, as the following example shows, it gave off a very noisy result that would certainly distort our next operations' results, 
    # so we abandoned it and followed a more refined approach with blurring and edge detection + connected component analysis + closing to find the guitar masks.
    #random_b_bg_guitar_file_abs_path = '/Users/chkapsalis/project_little_dragon/st_guitars/details/images/evh_stripe_black.jpg'#fender_michael_landau_coma_strat.jpg'
    random_b_bg_guitar_file_abs_path = '../st_guitars/details/images/jackson_js22_dinky_blk_ah.jpg'#evh_stripe_black.jpg'#fender_michael_landau_coma_strat.jpg'
    random_img = cv2.imread(random_b_bg_guitar_file_abs_path, cv2.IMREAD_GRAYSCALE)
    bin_value = plot_histogram(img_obj=random_img[:, col1:col2], bins_no=256, intended_range=[0,256], mode='bin')
    if bin_value in range(33-5, 33+5+1):
        ### dark background image 
        print('Dark-background image detected.')
    
    # Hard semi-thresholding example
    #    ## Implementing semi-thresholding to throw away the background (replacing with white) and keep everything else - !!! in the report
    #    # I need to mention the formula of doing this !!! 
    #    semi_thr_random_img = semi_thresholding(random_img, [bin_value])
    #    cv2.imshow('Image after semi-thresholding', semi_thr_random_img)
    #    cv2.waitKey(0)
    #    cv2.destroyWindow('Image after semi-thresholding')

    
    ## Image blurring - this is very important as guitars have various and rather diverse features that add a lot of detail on them. 
    # We want to blur as much as possible features like the guitar pickups, strings, volume/tone knobs and switces, so as to only focus 
    # on the literal outline of what makes a guitar. This we expect to benefit both our 'simpler' (aka based on template matching)
    # and more 'advanced' (aka based on a Haar Classifier for guitar object detection) approaches.
    # However, doing this now would also probably alter the outer edges of the guitar, especially in cases where a guitar has parts
    # close to the background (e.g. its head, fretboard, and body edges) that are darker against a dark background or the exact opposite
    # (light against a light background).
    
    # So we go on with implementing the Canny Edge Detector as is.
    # Since the background is very consistent, we can experiment with very low values for the thresholds of what constitutes an edge.
        edges = cv2.Canny(random_img, 20, 100)
        cv2.imshow('canny edge result on a dark background img', edges)
        cv2.waitKey(0)
        cv2.destroyWindow('canny edge result on a dark background img')

    # We can see that the default values for the main and the Hysterisis threshold give off very good results for dark-background images (as well as 
    # light-background ones as we see later on)
    # However, in cases where there is a dark background, there is a potential issue arising from the fact that the universal 'col1' and 'col2' boundary
    # values we came up before do not actually apply on all images, so we might get some where there are some dark vertical lines still left off. 
    # To capture these objects and drop them (i.e. turn to white), we will implement connected component analysis, find the components that have a length of 
    # about 600 pixels (since this is the height of all of our raw images from thomann)

    # This would lead us to only the guitar being left in the image. 
    # But we could also achieve this immediately by managing to capture the largest connected component on the img, which should be the guitar.
    # However, due to the interactions of the guitar parts' colors with the background, we could get a guitar fragmented in multiple different parts.
    # Thus, we have to implement rounds of dilation in order to fill-in gaps (and then the same number of erosion pairs to return all objects back to their initial sizes).
    # So before connected components analysis, we will implenent the morphological operation of 'closing' on the result of the canny edge detection, which will try and 
    # and 'fill in' blanks among the edges. We can use a rather large kernel for this, since the actual guitar is distanced far enough for the potential remaining
    # vertical lines. For the very same reason, we will use a cross (to avoid square-like alterations at the edges of the guitar shape) 10x10 kernel. 

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('closed_edges_dark', closed_edges)
        cv2.waitKey(0)
        cv2.destroyWindow('closed_edges_dark')

    ## Now we can implement connected component analysis and keep only the largest contour, which will now (most probably) be constisted of the guitar body outline. 
    # A contour is the 'boundary' of a component on an image. We do not want any 'child countours' (inside contours lying on the inside of a larger outer identified object).
    # So we use the cv2.RETR_EXTERNAL parameter. We opt for the cv2.CHAIN_APPROX_SIMPLE value for the contour approximation method, as this 
    # is more computationally efficient. 
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)

    ## Drawing the largest contour - I will create a new image (in the form of a numpy array) to cast the contour on it painted in total white against a total black background
        mask = np.zeros_like(random_img)
    # I want to completely fill in this contour with white to act as a map to the actual guitar's pixels, without bothering with any of the more specific 
    # parts of the guitar (e.g. pickups, knobs, etc)
        cv2.imshow('pre-filled mask', mask)
        cv2.waitKey(0)
        cv2.destroyWindow('pre-filled mask')
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        cv2.imshow('filled mask', mask)
        cv2.waitKey(0)
        cv2.destroyWindow('filled mask')
       
    # Just to make sure, we implement the same 'closure' morphological operation to be certain that there are no black spots inside of the guitar body
        filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # we use the same kernel as before 


    elif bin_value in range(250, 256):
        ### light background image 
        print('Light background image detected.')
        edges = cv2.Canny(random_img, 50, 150)
        #cv2.imshow('canny edge result on a light background img', edges)
        #cv2.waitKey(0)
        #cv2.destroyWindow('canny edge result on a light background img')

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('closed_edges_dark', closed_edges)
        cv2.waitKey(0)
        cv2.destroyWindow('closed_edges_dark')

        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)

        mask = np.zeros_like(random_img)

        cv2.imshow('pre-filled mask', mask)
        cv2.waitKey(0)
        cv2.destroyWindow('pre-filled mask')

        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        cv2.imshow('filled mask', mask)
        cv2.waitKey(0)
        cv2.destroyWindow('filled mask')
       
    # Just to make sure, we implement the same 'closure' morphological operation to be certain that there are no black spots inside of the guitar body
        filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # we use the same kernel as before 

    
    
    	


