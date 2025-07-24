import cv2
import numpy as np

# Paths to the images
w_bg_guitar_file_abs_path = '../st_guitars/details/images/schecter_c_1_fr_s_sls_evil_twin_sbk.jpg'

# Load the image
w_bg_img = cv2.imread(w_bg_guitar_file_abs_path)

# Resize the image for visualization if needed
h, w = 600, 600
w_bg_img = cv2.resize(w_bg_img, (w, h))

# Function to update vertical lines dynamically
def update_lines(pos):
    col1 = cv2.getTrackbarPos("Col1", "White BG Image")
    col2 = cv2.getTrackbarPos("Col2", "White BG Image")

    # Create a copy of the image to draw lines
    w_bg_img_copy = w_bg_img.copy()

    # Draw vertical lines on the image
    cv2.line(w_bg_img_copy, (col1, 0), (col1, h), (0, 255, 0), 2)
    cv2.line(w_bg_img_copy, (col2, 0), (col2, h), (0, 255, 0), 2)

    # Show the image with lines
    cv2.imshow("White BG Image", w_bg_img_copy)

# Create a window
cv2.namedWindow("White BG Image")

# Initial column values
initial_col1 = 195
initial_col2 = 405

# Create trackbars for column boundaries
cv2.createTrackbar("Col1", "White BG Image", initial_col1, w - 1, update_lines)
cv2.createTrackbar("Col2", "White BG Image", initial_col2, w - 1, update_lines)

# Initialize the display
update_lines(0)

# Wait for user interaction
cv2.waitKey(0)
cv2.destroyAllWindows()

