import cv2
import numpy as np

def draw_lines_with_trackbar(img_path):
    def nothing(x):
        pass

    # Load the image
    img = cv2.imread(img_path)
    img_copy = img.copy()

    # Create a window and trackbars
    cv2.namedWindow('Line Tester')
    cv2.createTrackbar('Row1', 'Line Tester', 0, img.shape[0] - 1, nothing)
    cv2.createTrackbar('Row2', 'Line Tester', 0, img.shape[0] - 1, nothing)
    cv2.createTrackbar('Col1', 'Line Tester', 0, img.shape[1] - 1, nothing)
    cv2.createTrackbar('Col2', 'Line Tester', 0, img.shape[1] - 1, nothing)

    while True:
        # Get the trackbar positions
        row1 = cv2.getTrackbarPos('Row1', 'Line Tester')
        row2 = cv2.getTrackbarPos('Row2', 'Line Tester')
        col1 = cv2.getTrackbarPos('Col1', 'Line Tester')
        col2 = cv2.getTrackbarPos('Col2', 'Line Tester')

        # Reset the image to its original state
        img_copy = img.copy()

        # Draw vertical and horizontal lines based on trackbar values
        cv2.line(img_copy, (col1, 0), (col1, img_copy.shape[0]), (0, 0, 255), 2)  # Vertical line at Col1
        cv2.line(img_copy, (col2, 0), (col2, img_copy.shape[0]), (0, 255, 0), 2)  # Vertical line at Col2
        cv2.line(img_copy, (0, row1), (img_copy.shape[1], row1), (255, 0, 0), 2)  # Horizontal line at Row1
        cv2.line(img_copy, (0, row2), (img_copy.shape[1], row2), (255, 255, 0), 2)  # Horizontal line at Row2

        # Display the image
        cv2.imshow('Line Tester', img_copy)

        # Exit when 'esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Provide the path to your image
    strat_temp = "../templates/3.single_cut_temp.jpg"
    draw_lines_with_trackbar(strat_temp)
