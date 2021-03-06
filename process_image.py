import numpy as np
import cv2
from utils import grayscale, canny, hough_lines, weighted_img, gaussian_blur, region_of_interest, draw_line, \
    separate_by_slope, slope_from_lin_reg
from scipy.stats import linregress


def process_image(image):
    original_image = image.copy()
    ysize = image.shape[0]
    xsize = image.shape[1]

    gray = grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    left_bottom = [0, ysize]
    right_bottom = [xsize, ysize]
    apex = [xsize / 2, ysize / 1.72]

    # This time we are defining a four sided polygon to mask
    vertices = np.array([[left_bottom, apex, apex, right_bottom]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    # lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    left_points, right_points = separate_by_slope(lines)

    if left_points:
        # Find the slope based on the generated points for the left line
        slope = slope_from_lin_reg (left_points)
        # Calculate x for the largest y. i.e find the lowest point on the image part of the  extrapolate line
        x2 = int(max(left_points)[0] + (ysize-max(left_points)[1])/slope)
        up_left_point = max(left_points)
        down_left_point = [x2, ysize]
        draw_line(line_image, up_left_point, down_left_point)

    if right_points:
        slope = slope_from_lin_reg(right_points)
        # Calculate x for the largest y. i.e find the lowest point on the image part of the  extrapolate line
        x2 = int(max(right_points)[0] + (ysize - max(right_points)[1]) / slope)
        up_right_point = min(right_points)
        down_right_point = [x2, ysize]
        draw_line(line_image, up_right_point, down_right_point)

    # Draw the lines on the edge image
    lines_edges = weighted_img(line_image, original_image)

    return lines_edges