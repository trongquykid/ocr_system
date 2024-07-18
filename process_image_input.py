import cv2
import numpy as np
from PIL import Image

def process_image_v2(image):
    # Convert to numpy array
    image_np = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to the entire image to create a blurred background
    blurred_background = cv2.GaussianBlur(gray, (55, 55), 0)

    # Use adaptive thresholding to create a binary image where the text is white and the background is black
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 9)

    # Invert the binary image so the text is black and the background is white
    binary_inv = cv2.bitwise_not(binary)

    # Use the binary image as a mask to combine the blurred background with the original text
    result = cv2.bitwise_and(blurred_background, blurred_background, mask=binary_inv)
    result = cv2.bitwise_or(result, cv2.bitwise_and(gray, gray, mask=binary))

    result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    return result_rgb
