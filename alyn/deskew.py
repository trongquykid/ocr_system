import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from alyn.skew_detect import SkewDetect
from skimage.transform import rotate

class Deskew:

    def __init__(self, input_file, display_image, output_file, r_angle):
        self.input_file = input_file
        self.display_image = display_image
        self.output_file = output_file
        self.r_angle = r_angle
        self.skew_obj = SkewDetect(image=input_file)  # Pass the image directly

    def deskew(self):
        res = self.skew_obj.determine_skew(None)  # Use None as the file path since we're passing the image
        angle = res['Estimated Angle']

        if angle >= 0 and angle <= 90:
            rot_angle = angle - 90 + self.r_angle
        elif angle >= -45 and angle < 0:
            rot_angle = angle - 90 + self.r_angle
        elif angle >= -90 and angle < -45:
            rot_angle = 90 + angle + self.r_angle

        rotated = rotate(self.input_file, rot_angle, resize=True)  # Rotate the image
        rotated = Image.fromarray((rotated * 255).astype(np.uint8))
        rotated = rotated.convert('RGB')
        return rotated

    def run(self):
        self.deskew()