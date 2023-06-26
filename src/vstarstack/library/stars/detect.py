#
# Copyright (c) 2023 Vladislav Tsendrovskii
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

import math
import numpy as np
import cv2
import imutils
import imutils.contours

from skimage import measure

INSIDE_COEFF = 0.55
OUTSIDE_COEFF = 0.4
BRIGHTNESS_OVER_AREA = 2
BORDER_WIDTH = 10
MIN_STAR_R = 2
MAX_STAR_R = 20

def _make_patch(image : np.ndarray, x : int, y : int, r : int):
    patch = image[y-r*2:y+r*2+1,x-r*2:x+r*2+1]
    if patch.shape[0] == 0 or patch.shape[1] == 0:
        return None
    return patch

def check_round(patch : np.ndarray, r : int):
    """Check that image contains circle at (x,y) of r"""
    circle_mask = np.zeros(patch.shape)
    center = (int(patch.shape[0]/2+0.5), int(patch.shape[1]/2+0.5))
    cv2.circle(circle_mask, center, r, 1, -1)
    num_circle = cv2.countNonZero(circle_mask)
    num_inside_circle = cv2.countNonZero(circle_mask*patch)
    if num_inside_circle < num_circle*INSIDE_COEFF:
        # too big circle - too many black pixels inside circle
        return 1
    num_outside_circle = cv2.countNonZero((1-circle_mask)*patch)
    if num_outside_circle > num_circle*OUTSIDE_COEFF:
        # too small circle - too many white pixels outside circle
        return -1
    # good
    return 0

def check_star(binary_image : np.ndarray, x : int, y : int, min_r : int, max_r : int):
    """Check that object at (x,y) is a star"""
    low_r = min_r
    high_r = max_r
    patch = _make_patch(binary_image, x, y, low_r*2)
    if patch is None or check_round(patch, low_r) > 0:
        # Not a star - too low white pixels inside minimal circle
        return False, None

    patch = _make_patch(binary_image, x, y, high_r*2)
    while high_r > low_r and (patch is None or check_round(patch, high_r) < 0):
        high_r -= 1
        patch = _make_patch(binary_image, x, y, high_r*2)

    if high_r == low_r:
        # Not a star - too many white pixels outside minimal circle
        return False, None

    while high_r - low_r > 1:
        middle_r = int((low_r+high_r)/2+0.5)
        patch = _make_patch(binary_image, x, y, middle_r*2)
        atmiddle = check_round(patch, middle_r)
        if atmiddle == 0:
            return True, middle_r
        if atmiddle < 0:
            low_r = middle_r
        elif atmiddle > 0:
            high_r = middle_r
    middle_r = int((high_r + low_r)/2+0.5)
    patch = _make_patch(binary_image, x, y, middle_r*2)
    atmiddle = check_round(patch, middle_r)
    if atmiddle == 0:
        return True, middle_r
    return False, None

def calculate_brightness(image : np.ndarray, x : int, y : int, r : int):
    """Calculate brightness of a star at (x,y) with radius r"""
    patch = image[y-r:y+r+1, x-r:x+r+1]
    pos_mask = np.zeros(patch.shape)
    cv2.circle(pos_mask, (r, r), r, 1, -1)

    masked = patch * pos_mask
    brightness = math.sqrt(np.sum(masked) / math.pi)
    return brightness
    #intensity = np.sum(masked) / np.sum(pos_mask)
    #return intensity * r

def find_stars(binary_image : np.ndarray,
               gray_image : np.ndarray,
               min_r : int,
               max_r : int):
    """Find stars on image"""
    shape = binary_image.shape
    labels = measure.label(binary_image, connectivity=2, background=0)
    mask = np.zeros(shape, dtype="uint8")

    min_pixels = math.pi * min_r**2 * INSIDE_COEFF
    max_pixels = math.pi * max_r**2 * (1+OUTSIDE_COEFF)

    for label in np.unique(labels):
        if label == 0:
            continue
        label_mask = np.zeros(shape, dtype="uint8")
        label_mask[labels == label] = 255
        num_pixels = cv2.countNonZero(label_mask)
        if num_pixels < min_pixels or num_pixels > max_pixels:
            continue
        mask = cv2.add(mask, label_mask)

    mask = mask.copy()

    contours = cv2.findContours(mask,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = imutils.contours.sort_contours(contours)[0]

    stars = []

    # loop over the contours
    for contour in contours:
        (center_x, center_y), _ = cv2.minEnclosingCircle(contour)
        center_x = int(center_x+0.5)
        center_y = int(center_y+0.5)
        is_star, radius = check_star(mask, center_x, center_y, min_r, max_r)
        if not is_star:
            continue

        brightness = calculate_brightness(gray_image, center_x, center_y, radius)
        stars.append({"x": center_x, "y": center_y, "size": brightness, "radius" : radius})

    stars.sort(key=lambda s: s["size"], reverse=True)
    return stars

def detect_stars(image : np.ndarray):
    """Detect stars on image"""
    width = image.shape[1]
    height = image.shape[0]
    blur_size = int(MAX_STAR_R*2)
    if blur_size % 2 == 0:
        blur_size += 1

    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = image / np.amax(image)
    image = np.clip(image, 0, 1)
    image = image - np.min(image)
    image = image / np.amax(image)
    image = np.clip(image, 0, 1)

    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    mask = image > blurred*BRIGHTNESS_OVER_AREA

    mask[:, 0:BORDER_WIDTH] = 0
    mask[0:BORDER_WIDTH, :] = 0
    mask[:, (width-BORDER_WIDTH):width] = 0
    mask[(height-BORDER_WIDTH):height, :] = 0

    return find_stars(mask, image, MIN_STAR_R, MAX_STAR_R)

def configure_detector(*,
                       min_r = None,
                       max_r=None,
                       border = None,
                       brightness_over_area = None,
                       inside_coeff = None,
                       outside_coeff = None):
    """Configure detector parameters"""
    global MIN_STAR_R
    global MAX_STAR_R
    global BORDER_WIDTH
    global BRIGHTNESS_OVER_AREA
    global INSIDE_COEFF
    global OUTSIDE_COEFF
    if min_r is not None:
        MIN_STAR_R = min_r
    if max_r is not None:
        MAX_STAR_R = max_r
    if border is not None:
        BORDER_WIDTH = border
    if brightness_over_area is not None:
        BRIGHTNESS_OVER_AREA = brightness_over_area
    if inside_coeff is not None:
        INSIDE_COEFF = inside_coeff
    if outside_coeff is not None:
        OUTSIDE_COEFF = outside_coeff
