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

_detector_cfg = {
    "THRESHOLD_BLOCK_SIZE" : 31,
    "THRESHOLD_COEFF" : 1.2,
    "BORDER_WIDTH" : 10,
    "MIN_STAR_R" : 2,
    "MAX_STAR_R" : 20,
}

def calculate_brightness(image : np.ndarray, x : int, y : int, r : int):
    """Calculate brightness of a star at (x,y) with radius r"""
    patch = image[y-r:y+r+1, x-r:x+r+1]
    if patch.shape[0] != 2*r+1 or patch.shape[1] != 2*r+1:
        return None
    pos_mask = np.zeros(patch.shape)
    cv2.circle(pos_mask, (r, r), r, 1, -1)
    masked = patch * pos_mask
    brightness = math.sqrt(np.sum(masked) / math.pi)
    return brightness

def _threshold(image, radius, ratio):
    kernel = np.zeros((2*radius+1, 2*radius+1))
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    kernel = kernel / np.sum(kernel)
    filtered = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    mask = (image > filtered*ratio).astype('uint8')
    return mask

def _find_stars(gray_image : np.ndarray):
    """Find stars on image"""
    shape = gray_image.shape

    gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    gray_image = (gray_image / np.amax(gray_image) * 255).astype('uint8')

    thresh = _threshold(gray_image, _detector_cfg["THRESHOLD_BLOCK_SIZE"],
                                    _detector_cfg["THRESHOLD_COEFF"])
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)

    labels = measure.label(blob, connectivity=2, background=0)
    mask = np.zeros(shape, dtype="uint8")

    for label in np.unique(labels):
        if label == 0:
            continue
        label_mask = np.zeros(shape, dtype="uint8")
        label_mask[labels == label] = 255
        num_pixels = cv2.countNonZero(label_mask)
        mask = cv2.add(mask, label_mask)

    mask = mask.copy()
    contours = cv2.findContours(mask,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if len(contours) == 0:
        return []
    contours = imutils.contours.sort_contours(contours)[0]

    stars = []

    # loop over the contours
    for contour in contours:
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        center_x = int(center_x+0.5)
        center_y = int(center_y+0.5)

        if radius < _detector_cfg["MIN_STAR_R"]:
            continue
        if radius > _detector_cfg["MAX_STAR_R"]:
            continue

        brightness = calculate_brightness(gray_image, center_x, center_y, int(radius+0.5))
        if brightness is None:
            continue
        stars.append({"x": center_x, "y": center_y, "size": brightness, "radius" : radius})

    stars.sort(key=lambda s: s["size"], reverse=True)
    return stars

def detect_stars(image : np.ndarray):
    """Detect stars on image"""
    return _find_stars(image)

def configure_detector(*,
                       min_r = None,
                       max_r = None,
                       border = None,
                       thresh_block_size = None,
                       thresh_coeff = None):
    """Configure detector parameters"""
    global _detector_cfg
    if min_r is not None:
        _detector_cfg["MIN_STAR_R"] = min_r
    if max_r is not None:
        _detector_cfg["MAX_STAR_R"] = max_r
    if border is not None:
        _detector_cfg["BORDER_WIDTH"] = border
    if thresh_block_size is not None:
        _detector_cfg["THRESHOLD_BLOCK_SIZE"] = thresh_block_size
    if thresh_coeff is not None:
        _detector_cfg["THRESHOLD_COEFF"] = thresh_coeff
