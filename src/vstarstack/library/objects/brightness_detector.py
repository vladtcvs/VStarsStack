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
import cv2
import numpy as np
from skimage import measure
import imutils

def detect(layer : np.ndarray,
           min_size : int,
           max_size : int,
           thr : float):
    """Detect object by brightness"""
    min_pixels = math.floor(math.pi/4*min_size**2)
    max_pixels = math.ceil(math.pi/4*max_size**2)

    blurred = cv2.GaussianBlur(layer, (5, 5), 0)
    blurred = blurred / np.amax(blurred) * 255

    thresh = cv2.threshold(blurred, int(thr*255), 255, cv2.THRESH_BINARY)[1]

    labels = measure.label(thresh, connectivity=2, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
            # otherwise, construct the label mask and count the
            # number of pixels
        label_mask = np.zeros(thresh.shape, dtype="uint8")
        label_mask[labels == label] = 255
        num_pixels = cv2.countNonZero(label_mask)
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if num_pixels >= min_pixels and num_pixels <= max_pixels:
            mask = cv2.add(mask, label_mask)

    contours = cv2.findContours(mask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if len(contours) == 0:
        return None
    contours = contours.sort_contours(contours)[0]

    planetes = []
    # loop over the contours
    for contour in contours:
        # draw the bright spot on the image
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        planetes.append({"x": center_x, "y": center_y, "r": radius})

    return sorted(planetes, key=lambda x : x["r"], reverse=True)
