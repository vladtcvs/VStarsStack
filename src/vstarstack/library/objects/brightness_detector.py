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

    size = min(layer.shape)
    sky_blur = int(size / 2)
    if sky_blur % 2 == 0:
        sky_blur = sky_blur + 1
    sky = cv2.GaussianBlur(layer, (sky_blur, sky_blur), 0)
    blurred = blurred - sky
    blurred = blurred / np.amax(blurred)
    blurred = np.clip(blurred, 0, 1)
    blurred = (blurred * 255).astype('uint8')

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

    object_contours = cv2.findContours(mask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    object_contours = imutils.grab_contours(object_contours)
    if len(object_contours) == 0:
        return []
    object_contours = imutils.contours.sort_contours(object_contours)[0]

    planetes = []
    # loop over the contours
    for contour in object_contours:
        # draw the bright spot on the image
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        planetes.append({"x": center_x, "y": center_y, "r": radius})

    return sorted(planetes, key=lambda x : x["r"], reverse=True)
