#
# Copyright (c) 2022 Vladislav Tsendrovskii
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

import vstarstack.cfg

import cv2
import numpy as np
from imutils import contours
from skimage import measure
import matplotlib.pyplot as plt
import imutils

def detect(layer, debug=False):
    min_pixels = vstarstack.cfg.config["compact_objects"]["brightness"]["minPixels"]
    max_pixels = vstarstack.cfg.config["compact_objects"]["brightness"]["maxPixels"]
    thr = vstarstack.cfg.config["compact_objects"]["threshold"]

    blurred = cv2.GaussianBlur(layer, (5, 5), 0)
    mb = np.amax(blurred)
    blurred = blurred / mb * 255

    thresh = cv2.threshold(blurred, int(thr*255), 255, cv2.THRESH_BINARY)[1]

    if debug:
        plt.imshow(thresh, cmap="gray")
        plt.show()

    labels = measure.label(thresh, connectivity=2, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
	    # otherwise, construct the label mask and count the
	    # number of pixels
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
		# if the number of pixels in the component is sufficiently
		# large, then add it to our mask of "large blobs"
        if numPixels >= min_pixels and numPixels <= max_pixels:
            mask = cv2.add(mask, labelMask)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	                        cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts) == 0:
            return None
        cnts = contours.sort_contours(cnts)[0]

    planetes = []

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        planetes.append({"x": cX, "y": cY, "r": radius})

    if len(planetes) != 1:
        print("Error: len(planetes) = %i" % (len(planetes)))
        return None
    return planetes[0]
