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

import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import random
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull


from typing import Tuple

def get_point(contour, index):
    """Get point on contour"""
    while index < 0:
        index += contour.shape[0]
    while index >= contour.shape[0]:
        index -= contour.shape[0]
    return contour[index, 0, :]


def circle_from_3pts(p1, p2, p3):
    """Окружность по трём точкам"""
#    print(p1, p2, p3)
    x1 = p1[0]
    x2 = p2[0]
    x3 = p3[0]
    y1 = p1[1]
    y2 = p2[1]
    y3 = p3[1]

    a = y1 - y2
    b = y3 - y1
    c = x2 - x1
    d = x1 - x3

    p = (x3 - x2) / 2
    q = (y3 - y2) / 2

    D = a*d-b*c
    ia = d / D
    ib = -b / D
    ic = -c / D
    id = a / D

    t = ia * p + ib * q
    s = ic * p + id * q

    xc2 = x1 + (x2 - x1)/2 + t * (y1 - y2)
    yc2 = y1 + (y2 - y1)/2 + t * (x2 - x1)
    
    xc3 = x1 + (x3 - x1)/2 + s * (y1 - y3)
    yc3 = y1 + (y3 - y1)/2 + s * (x3 - x1)

    xc = (xc2 + xc3)/2
    yc = (yc2 + yc3)/2

    r = math.sqrt((xc - x1)**2 + (yc - y1)**2)

    return xc, yc, r

def find_disc(contour_points, iterations, threshold):
    L = contour_points.shape[0]
    best_inliers = []
    best_circle = None
    for _ in range(iterations):
        i1 = int(random.random() * L)
        i2 = int(random.random() * L)
        i3 = int(random.random() * L)
        if i1 == i2 or i1 == i3 or i2 == i3:
            continue
        p1 = contour_points[i1]
        p2 = contour_points[i2]
        p3 = contour_points[i3]

        try:
            xc, yc, r = circle_from_3pts(p1, p2, p3)
        except:
            continue
        dists = np.abs(np.sqrt((contour_points[:,0]-xc)**2 + (contour_points[:,1]-yc)**2) - r)
        inliers = contour_points[dists < threshold]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_circle = (xc, yc, r)
            #print(len(inliers), " : ", p1, p2, p3, xc, yc, r)

    return best_circle, np.array(best_inliers)

def display_contour(thresh_img : np.ndarray, contour : np.ndarray, centers : np.ndarray | None) -> None:
    import matplotlib.pyplot as plt
    rgb = np.zeros((thresh_img.shape[0], thresh_img.shape[1], 3))
    thresh_img = thresh_img.astype('uint8')*255/4
    rgb[:,:,0] = thresh_img
    rgb[:,:,1] = thresh_img
    rgb[:,:,2] = thresh_img

    h = thresh_img.shape[0]
    w = thresh_img.shape[1]

    for i in range(contour.shape[0]):
        cx = contour[i, 0, 0]
        cy = contour[i, 0, 1]
        rgb[cy, cx, 0] = 255
        rgb[cy, cx, 1] = 0
        rgb[cy, cx, 2] = 0
        if cx + 1 < w:
            rgb[cy, cx+1, 0] = 255
            rgb[cy, cx+1, 1] = 0
            rgb[cy, cx+1, 2] = 0
        if cx - 1 >= 0:
            rgb[cy, cx-1, 0] = 255
            rgb[cy, cx-1, 1] = 0
            rgb[cy, cx-1, 2] = 0
        if cy + 1 < h:
            rgb[cy+1, cx, 0] = 255
            rgb[cy+1, cx, 1] = 0
            rgb[cy+1, cx, 2] = 0
        if cy - 1 >= 0:
            rgb[cy-1, cx, 0] = 255
            rgb[cy-1, cx, 1] = 0
            rgb[cy-1, cx, 2] = 0
    
    if centers is not None:
        for i in range(centers.shape[0]):
            x = int(centers[i,0])
            y = int(centers[i,1])
            if x < 0 or y < 0 or x >= rgb.shape[1] or y >= rgb.shape[0]:
                continue
            rgb[y, x, 0] = 0
            rgb[y, x, 1] = 255
            rgb[y, x, 2] = 0

    plt.imshow(rgb)
    plt.show()

def find_largest_contour(layer : np.ndarray,
                         thresh : float,
                         circle_threshold : float):
    """Detect part of disc on image"""
    blurred = gaussian_filter(layer, 2)
    blurred = (blurred / np.amax(blurred) * 255).astype(np.uint8)

    thresh = np.average(blurred) * (1 - thresh) + np.amax(blurred) * thresh
    thresh = int(thresh/np.amax(blurred)*255)

    thresh_img = (blurred > thresh).astype(np.uint8)*255

    #plt.imshow(thresh_img)
    #plt.show()

    contours, _ = cv2.findContours(
        image=thresh_img,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_NONE)

    # contour contains data in (x,y) format
    if len(contours) == 0:
        return []

    contour = max(contours, key=cv2.contourArea)
    points = np.reshape(contour, (contour.shape[0], 2))
    hull = ConvexHull(points)
    contour = contour[hull.vertices, :, :]
    points = points[hull.vertices, :]

    (x, y, radius), inliers = find_disc(points, 500, circle_threshold)

    inliers = np.reshape(inliers, (inliers.shape[0], 1, 2))
    #display_contour(thresh_img, contour, None)    
    planet = {
        "x": x,
        "y": y,
        "r": int(radius + 0.5)
    }
    return [planet]

def detect(layer : np.ndarray,
           thresh : float,
           circle_threshold : float):
    return find_largest_contour(layer, thresh, circle_threshold)
