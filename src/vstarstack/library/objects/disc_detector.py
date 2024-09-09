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

import cv2
import numpy as np
import math

from typing import Tuple

def len_of_vec(vec):
    """Vector length"""
    return (vec[0]**2+vec[1]**2)**0.5

def dir_of_vec(vec):
    """Direction of vector"""
    return vec / len_of_vec(vec)

def angle(vec1, vec2):
    """Angle between vectors"""
    vec1 = dir_of_vec(vec1)
    vec2 = dir_of_vec(vec2)

    scalar = vec1[0]*vec2[0] + vec1[1]*vec2[1]
    scalar = np.clip(scalar, 0, 1)
    return math.acos(scalar)

def get_point(contour, index):
    """Get point on contour"""
    while index < 0:
        index += contour.shape[0]
    while index >= contour.shape[0]:
        index -= contour.shape[0]
    return contour[index, 0, :]


def left(vec):
    """Vector, left to the specified"""
    return np.array([-vec[1], vec[0]])

def contour_curvature_d(contour, delta):
    """Calculate contour curvature"""
    centers = np.zeros((contour.shape[0], 2))
    curvatures = np.zeros((contour.shape[0]))
    for index in range(0, contour.shape[0]):
        prev_index = index - delta
        next_index = index + delta
        p_prev = get_point(contour, prev_index)
        p_cur = get_point(contour, index)
        p_next = get_point(contour, next_index)

        p = p_prev - p_cur
        n = p_next - p_cur
        px = p[0]
        py = p[1]
        nx = n[0]
        ny = n[1]

        D = 2*(ny*px-nx*py)
        if abs(D) < 1e-12:
            continue

        t = (ny*(ny-py))/D+(nx*(nx-px))/D
        # s = -((ny-py)*py)/D-((nx-px)*px)/D
        center = p_cur + p/2 + left(p)*t
        centers[index, :] = center
        radius = ((center[0]-p_cur[0])**2 + (center[1]-p_cur[1])**2)**0.5
        curvature = 1/radius
        curvatures[index] = curvature
    return centers, curvatures

def contour_curvature(contour, mindelta, maxdelta):
    """Find contour center and curvature at each point"""
    nump = maxdelta-mindelta+1
    contour_len = contour.shape[0]
    centers = np.zeros((nump, contour_len, 2))
    curvatures = np.zeros((nump, contour_len))
    index = 0
    for delta in range(mindelta, maxdelta+1):
        center, curvature = contour_curvature_d(contour, delta)
        centers[index] = center
        curvatures[index] = curvature
        index += 1
    centers = np.median(centers, axis=0)
    curvatures = np.median(curvatures, axis=0)
    return centers, curvatures

def radius_to_contour(contour, center):
    """Median radius from center to contour"""
    distances = []
    for i in range(len(contour)):
        point = contour[i, 0]
        distance = math.sqrt((point[0]-center[0])**2 + (point[1]-center[1])**2)
        distances.append(distance)
    return np.median(distances)

def sigma_clip(values, coefficient):
    """Select only values which are near the mean"""
    mean = np.mean(values)
    dispersion = np.std(values)*coefficient
    values = values[np.where(values >= mean - dispersion)]
    values = values[np.where(values <= mean + dispersion)]
    return values

def mean_center(centers):
    """Mean center"""
    centers0 = centers[:, 0]
    centers1 = centers[:, 1]

    centers0 = sigma_clip(centers0, 1)
    centers1 = sigma_clip(centers1, 1)

    center0 = np.mean(centers0)
    center1 = np.mean(centers1)
    center = np.array([center0, center1])
    return center

def measure_ring(layer : np.ndarray, x : int, y : int, radius1 : int, radius2 : int) -> float:
    """Measure average value of ring"""
    mask = np.zeros(layer.shape)
    cv2.circle(mask, (x, y), radius2, 1, -1)
    cv2.circle(mask, (x, y), radius1, 0, -1)
    pixels = layer * mask
    return np.average(pixels), np.std(pixels)

def display_contour(thresh_img : np.ndarray, contour : np.ndarray, centers : np.ndarray | None) -> None:
    import matplotlib.pyplot as plt
    rgb = np.zeros((thresh_img.shape[0], thresh_img.shape[1], 3))
    thresh_img = thresh_img.astype('uint8')*255/4
    rgb[:,:,0] = thresh_img
    rgb[:,:,1] = thresh_img
    rgb[:,:,2] = thresh_img

    for i in range(contour.shape[0]):
        cx = contour[i, 0, 0]
        cy = contour[i, 0, 1]
        rgb[cy, cx, 0] = 255
        rgb[cy, cx, 1] = 0
        rgb[cy, cx, 2] = 0
    
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

def detect(layer : np.ndarray,
           thresh : float,
           mindelta : float,
           maxdelta : float,
           num_bins_curvature : int,
           num_bins_distance : int):
    """Detect part of disc on image"""
    blurred = cv2.GaussianBlur(layer, (5, 5), 0)
    blurred = (blurred / np.amax(blurred) * 255).astype(np.uint8)

    thresh = np.average(blurred) * (1 - thresh) + np.amax(blurred) * thresh
    thresh = int(thresh/np.amax(blurred)*255)

    _, thresh_img = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        image=thresh_img,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_NONE)

    # contour contains data in (x,y) format
    if len(contours) == 0:
        return []

    # select 3 maximal contours
    contours = sorted(contours, key=lambda item: len(item), reverse=True)
    contours = [item for item in contours if len(item) >= len(contours[0])/2]

    # select contour with most stable curvature
    for contour in contours:
        centers, curvatures = contour_curvature(contour,
                                                mindelta=mindelta,
                                                maxdelta=maxdelta)
        

        # select only points of contour, where curvature is near to the most frequet
        values, bins = np.histogram(curvatures, bins=num_bins_curvature)
        ind = np.argmax(values)
        ks1 = bins[ind]
        ks2 = bins[ind+1]

        contour = contour[np.where(curvatures <= ks2)]
        centers = centers[np.where(curvatures <= ks2)]
        curvatures = curvatures[np.where(curvatures <= ks2)]
        contour = contour[np.where(curvatures >= ks1)]
        centers = centers[np.where(curvatures >= ks1)]
        curvatures = curvatures[np.where(curvatures >= ks1)]

#        display_contour(thresh_img, contour, centers)

        # select only points of contour, where center is near to the most frequent

        centers_x = centers[:,0]

        values, bins = np.histogram(centers_x, bins=num_bins_distance)
        ind = np.argmax(values)
        ks1 = bins[ind]
        ks2 = bins[ind+1]

        idx = np.where((centers_x <= ks2) & (centers_x >= ks1))
        contour = contour[idx]
        centers = centers[idx]
        curvatures = curvatures[idx]

        centers_y = centers[:,1]

        values, bins = np.histogram(centers_y, bins=num_bins_distance)
        ind = np.argmax(values)
        ks1 = bins[ind]
        ks2 = bins[ind+1]
        
        idx = np.where((centers_y <= ks2) & (centers_y >= ks1))
        contour = contour[idx]
        centers = centers[idx]
        curvatures = curvatures[idx]

#        display_contour(thresh_img, contour, centers)

        center = mean_center(centers)

        radiuses = np.zeros(centers.shape[0])
        for i in range(centers.shape[0]):
            point = contour[i, 0, :]
            radius = math.sqrt((point[0]-center[0])**2 + (point[1]-center[1])**2)
            radiuses[i] = radius

        radius = int(radius_to_contour(contour, center))

        # check that inside is brighter than outside
        x = int(center[0] + 0.5)
        y = int(center[1] + 0.5)
        light_inside, std_inside  = measure_ring(layer, x, y, radius-20, radius)
        light_outside, std_outside = measure_ring(layer, x, y, radius, radius+20)
        #print("inside: ", light_inside, " ", std_inside)
        #print("outside: ", light_outside, " ", std_outside)
        if light_inside > light_outside:
            break

    planet = {
        "x": x,
        "y": y,
        "r": int(radius + 0.5)
    }
    return [planet]
