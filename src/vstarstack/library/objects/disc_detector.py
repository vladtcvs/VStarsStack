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
        distance = ((point[0]-center[0])**2 + (point[1]-center[1])**2)**0.5
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

def detect(layer : np.ndarray,
           thresh : float,
           mindelta : float,
           maxdelta : float,
           num_bins_curvature : int,
           num_bins_distance : int):
    """Detect part of disc on image"""
    blurred = cv2.GaussianBlur(layer, (5, 5), 0)
    blurred = (blurred / np.amax(blurred) * 255).astype(np.uint8)
    thresh = int(thresh*255)

    _, thresh_img = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        image=thresh_img,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)

    # contour contains data in (x,y) format
    if len(contours) == 0:
        return None

    # select maximal contour
    contour = sorted(contours, key=lambda item: len(item), reverse=True)[0]

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

    # select only points of contour, which distance
    # to centers is near to the most frequent
    center = mean_center(centers)

    radiuses = np.zeros(centers.shape[0])
    for i in range(centers.shape[0]):
        point = contour[i, 0, :]
        radius = ((point[0]-center[0])**2 + (point[1]-center[1])**2)**0.5
        radiuses[i] = radius

    values, bins = np.histogram(radiuses, bins=num_bins_distance)
    ind = np.argmax(values)
    rs1 = bins[ind]
    rs2 = bins[ind+1]

    idx = np.where(radiuses <= rs2)
    contour = contour[idx]
    centers = centers[idx]
    curvatures = curvatures[idx]
    radiuses = radiuses[idx]

    idx = np.where(radiuses >= rs1)
    contour = contour[idx]
    centers = centers[idx]
    curvatures = curvatures[idx]
    radiuses = radiuses[idx]

    center = mean_center(centers)
    radius = radius_to_contour(contour, center)

    planet = {
        "x": int(center[0] + 0.5),
        "y": int(center[1] + 0.5),
        "r": int(radius + 0.5)
    }
    return [planet]
