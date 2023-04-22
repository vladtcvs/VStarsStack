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

import math
import sys
import cv2
import numpy as np
from imutils import contours
from skimage import measure
import matplotlib.pyplot as plt
import imutils


def len_of_vec(vec):
    len = (vec[0]**2+vec[1]**2)**0.5
    return len


def dir_of_vec(vec):
    return vec / len_of_vec(vec)


def angle(vec1, vec2):
    vec1 = dir_of_vec(vec1)
    vec2 = dir_of_vec(vec2)

    s = vec1[0]*vec2[0] + vec1[1]*vec2[1]
    s = np.clip(s, 0, 1)
    return math.acos(s)


def get_point(contour, index):
    while index < 0:
        index += contour.shape[0]
    while index >= contour.shape[0]:
        index -= contour.shape[0]
    return contour[index, 0, :]


def left(vec):
    return np.array([-vec[1], vec[0]])


def contour_curvature_d(contour, delta):
    centers = np.zeros((contour.shape[0], 2))
    ks = np.zeros((contour.shape[0]))
    for i in range(0, contour.shape[0]):
        prev = i - delta
        next = i + delta
        p_prev = get_point(contour, prev)
        p_cur = get_point(contour, i)
        p_next = get_point(contour, next)

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
        centers[i, :] = center
        r = ((center[0]-p_cur[0])**2 + (center[1]-p_cur[1])**2)**0.5
        k = 1/r
        ks[i] = k
    return centers, ks


def contour_curvature(contour, mindelta, maxdelta):
    nump = maxdelta-mindelta+1
    len = contour.shape[0]
    cs = np.zeros((nump, len, 2))
    ks = np.zeros((nump, len))
    id = 0
    for delta in range(mindelta, maxdelta+1):
        c, k = contour_curvature_d(contour, delta)
        cs[id] = c
        ks[id] = k
        id += 1
    cs = np.median(cs, axis=0)
    ks = np.median(ks, axis=0)
    return cs, ks


def radius_to_contour(contour, center):
    ds = []
    for i in range(len(contour)):
        p = contour[i, 0]
        d = ((p[0]-center[0])**2 + (p[1]-center[1])**2)**0.5
        ds.append(d)
    return np.median(ds)


def sigma_clip(values, k):
    mean = np.mean(values)
    d = np.std(values)*k
    values = values[np.where(values >= mean - d)]
    values = values[np.where(values <= mean + d)]
    return values


def mean_center(centers):
    centers0 = centers[:, 0]
    centers1 = centers[:, 1]

    centers0 = sigma_clip(centers0, 1)
    centers1 = sigma_clip(centers1, 1)

    center0 = np.mean(centers0)
    center1 = np.mean(centers1)
    center = np.array([center0, center1])
    return center


def detect(project, layer, debug=False):
    thresh = project.config["compact_objects"]["threshold"]
    mindelta = project.config["compact_objects"]["disc"]["mindelta"]
    maxdelta = project.config["compact_objects"]["disc"]["maxdelta"]
    bins_ks = project.config["compact_objects"]["disc"]["num_bins_curvature"]
    bins_d = project.config["compact_objects"]["disc"]["num_bins_distance"]

    blurred = cv2.GaussianBlur(layer, (5, 5), 0)
    mb = np.amax(blurred)
    blurred = (blurred / mb * 255).astype(np.uint8)

    thresh = int(thresh*255)

    ret, thresh_img = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        image=thresh_img,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)

    # contour contains data in (x,y) format

    # draw contours on the original image
    if len(contours) == 0:
        return None
    contour = sorted(contours, key=lambda item: len(item), reverse=True)[0]

    centers, ks = contour_curvature(
        contour, mindelta=mindelta, maxdelta=maxdelta)

    # select only points of contour, where curvature is near to the most frequet
    values, bins = np.histogram(ks, bins=bins_ks)
    ind = np.argmax(values)
    ks1 = bins[ind]
    ks2 = bins[ind+1]

    contour = contour[np.where(ks <= ks2)]
    centers = centers[np.where(ks <= ks2)]
    ks = ks[np.where(ks <= ks2)]
    contour = contour[np.where(ks >= ks1)]
    centers = centers[np.where(ks >= ks1)]
    ks = ks[np.where(ks >= ks1)]

    # select only points of contour, which distance
    # to centers is near to the most frequent
    center = mean_center(centers)

    rs = np.zeros(centers.shape[0])
    for i in range(centers.shape[0]):
        p = contour[i, 0, :]
        r = ((p[0]-center[0])**2 + (p[1]-center[1])**2)**0.5
        rs[i] = r

    values, bins = np.histogram(rs, bins=bins_d)
    ind = np.argmax(values)
    rs1 = bins[ind]
    rs2 = bins[ind+1]

    idx = np.where(rs <= rs2)
    contour = contour[idx]
    centers = centers[idx]
    ks = ks[idx]
    rs = rs[idx]

    idx = np.where(rs >= rs1)
    contour = contour[idx]
    centers = centers[idx]
    ks = ks[idx]
    rs = rs[idx]

    center = mean_center(centers)
    r = radius_to_contour(contour, center)

    if debug:
        image_copy = np.zeros((layer.shape[0], layer.shape[1], 3))
        image_copy[:, :, 0] = layer

        for i in range(len(centers)):
            x, y = centers[i].astype(np.int32)
            cv2.circle(image_copy, (x, y), 2, (0, 255, 255), -1)

        x, y = center.astype(np.int32)
        cv2.circle(image_copy, (x, y), int(r), (0, 255, 255), 2)
        cv2.circle(image_copy, (x, y), 4, (255, 255, 255), 2)

        cv2.drawContours(image_copy, [contour], 0, (0, 255, 0), 2)
        plt.imshow(image_copy)
        plt.show()

    planet = {
        "x": int(center[0]),
        "y": int(center[1]),
        "r": int(r)
    }
    return planet
