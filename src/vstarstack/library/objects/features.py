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

import imutils
import imutils.contours

from skimage import measure
import matplotlib.pyplot as plt

import vstarstack.library.data
import vstarstack.tool.cfg

def get_subimage(image, num_splits):
    image_shape = image.shape
    for i in range(num_splits):
        base_y = int(image_shape[0]/num_splits*i)
        next_y = int(image_shape[0]/num_splits*(i+1))
        for j in range(num_splits):
            base_x = int(image_shape[1]/num_splits*j)
            next_x = int(image_shape[1]/num_splits*(j+1))
            subimage = image[base_y:next_y, base_x:next_x]
            yield subimage, base_x, base_y

def _find_keypoints_orb(image : np.ndarray, base_x : int, base_y : int, detector):
    """Find keypoints with ORB detector"""
    cpts = []
    if image.dtype != np.uint8:
        image = (image * 255 / np.amax(image)).astype("uint8")
    points = detector.detect(image, mask=None)
    for point in points:
        pdesc = {
                "x": point.pt[0]+base_x,
                "y": point.pt[1]+base_y,
                "size": point.size,
                }
        cpts.append(pdesc)
    return cpts

def find_keypoints_orb(image, num_splits, param):
    points = []
    orb = cv2.ORB_create(patchSize=param["patchSize"])
    for subimage, bx, by in get_subimage(image, num_splits):
        points += _find_keypoints_orb(subimage, bx, by, orb)
    return points

def _find_keypoints_brightness(image, base_x, base_y, params):
    """Find keypoints with brightness detector"""
    blur_size = int(params["blur_size"])
    k_thr = params["k_thr"]
    minv = params["min_value"]
    min_pixel = params["min_pixel"]
    max_pixel = params["max_pixel"]

    if blur_size % 2 == 0:
        blur_size += 1

    image = image / np.amax(image)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    avg = cv2.GaussianBlur(blurred, (blur_size, blur_size), 0)
    thresh = (blurred > avg * k_thr) * (blurred > minv)
    labels = measure.label(thresh, background=0, connectivity=2)
    mask = np.zeros(thresh.shape, dtype="uint8")

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
        if num_pixels >= min_pixel and num_pixels < max_pixel:
            mask = cv2.add(mask, label_mask)

    contours = cv2.findContours(mask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    keypoints = []

    if len(contours) > 0:
        contours = imutils.contours.sort_contours(contours)[0]

        # loop over the contours
        for contour in contours:
            ((center_x, center_y), radius) = cv2.minEnclosingCircle(contour)
            keypoints.append({"x": center_x+base_x, "y": center_y+base_y, "size": radius+10})

    return keypoints

def find_keypoints_brightness(image, num_splits, params):
    points = []
    for subimage, bx, by in get_subimage(image, num_splits):
        points += _find_keypoints_brightness(subimage, bx, by, params)
    return points

def describe_keypoints(image : np.ndarray,
                       keypoints : list,
                       param : dict) -> list:
    """
    Build keypoints and calculate their descriptors

    Arguments:
        image         : image with target image
        num_split     : split image to num_split subimages
        detector_type : orb or brightness
        params        : parameters

    Return: list of keypoint and list of descriptors
    """
    orb = cv2.ORB_create(patchSize=param["patchSize"])
    kps = [cv2.KeyPoint(point["x"], point["y"], point["size"]) for point in keypoints]
    image = np.clip(image / np.amax(image), 0, 1)*255
    image = image.astype('uint8')
    _, descs = orb.compute(image, kps)
    return descs

def match_images(points : dict, descs : dict,
                 max_feature_delta : float,
                 features_percent : float,
                 match_list : list):
    """Match images"""
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = {}
    for name1, name2 in match_list:
        if name1 not in matches:
            matches[name1] = {}

        if name2 not in matches:
            matches[name2] = {}

        points1 = points[name1]
        descs1 = descs[name1]

        matches[name1][name2] = []
        matches[name2][name1] = []

        points2 = points[name2]
        descs2 = descs[name2]

        if descs1 is None or descs2 is None:
            print(f"Skipping {name1} <-> {name2}")
            continue


        imatches = bf_matcher.match(descs1, descs2)
        imatches = sorted(imatches, key=lambda x: x.distance)

        num_matches = int(len(imatches) * features_percent)
        imatches = imatches[:num_matches]

        if len(imatches) == 0:
            continue

        delta_xs = []
        delta_ys = []

        for match in imatches:
            index2 = match.trainIdx
            index1 = match.queryIdx

            point1 = points1[index1]
            point2 = points2[index2]

            delta_xs.append(point1["x"] - point2["x"])
            delta_ys.append(point1["y"] - point2["y"])

        mean_delta_x = sum(delta_xs) / len(delta_xs)
        mean_delta_y = sum(delta_ys) / len(delta_ys)

        for match in imatches:
            index2 = match.trainIdx
            index1 = match.queryIdx

            point1 = points1[index1]
            point2 = points2[index2]

            delta_x = point1["x"] - point2["x"]
            delta_y = point1["y"] - point2["y"]
            if abs(delta_x - mean_delta_x) > max_feature_delta:
                continue
            if abs(delta_y - mean_delta_y) > max_feature_delta:
                continue

            matches[name1][name2].append((index1, index2, match.distance))
            matches[name2][name1].append((index2, index1, match.distance))

    return matches

def build_index_clusters(matches : dict):
    """Build clusters of features"""
    clusters = []
    for name1 in matches:
        for name2 in matches[name1]:
            matches_list = matches[name1][name2]
            for match in matches_list:
                id1 = match[0]
                id2 = match[1]
                for cluster in clusters:
                    if name1 in cluster and cluster[name1] == id1:
                        cluster[name2] = id2
                        break
                    if name2 in cluster and cluster[name2] == id2:
                        cluster[name1] = id1
                        break
                else:
                    cluster = {
                        name1: id1,
                        name2: id2,
                    }
                    clusters.append(cluster)
    clusters = [item for item in clusters if len(item) > 1]
    return clusters

def build_crd_clusters(index_clusters : dict, points : dict):
    """Build coordinate clusters """
    crd_clusters = []
    #print(points.keys())
    for cluster in index_clusters:
        crd_cluster = {}
        for name in cluster:
            index = cluster[name]
            crd_cluster[name] = points[name][index]
        crd_clusters.append(crd_cluster)
    return crd_clusters

def build_clusters(points : dict, descs : dict,
                   max_feature_delta : float,
                   features_percent : float,
                   match_list : list):
    """Build clusters"""
    print("Match images")
    matches = match_images(points, descs, max_feature_delta, features_percent, match_list)
    print("Build index clusters")
    index_clusters = build_index_clusters(matches)
    print("Build coordinate clusters")
    crd_clusters = build_crd_clusters(index_clusters, points)
    return crd_clusters

def draw_matches(points : dict,
                 fnames : dict,
                 matches : dict,
                 channel : str,
                 name1 : str,
                 name2 : str):
    """Draw matches"""
    points1 = points[name1]
    points2 = points[name2]
    fname1 = fnames[name1]
    fname2 = fnames[name2]

    d1 = vstarstack.library.data.DataFrame.load(fname1)
    img1, _ = d1.get_channel(channel)
    d2 = vstarstack.library.data.DataFrame.load(fname2)
    img2, _ = d2.get_channel(channel)

    img1 = (img1 / np.amax(img1) * 255).astype(np.uint8)
    img2 = (img2 / np.amax(img2) * 255).astype(np.uint8)

    ms = matches[name1][name2]
    matches_fmt = [cv2.DMatch(msitem[1], msitem[2], 0) for msitem in ms]

    kps1 = [cv2.KeyPoint(point["x"], point["y"], point["size"])
            for point in points1]
    kps2 = [cv2.KeyPoint(point["x"], point["y"], point["size"])
            for point in points2]

    img3 = cv2.drawMatches(img1, kps1, img2, kps2,
                           matches_fmt, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()
