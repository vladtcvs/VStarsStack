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

import vstarstack.library.data
import vstarstack.tool.cfg
import cv2
import numpy as np

import imutils
from skimage import measure
import json
import matplotlib.pyplot as plt
import os


def find_keypoints_orb(image, x1, y1, detector):
    cpts = []
    points = detector.detect(image, mask=None)
    for point in points:
        cpts.append(
            {"x": point.pt[0]+x1, "y": point.pt[1]+y1, "size": point.size})
    return cpts


def find_keypoints_brightness(image, x1, y1, detector):
    blurSize = int(detector["blurSize"])
    k_thr = detector["k_thr"]
    minv = detector["minValue"]
    minPixel = detector["minPixel"]
    maxPixel = detector["maxPixel"]

    if blurSize % 2 == 0:
        blurSize += 1

    image = image / np.amax(image)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    avg = cv2.GaussianBlur(blurred, (blurSize, blurSize), 0)
    thresh = (blurred > avg * k_thr) * (blurred > minv)
    labels = measure.label(thresh, background=0, connectivity=2)
    mask = np.zeros(thresh.shape, dtype="uint8")

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
        if numPixels >= minPixel and numPixels < maxPixel:
            mask = cv2.add(mask, labelMask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cpts = []

    if len(cnts) > 0:
        cnts = imutils.contours.sort_contours(cnts)[0]
        # loop over the contours

        for _, c in enumerate(cnts):
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(cX), int(cY)), int(radius)+5, 1, 2)
            cpts.append({"x": cX+x1, "y": cY+y1, "size": radius+10})

#    plt.imshow(image)
#    plt.show()
    return cpts


def select_keypoints(image, fun, detector):
    # image = cv2.GaussianBlur(image, (3, 3), 0)
    shape = image.shape
    N = 1
    cpts = []

    for i in range(N):
        y1 = int(shape[0]/N*i)
        y2 = int(shape[0]/N*(i+1))
        for j in range(N):
            x1 = int(shape[1]/N*j)
            x2 = int(shape[1]/N*(j+1))
            subimage = image[y1:y2, x1:x2]
            cpts += fun(subimage, x1, y1, detector)

    return cpts


def find_keypoints(project, files):
    points = {}
    orb = cv2.ORB_create()

    definition = project.config["compact_objects"]["features"]["detector"]
    if definition["type"] == "orb":
        detector = orb
        fun = find_keypoints_orb
    elif definition["type"] == "brightSpots":
        detector = definition
        fun = find_keypoints_brightness

    for fname in files:
        name = os.path.splitext(os.path.basename(fname))[0]
        print(name)
        dataframe = vstarstack.library.data.DataFrame.load(fname)
        for channel in dataframe.get_channels():
            image, opts = dataframe.get_channel(channel)
            if opts["weight"]:
                continue
            if opts["encoded"]:
                continue
            if not opts["brightness"]:
                continue
            if channel not in points:
                points[channel] = {}
            image = (image / np.amax(image) * 255).astype(np.uint8)

            keypoints = select_keypoints(image, fun, detector)

            kps = [cv2.KeyPoint(point["x"], point["y"], point["size"])
                   for point in keypoints]
            _, descs = orb.compute(image, kps)
            points[channel][name] = {
                "points": keypoints, "descs": descs, "fname": fname}
    return points


def match_images(points):
    kdist = float(project.config["compact_objects"]
                  ["features"]["featuresPercent"])/100
    maxd = float(project.config["compact_objects"]
                 ["features"]["maxFeatureDelta"])

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = {}
    for channel in points:
        print("Channel = %s" % channel)
        names = sorted(list(points[channel].keys()))
        matches[channel] = {}
        for ind1 in range(len(names)):
            name1 = names[ind1]
            if name1 not in matches[channel]:
                matches[channel][name1] = {}

            for ind2 in range(ind1):
                name2 = names[ind2]
                if name2 not in matches[channel]:
                    matches[channel][name2] = {}

                matches[channel][name1][name2] = []
                matches[channel][name2][name1] = []

                print("\t%s <-> %s" % (name1, name2))

                des1 = points[channel][name1]["descs"]
                des2 = points[channel][name2]["descs"]

                imatches = bf.match(des1, des2)
                imatches = sorted(imatches, key=lambda x: x.distance)

                nm = int(len(imatches) * kdist)
                imatches = imatches[:nm]
                # imatches = imatches[:150]

                if len(imatches) == 0:
                    continue

                deltaxs = []
                deltays = []

                for match in imatches:
                    index2 = match.trainIdx
                    index1 = match.queryIdx

                    point1 = points[channel][name1]["points"][index1]
                    point2 = points[channel][name2]["points"][index2]

                    deltaxs.append(point1["x"] - point2["x"])
                    deltays.append(point1["y"] - point2["y"])

                meandx = sum(deltaxs) / len(deltaxs)
                meandy = sum(deltays) / len(deltays)

                for match in imatches:
                    index2 = match.trainIdx
                    index1 = match.queryIdx

                    point1 = points[channel][name1]["points"][index1]
                    point2 = points[channel][name2]["points"][index2]

                    dx = point1["x"] - point2["x"]
                    dy = point1["y"] - point2["y"]
                    if abs(dx - meandx) > maxd:
                        continue
                    if abs(dy - meandy) > maxd:
                        continue

                    matches[channel][name1][name2].append(
                        (index1, index2, match.distance))
                    matches[channel][name2][name1].append(
                        (index2, index1, match.distance))

                if vstarstack.tool.cfg.DEBUG:
                    draw_matches(points, matches, channel, name1, name2)
    return matches


def draw_matches(points, matches, channel, name1, name2):
    points1 = points[channel][name1]["points"]
    points2 = points[channel][name2]["points"]
    fname1 = points[channel][name1]["fname"]
    fname2 = points[channel][name2]["fname"]

    d1 = vstarstack.library.data.DataFrame.load(fname1)
    img1, _ = d1.get_channel(channel)
    d2 = vstarstack.library.data.DataFrame.load(fname2)
    img2, _ = d2.get_channel(channel)

    img1 = (img1 / np.amax(img1) * 255).astype(np.uint8)
    img2 = (img2 / np.amax(img2) * 255).astype(np.uint8)

    ms = matches[channel][name1][name2]
    matches_fmt = [cv2.DMatch(msitem[0], msitem[1], 0) for msitem in ms]

    kps1 = [cv2.KeyPoint(point["x"], point["y"], point["size"])
            for point in points1]
    kps2 = [cv2.KeyPoint(point["x"], point["y"], point["size"])
            for point in points2]

    img3 = cv2.drawMatches(img1, kps1, img2, kps2,
                           matches_fmt, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()


def build_clusters(matches):
    clusters = {}
    for channel in matches:
        clusters[channel] = []
        channel_matches = matches[channel]
        for name1 in channel_matches:
            for name2 in channel_matches[name1]:
                matches_list = channel_matches[name1][name2]
                for match in matches_list:
                    id1 = match[0]
                    id2 = match[1]
                    for cluster in clusters[channel]:
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
                        clusters[channel].append(cluster)
    return clusters


def build_coordinate_clusters(clusters, points):
    crd_clusters = {}
    for channel in clusters:
        crd_clusters[channel] = []
        for cluster in clusters[channel]:
            crd_cluster = {}
            for name in cluster:
                id = cluster[name]
                crd_cluster[name] = points[channel][name]["points"][id]
            crd_clusters[channel].append(crd_cluster)
    return crd_clusters


def run(project: vstarstack.tool.cfg.Project, argv: list):
    inputs = argv[0]
    clusters_fname = argv[1]

    files = vstarstack.common.listfiles(inputs, ".zip")
    files = [filename for name, filename in files]
    points = find_keypoints(files)
    matches = match_images(points)
    clusters = build_clusters(matches)
    crd_clusters = build_coordinate_clusters(clusters, points)

    total_clusters = []
    for channel in crd_clusters:
        ch_clusters = crd_clusters[channel]
        total_clusters += ch_clusters
    with open(clusters_fname, "w") as f:
        json.dump(total_clusters, f, indent=4, ensure_ascii=False)
