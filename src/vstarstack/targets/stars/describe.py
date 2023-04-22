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

import os
import sys
import json
import math

import numpy as np
import vstarstack.common
import vstarstack.usage
import vstarstack.cfg


def dirvec(lat, lon):
    return np.array([math.cos(lon) * math.cos(lat),
                     math.sin(lon) * math.cos(lat),
                     math.sin(lat)])


def star2star(star1, star2):
    lon1 = star1["lon"]
    lat1 = star1["lat"]
    pos1 = dirvec(lat1, lon1)

    lon2 = star2["lon"]
    lat2 = star2["lat"]
    pos2 = dirvec(lat2, lon2)

    diff = pos2 - pos1
    dist = np.linalg.norm(diff)
    angle = 2*math.asin(dist/2) * 180/math.pi

    diff = diff / dist
    diff -= pos1 * np.sum(diff*pos1)
    diff /= np.linalg.norm(diff)
    return angle, diff


def build_description_angled(main):

    for i in range(len(main)):
        star = main[i]
        size = star["size"]
        star["descriptor"] = []
        other = []
        for j in range(len(main)):
            if i != j:
                other.append((j, main[j]))

        for j in range(len(other)-1):
            id1, other1 = other[j]
            angle1, dir1 = star2star(star, other1)
            size1 = other1["size"]

            for k in range(j+1, len(other)):
                id2, other2 = other[k]
                angle2, dir2 = star2star(star, other2)
                size2 = other2["size"]

                cosa = np.sum(dir1 * dir2)
                if cosa > 1:
                    cosa = 1
                if cosa < -1:
                    cosa = -1
                dangle = math.acos(cosa) * 180/math.pi
                if angle1 < angle2:
                    star["descriptor"].append(
                        (id1, angle1, size1/size, id2, angle2, size2/size, dangle))
                else:
                    star["descriptor"].append(
                        (id2, angle2, size2/size, id1, angle1, size1/size, dangle))
    return main


def build_description_distance(main):

    for i in range(len(main)):
        star = main[i]
        size = star["size"]
        star["descriptor"] = []
        other = []
        for j in range(len(main)):
            if i != j:
                other.append((j, main[j]))

        for j in range(len(other)):
            id1, other1 = other[j]
            angle1, dir1 = star2star(star, other1)
            size1 = other1["size"]

            star["descriptor"].append(
                (id1, angle1, size1/size, id1, angle1, size1/size, 0))
    return main


def get_brightest(stars, num, h, w, mindistance):
    mindistance = h * mindistance
    sample = []
    for i in range(len(stars)):
        star = stars[i]
        for s in sample:
            if abs(s["y"] - star["y"]) < mindistance and abs(s["x"] - star["x"]) < mindistance:
                break
        else:
            sample.append(star)
            if len(sample) >= num:
                break
    return sample


def build_descriptions(image, num_main, mindist, use_angles):
    main = get_brightest(image["stars"], num_main,
                         image["h"], image["w"], mindist)
    if use_angles:
        main = build_description_angled(main)
    else:
        main = build_description_distance(main)

    image.pop("stars")
    image["main"] = main
    return image


def process(project: vstarstack.cfg.Project, argv: list):
    if len(argv) >= 2:
        path = argv[0]
        outpath = argv[1]
    else:
        path = project.config["paths"]["descs"]
        outpath = project.config["paths"]["descs"]

    num_main = project.stars["describe"]["num_main"]
    mindist = project.stars["describe"]["mindist"]

    files = vstarstack.common.listfiles(path, ".json")

    for name, filename in files:
        print(name)
        with open(filename) as f:
            image = json.load(f)
        image = build_descriptions(
            image, num_main, mindist, project.stars["use_angles"])

        with open(os.path.join(outpath, name + ".json"), "w") as f:
            json.dump(image, f, indent=4)


def run(project: vstarstack.cfg.Project, argv: list):
    process(project, argv)
