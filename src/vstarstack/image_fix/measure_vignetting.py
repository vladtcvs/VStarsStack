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
import sys
import vstarstack.common
import vstarstack.projection.perspective
import math
import os
import numpy as np
import cv2


def run(project: vstarstack.cfg.Project, argv: list):
    proj = vstarstack.projection.perspective.Projection(project.camera.W,
                                                        project.camera.H,
                                                        project.scope.F,
                                                        project.camera.w,
                                                        project.camera.h)
    path = argv[0]
    outpath = argv[1]
    img = np.load(path).astype(np.float32)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    img[:, :, 0] /= np.amax(img[:, :, 0])
    img[:, :, 1] /= np.amax(img[:, :, 1])
    img[:, :, 2] /= np.amax(img[:, :, 2])

    h = img.shape[0]
    w = img.shape[1]
    N = 100

    f = open(os.path.join(outpath, "measure1"), "w")
    y = int(h/2)
    for i in range(N):
        x = int(i / N * (w-1))
        lat, lon = proj.project(y, x)
        cosa = math.cos(lon)*math.cos(lat)
        b = (img[y][x][0] + img[y][x][1] + img[y][x][2])/3
        f.write("%i %i %i %f %f\n" % (i, y, x, cosa, b))
    f.close()

    f = open(os.path.join(outpath, "measure2"), "w")
    x = int(w/2)
    for i in range(N):
        y = int(i / N * (h-1))
        lat, lon = proj.project(y, x)
        cosa = math.cos(lon)*math.cos(lat)
        b = (img[y][x][0] + img[y][x][1] + img[y][x][2])/3
        f.write("%i %i %i %f %f\n" % (i, y, x, cosa, b))
    f.close()

    f = open(os.path.join(outpath, "measure3"), "w")
    for i in range(N):
        x = int(i / N * (w-1))
        y = int(i / N * (h-1))
        lat, lon = proj.project(y, x)
        cosa = math.cos(lon)*math.cos(lat)
        b = (img[y][x][0] + img[y][x][1] + img[y][x][2])/3
        f.write("%i %i %i %f %f\n" % (i, y, x, cosa, b))
    f.close()

    np.save(os.path.join(outpath, ""), img)
