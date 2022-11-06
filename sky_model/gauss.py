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

import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import sys
import cv2
import usage
import os
import common
import multiprocessing as mp
import targets.stars.detect
import cfg

def model(image):
	_,mask = targets.stars.detect.detect(image)
	shape = image.shape
	sky_blur = int(shape[1]/4)*2+1

	sky = np.zeros(shape)
	idx  = (mask==0)
	nidx = (mask!=0)
	sky[idx]  = image[idx]
	average   = np.mean(sky, axis=(0,1))
	sky[nidx] = average
	sky = cv2.GaussianBlur(sky, (sky_blur, sky_blur), 0)
	return sky
