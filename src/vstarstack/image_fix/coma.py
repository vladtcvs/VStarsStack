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

import math
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage.restoration
import sys

from PIL import Image

from numpy.fft import fft, ifft, ifftshift


def coma_psf(w, h, x, y, max_coma, size):
    dx = x - w/2
    dy = y - h/2
    r2 = (dx**2+dy**2)
    r02 = ((w/2)**2 + (h/2)**2)
    # here we assume that coma length ~ (r/r0)**2
    coma = max_coma * r2/r02
    angle = int(math.atan2(dy, dx)*(180/math.pi)+0.5)
    # we create ellips with minor axis = size
    # and major axis = size + coma. It rotated to the angle = angle
    s = math.ceil(size + max_coma)
    if s % 2 == 0:
        s += 1
    r = int(s/2)
    psf = np.zeros((s, s))
    a = math.ceil((coma+size)/2)
    b = math.ceil(size/2)
    # print(a,b)
    cv2.ellipse(psf, (r, r), (a, b), angle, 0, 360, 1, -1)
    psf /= np.sum(psf)
    return psf


def imgpix(img, x, y):
    w = img.shape[1]
    h = img.shape[0]
    if x < 0 or y < 0 or x >= w or y >= h:
        return 0
    return img[y, x]


def convolve(img, psf_function, max_coma):
    res = np.zeros(img.shape)
    w = img.shape[1]
    h = img.shape[0]
    for y in range(h):
        print(y)
        for x in range(w):
            psf = psf_function(w, h, x, y, max_coma, 1)
            sz = int(psf.shape[0]/2)
            block = img[y-sz:y+1+sz, x-sz:x+1+sz]

            # print(block.shape, psf.shape, sz, y, x)
            try:
                res[y, x] = np.sum(block * psf)
            except:
                pass
    return res


img = np.array(Image.open('test.png'))[:, :, 0]
img = img / np.amax(img)

r = 10
# kernel
# kernel = coma_psf(img.shape[1], img.shape[0], 0, 0, 18, 1)
# plt.imshow(kernel)
# plt.show()
# sys.exit()
# kernel = np.zeros((2*r+1,2*r+1))
# cv2.circle(kernel, (r, r), r, 1, -1)
# kernel /= np.sum(kernel)


conv = convolve(img, coma_psf, 18)

# print(conv.shape, kernel.shape)

# deconv = skimage.restoration.wiener(img, kernel, 0.05)
# deconv2 = skimage.restoration.richardson_lucy(img, kernel, 10)

fig, axs = plt.subplots(2)
fig.patch.set_facecolor('#222222')
axs[0].imshow(img, cmap='gray')
axs[1].imshow(conv, cmap='gray')
# axs[1,0].imshow(deconv, cmap='gray')
# axs[1,1].imshow(deconv2, cmap='gray')
plt.show()
