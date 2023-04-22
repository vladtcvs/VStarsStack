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

from scipy.signal import convolve2d as conv2
import scipy

from skimage import color, restoration

from PIL import Image
import numpy as np
import sys

import vstarstack.data
import vstarstack.cfg


def gaussuian_filter(kernel_size, sigma=1, muu=0):

    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size

    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)

    # Calculating Gaussian filter
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2)))
    norm = np.sum(gauss)
    gauss /= norm

#    plt.imshow(gauss)
#    plt.show()

    return gauss


def make_deconvolution(image, psf):
    # image = scipy.ndimage.median_filter(image, size=7)
    # return image
    maxv = np.amax(image)
    minv = np.amin(image)
    image = (image - minv) / (maxv-minv)
    # Restore Image using Richardson-Lucy algorithm
    # result = restoration.richardson_lucy(image, psf, num_iter=130)
    result = restoration.wiener(image, psf, 0.01)

    result = result * (maxv - minv) + minv
    return result


def process(fname, psf_name, outfname):
    dataframe = vstarstack.data.DataFrame.load(fname)
    psf = np.asarray(Image.open(psf_name)).astype(np.float64)
    if len(psf.shape) == 3:
        psf = psf[:, :, 0]
    psf = psf / np.sum(psf)
    print(psf.shape)
    # psf = gaussuian_filter(13, 0.25, 0)
    for channel in dataframe.get_channels():
        image, opts = dataframe.get_channel(channel)
        if opts["weight"]:
            continue
        if opts["encoded"]:
            continue
        image = make_deconvolution(image, psf)
        dataframe.add_channel(image, channel, **opts)
    dataframe.store(outfname)


def run(project: vstarstack.cfg.Project, argv: list):
    infname = argv[0]
    psfname = argv[1]
    outfname = argv[2]
    process(infname, psfname, outfname)
