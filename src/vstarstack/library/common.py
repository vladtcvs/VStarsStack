"""Common methods"""
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
import typing

import skimage.color
from skimage import exposure

import cv2
import numpy as np

import abc

import vstarstack.library.data

def getpixel_linear(img, y, x):
    xm = math.floor(x)
    xmc = 1-(x-xm)
    xp = math.ceil(x)
    xpc = 1-(xp-x)
    ym = math.floor(y)
    ymc = 1-(y-ym)
    yp = math.ceil(y)
    ypc = 1-(yp-y)

    c = xmc + xpc
    xmc /= c
    xpc /= c

    c = ymc + ypc
    ymc /= c
    ypc /= c

    if ym < 0 or xm < 0 or xp >= img.shape[1] or yp >= img.shape[0]:
        if len(img.shape) == 2:
            return False, 0
        else:
            return False, np.zeros((img.shape[2],))

    imm = img[ym][xm]
    imp = img[ym][xp]
    ipm = img[yp][xm]
    ipp = img[yp][xp]

    return True, imm * ymc*xmc + imp * ymc*xpc + ipm * ypc*xmc + ipp * ypc*xpc

def df_weight_0_to_nan(df : vstarstack.library.data.DataFrame, in_place : bool = False):
    """Repalce pixels with weight = 0 to nan"""
    if in_place:
        fixed = df
    else:
        fixed = vstarstack.library.data.DataFrame(df.params)
    for channel in df.get_channels():
        layer, opts = df.get_channel(channel)
        if df.get_channel_option(channel, "encoded"):
            continue
        if df.get_channel_option(channel, "weight"):
            continue

        weight, _, _ = df.get_linked_channel(channel, "weight")
        copy = np.copy(layer)
        copy = copy.astype('float32')
        copy[np.where(weight == 0)] = np.nan
        fixed.add_channel(copy, channel, **opts)
    return fixed

def df_weight_0_to_0(df : vstarstack.library.data.DataFrame, in_place : bool = False):
    """Repalce pixels with weight = 0 to nan"""
    if in_place:
        fixed = df
    else:
        fixed = vstarstack.library.data.DataFrame(df.params)
    for channel in df.get_channels():
        layer, opts = df.get_channel(channel)
        if df.get_channel_option(channel, "encoded"):
            continue
        if df.get_channel_option(channel, "weight"):
            fixed.add_channel(layer, channel, **opts)
            continue

        weight, _, _ = df.get_linked_channel(channel, "weight")
        copy = np.copy(layer)
        copy = copy.astype('float32')
        copy[np.where(weight == 0)] = 0
        fixed.add_channel(copy, channel, **opts)
    fixed.links = df.links
    return fixed

def getpixel_none(img, y, x):
    x = round(x)
    y = round(y)

    if y < 0 or x < 0 or x >= img.shape[1] or y >= img.shape[0]:
        if len(img.shape) == 2:
            return False, 0
        else:
            return False, np.zeros((img.shape[2],))

    return True, img[y][x]


def getpixel(img, y, x, interpolate=True):
    if interpolate:
        return getpixel_linear(img, y, x)
    return getpixel_none(img, y, x)




def length(vec):
    return (vec[0]**2+vec[1]**2)**0.5


def norm(vec):
    l = (vec[0]**2+vec[1]**2)**0.5
    return (vec[0] / l, vec[1] / l)


def prepare_image_for_model(image):
    image = image[:, :, 0:3]
    image = skimage.color.rgb2gray(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    am = np.amax(image)
    if am >= 1:
        image /= am
    image = exposure.equalize_hist(image)
    return image

class IImageSource(abc.ABC):
    """Abstract image source"""

    @abc.abstractmethod
    def items(self) -> typing.Generator[vstarstack.library.data.DataFrame, None, None]:
        """Take elements from source"""

    @abc.abstractmethod
    def empty(self) -> bool:
        """Check if there are elements in source"""

class ListImageSource(IImageSource):
    """Get images from list"""
    def __init__(self, images : list[vstarstack.library.data.DataFrame]):
        self.images = images
        self.index = 0

    def items(self) -> typing.Generator[vstarstack.library.data.DataFrame, None, None]:
        """Take next element from source"""
        for item in self.images:
            yield item

    def empty(self) -> bool:
        """Check if there are elements in source"""
        return len(self.images) == 0

class FilesImageSource(IImageSource):
    """Get images from files"""
    def __init__(self, filenames : list[str]):
        self.filenames = filenames

    def items(self) -> typing.Generator[vstarstack.library.data.DataFrame, None, None]:
        """Take next element from source"""
        for fname in self.filenames:
            yield vstarstack.library.data.DataFrame.load(fname)

    def empty(self) -> bool:
        """Check if there are elements in source"""
        return len(self.filenames) == 0
