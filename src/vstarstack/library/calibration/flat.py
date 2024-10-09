#
# Copyright (c) 2022-2024 Vladislav Tsendrovskii
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
from typing import Tuple
import cv2
import numpy as np
import scipy.signal

import vstarstack.library.common
import vstarstack.library.data
import vstarstack.library.merge
import vstarstack.library.merge.simple_add
import vstarstack.library.stars.detect
import vstarstack.library.stars.cut
import vstarstack.library.image_process.blur
import vstarstack.library.merge.kappa_sigma
import vstarstack.library.image_process.normalize

from vstarstack.library.image_process.blur import BlurredSource
from vstarstack.library.image_process.nanmean_filter import nanmean_filter
from vstarstack.library.calibration.removehot import remove_hot_pixels

def flatten(dataframe : vstarstack.library.data.DataFrame,
            flat : vstarstack.library.data.DataFrame):
    """Apply flattening"""
    for channel in dataframe.get_channels():
        image, opts = dataframe.get_channel(channel)
        if not opts["brightness"]:
            continue

        if channel in flat.get_channels():
            image = image / flat.get_channel(channel)[0]

        dataframe.replace_channel(image, channel)
    return dataframe

def prepare_flat_simple(images : vstarstack.library.common.IImageSource,
                        smooth_size : int
                        ) -> vstarstack.library.data.DataFrame:
    """Prepare flat files for processing"""
    if smooth_size % 2 == 0:
        smooth_size += 1

    source = BlurredSource(images, smooth_size)
    dataframe = vstarstack.library.merge.simple_add.simple_add(source)
    return dataframe

def calculate_median(image, weight, smooth_size):
    """Apply median filter with mask (weight > 0)"""
    radius = int(smooth_size/2)
    idxs = np.where(weight == 0)
    image[idxs] = np.nan
    image = nanmean_filter(image, radius)
    return image

def prepare_flat_sky(images : vstarstack.library.common.IImageSource,
                     smooth_size : int
                     ) -> vstarstack.library.data.DataFrame:
    """Generate flat image"""
    no_star_images = []
    for dataframe in images.items():
        descs = []
        for name in dataframe.get_channels():
            layer, opts = dataframe.get_channel(name)
            if not opts["brightness"]:
                continue
            channel_descs = vstarstack.library.stars.detect.detect_stars(layer)
            descs += channel_descs

        no_stars_dataframe = vstarstack.library.stars.cut.cut_stars(dataframe, descs)
        no_stars_dataframe = remove_hot_pixels(no_stars_dataframe)
        no_star_images.append(no_stars_dataframe)
    
    no_star_source = vstarstack.library.common.ListImageSource(no_star_images)
    flat = vstarstack.library.merge.kappa_sigma.kappa_sigma(no_star_source, 1, 1, 2)
    flat = vstarstack.library.image_process.normalize.normalize(flat, False)
    for channel in flat.get_channels():
        layer, opts = flat.get_channel(channel)
        if not flat.get_channel_option(channel, "signal"):
            continue
        layer = cv2.GaussianBlur(layer, (15, 15), 0)
        flat.add_channel(layer, channel, **opts)
    for channel in list(flat.get_channels()):
        if not flat.get_channel_option(channel, "weight"):
            continue
        flat.remove_channel(channel)
    return flat


def get_quarters(image : np.ndarray, x : int, y : int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get quarters of the image"""
    w = image.shape[1]
    h = image.shape[0]
    left_top = image[0:y,0:x]
    left_bottom = image[y:h,0:x]
    right_top = image[0:y,x:w]
    right_bottom = image[y:h,x:w]
    left_top = left_top[::-1,::-1]
    left_bottom = left_bottom[:,::-1]
    right_top = right_top[::-1,:]
    minw = min(left_top.shape[1], left_bottom.shape[1], right_top.shape[1], right_bottom.shape[1])
    minh = min(left_top.shape[0], left_bottom.shape[0], right_top.shape[0], right_bottom.shape[0])
    left_top = left_top[0:minh,0:minw]
    left_bottom = left_bottom[0:minh,0:minw]
    right_top = right_top[0:minh,0:minw]
    right_bottom = right_bottom[0:minh,0:minw]
    return left_top, left_bottom, right_top, right_bottom

def correlate_images(im1 : np.ndarray, im2 : np.ndarray) -> float:
    """Find correlation coefficient between 2 images"""
    im1 = im1 - np.mean(im1)
    im2 = im2 - np.mean(im2)
    top = np.sum(im1*im2)
    bottom1 = np.sum(im1**2)
    bottom2 = np.sum(im2**2)
    return top / math.sqrt(bottom1 * bottom2)

def approximate_flat(image : np.ndarray) -> Tuple[int, int, float, float, float]:
    """Find smooth polynomial approximation of flat"""
    # We need to find center of vignetting and it's parameters
    # We fill find them with gradient descend
    w = image.shape[1]
    h = image.shape[0]

    x0, y0 = int(w/2), int(h/2)
    corrmax = -1

    centers = [(int(h/2), int(w/2))]
    for y in range(int(h/3),int(2*h/3)+1):
        for x in range(int(w/3),int(2*w/3)+1):
            centers.append((y,x))

    image = np.clip(image, 0, None)
    for y,x in centers:
        p1,p2,p3,p4 = get_quarters(image, x, y)
        c12 = correlate_images(p1, p2)
        c13 = correlate_images(p1, p3)
        c14 = correlate_images(p1, p4)
        c23 = correlate_images(p2, p3)
        c24 = correlate_images(p2, p4)
        c34 = correlate_images(p3, p3)
        corr = sum([c12, c13, c14, c23, c24, c34])/6
        if corr > corrmax:
            corrmax = corr
            x0,y0 = x,y

    p1,p2,p3,p4 = get_quarters(image, x0, y0)
    ww = p1.shape[1]
    hh = p1.shape[0]
    
    sw = int(ww/16)
    sh = int(hh/16)

    val0 = np.average([np.average(item[:sh,:sw]) for item in [p1,p2,p3,p4]])
    val_x = np.average([np.average(item[:sh,ww-sw:ww]) for item in [p1,p2,p3,p4]])/val0
    val_y = np.average([np.average(item[hh-sh:hh,:sw]) for item in [p1,p2,p3,p4]])/val0

    k_x = (1-val_x)/ww**2
    k_y = (1-val_y)/hh**2
    return x0, y0, val0, k_x, k_y

def generate_flat(w : int, h : int, x0 : int, y0 : int, val0 : float, k_x : float, k_y : float) -> np.ndarray:
    """Generate flat from parameters"""
    x, y = np.meshgrid(np.arange(0,w),np.arange(0,h))
    approx = val0*(1-k_x*(x-x0)**2-k_y*(y-y0)**2)
    return approx

def detect_spots(image : np.ndarray, approximated : np.ndarray) -> np.ndarray:
    """Detect spots on original flat and append them to approximated flat"""
    return image

def approximate_flat_image(flat : vstarstack.library.data.DataFrame) -> vstarstack.library.data.DataFrame:
    """Approximate flat"""
    for channel in flat.get_channels():
        if not flat.get_channel_option(channel, "signal"):
            continue
        layer,opts = flat.get_channel(channel)
        layer = layer.astype(np.float64)
        h,w = layer.shape
        minv = min(w,h)
        k = 128/minv
        h = int(h*k)
        w = int(w*k)
        layer_small = cv2.resize(layer, (w,h), interpolation=cv2.INTER_LINEAR)
        x0, y0, val0, kx, ky = approximate_flat(layer_small)
        x0 = x0 / k
        y0 = y0 / k
        kx = kx * k**2
        ky = ky * k**2
        layer_approximated = generate_flat(layer.shape[1], layer.shape[0], x0, y0, val0, kx, ky)
        layer_approximated = detect_spots(layer, layer_approximated)
        flat.replace_channel(layer_approximated, channel, **opts)
    return flat
