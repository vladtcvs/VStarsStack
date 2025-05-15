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
        for channel in no_stars_dataframe.get_channels():
            layer, opts = no_stars_dataframe.get_channel(channel)
            if not no_stars_dataframe.get_channel_option(channel, "signal"):
                continue
            layer = cv2.GaussianBlur(layer, (15, 15), 0)
            layer = layer / np.median(layer)
            no_stars_dataframe.replace_channel(layer, channel, **opts)
    
        no_star_images.append(no_stars_dataframe)
    
    no_star_source = vstarstack.library.common.ListImageSource(no_star_images)
    flat = vstarstack.library.merge.kappa_sigma.kappa_sigma(no_star_source, 1, 1, 2)
    flat = vstarstack.library.image_process.normalize.normalize(flat, False)
    for channel in flat.get_channels():
        layer, opts = flat.get_channel(channel)
        if not flat.get_channel_option(channel, "signal"):
            continue
        layer = cv2.GaussianBlur(layer, (15, 15), 0)
        layer = layer / np.amax(layer)
        flat.replace_channel(layer, channel, **opts)
    for channel in list(flat.get_channels()):
        if not flat.get_channel_option(channel, "weight"):
            continue
        flat.remove_channel(channel)
    return flat

def approximate_flat(image : np.ndarray) -> np.ndarray:
    """Find smooth polynomial approximation of flat"""
    w = image.shape[1]
    h = image.shape[0]
    fft = np.fft.fft2(image)
    fft = np.fft.fftshift(fft)
    c = np.zeros((h, w))
    c[int(h/2), int(w/2)] = 1
    c = cv2.GaussianBlur(c, (9, 9), 0)
    c = c / np.amax(c)
    fft = fft * c
    fft = np.fft.ifftshift(fft)
    flat = np.fft.ifft2(fft)
    flat = flat / np.amax(flat)
    return flat

def detect_spots(image : np.ndarray, approximated : np.ndarray) -> np.ndarray:
    """Detect spots on original flat and append them to approximated flat"""
    return approximated

def approximate_flat_image(flat : vstarstack.library.data.DataFrame) -> vstarstack.library.data.DataFrame:
    """Approximate flat"""
    for channel in flat.get_channels():
        if not flat.get_channel_option(channel, "signal"):
            continue
        layer,opts = flat.get_channel(channel)
        layer = layer.astype(np.float64)
        layer_approximated = approximate_flat(layer)
        layer_approximated = detect_spots(layer, layer_approximated)
        layer_approximated = layer_approximated / np.amax(layer_approximated)
        flat.replace_channel(layer_approximated, channel, **opts)
    return flat
