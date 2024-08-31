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

import cv2
import numpy as np

import vstarstack.library.common
import vstarstack.library.data
import vstarstack.library.merge
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
    dataframe = vstarstack.library.merge.simple_add(source)
    return dataframe

def calculate_median(image, weight, smooth_size):
    """Apply median filter with mask (weight > 0)"""
    radius = int(smooth_size/2)
    idxs = np.where(weight == 0)
    image[idxs] = np.nan
    image = nanmean_filter(image, radius)
    return image

class NoStarSource(vstarstack.library.common.IImageSource):
    """Images without stars"""
    def __init__(self,
                 src : vstarstack.library.common.IImageSource,
                 stars : list):
        self.src = src
        self.stars = stars

    def items(self) -> vstarstack.library.data.DataFrame:
        """Iterate images"""
        for index, image in enumerate(self.src.items()):
            image = vstarstack.library.stars.cut.cut_stars(image, self.stars[index])
            yield image

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
