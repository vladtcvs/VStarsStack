"""Reading NEF image files"""
#
# Copyright (c) 2023 Vladislav Tsendrovskii
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

import rawpy
import numpy as np

import vstarstack.library.common
import vstarstack.library.data

import vstarstack.library.loaders.tags


def readnef(filename: str):
    """Read NEF file"""
    img = rawpy.imread(filename)
    image = img.raw_image_visible

    tags = vstarstack.library.loaders.tags.read_tags(filename)

    params = {
        "w": image.data.shape[1],
        "h": image.data.shape[0],
    }

    exptime = tags["shutter"]*tags["iso"]

    dataframe = vstarstack.library.data.DataFrame(params, tags)

    weight = np.ones(image.data.shape)*exptime

    dataframe.add_channel(image, "raw", encoded=True)
    dataframe.add_channel(weight, "weight", weight=True)
    dataframe.add_channel_link("raw", "weight", "weight")

    dataframe.add_parameter("bayerGRBG", "format")
    yield dataframe