"""Reading common image files: jpg/png/tiff"""
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

import numpy as np
from PIL import Image

import vstarstack.library.common
import vstarstack.library.data
import vstarstack.library.loaders.tags

def readjpeg(fname: str):
    """Read single image (jpg, png, tiff) file"""
    rgb = np.asarray(Image.open(fname)).astype(np.float32)
    shape = rgb.shape
    shape = (shape[0], shape[1])

    tags = vstarstack.library.loaders.tags.read_tags(fname)
    params = {
        "w": shape[1],
        "h": shape[0],
    }

    try:
        exposure = tags["shutter"]*tags["iso"]
    except KeyError as _:
        exposure = 1

    weight = np.ones((shape[0], shape[1]))*exposure

    dataframe = vstarstack.library.data.DataFrame(params, tags)
    dataframe.add_channel(weight, "weight", weight=True)

    if len(rgb.shape) == 3:
        dataframe.add_channel(rgb[:, :, 0], "R", brightness=True, signal=True)
        dataframe.add_channel(rgb[:, :, 1], "G", brightness=True, signal=True)
        dataframe.add_channel(rgb[:, :, 2], "B", brightness=True, signal=True)
        dataframe.add_channel_link("R", "weight", "weight")
        dataframe.add_channel_link("G", "weight", "weight")
        dataframe.add_channel_link("B", "weight", "weight")
    elif len(rgb.shape) == 2:
        dataframe.add_channel(rgb[:, :], "L", brightness=True, signal=True)
        dataframe.add_channel_link("L", "weight", "weight")
    else:
        # unknown shape!
        pass
    yield dataframe
