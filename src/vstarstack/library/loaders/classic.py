"""Reading common image files: jpg/png/tiff"""
#
# Copyright (c) 2023-2024 Vladislav Tsendrovskii
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
import exifread

import vstarstack.library.common
import vstarstack.library.data
import vstarstack.library.loaders.tags

def readjpeg(fname: str):
    """Read single image (jpg, png, tiff) file"""
    rgb = np.asarray(Image.open(fname)).astype(np.float32)
    shape = rgb.shape
    shape = (shape[0], shape[1])

    with open(fname, 'rb') as file:
        tags = exifread.process_file(file)

    params = {
        "w": shape[1],
        "h": shape[0],
    }

    if "EXIF ExposureTime" in tags:
        tag = tags["EXIF ExposureTime"]
        params["exposure"] = float(tag.values[0])
    else:
        params["exposure"] = 1

    if "EXIF ISOSpeedRatings" in tags:
        tag = tags["EXIF ISOSpeedRatings"]
        params["gain"] = float(tag.values[0])
    else:
        params["gain"] = 1

    params["weight"] = params["exposure"] * params["gain"]

    dataframe = vstarstack.library.data.DataFrame(params, tags)

    if len(rgb.shape) == 3:
        dataframe.add_channel(rgb[:, :, 0], "R", brightness=True, signal=True)
        dataframe.add_channel(rgb[:, :, 1], "G", brightness=True, signal=True)
        dataframe.add_channel(rgb[:, :, 2], "B", brightness=True, signal=True)
    elif len(rgb.shape) == 2:
        dataframe.add_channel(rgb[:, :], "L", brightness=True, signal=True)
    else:
        # unknown shape!
        pass
    yield dataframe
