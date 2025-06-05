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

import vstarstack.library.data
from vstarstack.library.loaders.datatype import check_datatype

def readjpeg(fname: str):
    """Read single image (jpg, png, tiff) file"""
    rgb = np.asarray(Image.open(fname))
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

    max_value = np.iinfo(rgb.dtype).max

    dataframe = vstarstack.library.data.DataFrame(params, tags)

    if len(rgb.shape) == 3:
        for channel_name, channel_index in [("R",0), ("G", 1), ("B", 2)]:
            data = rgb[:,:,channel_index]
            dataframe.add_channel(check_datatype(data), channel_name, brightness=True, signal=True)
            overlight_idx = np.where(data >= max_value*0.99)
            if len(overlight_idx) > 0:
                weight = np.ones(data.shape)*params["weight"]
                weight[overlight_idx] = 0
                dataframe.add_channel(weight, f"weight-{channel_name}", weight=True)
                dataframe.add_channel_link(channel_name, f"weight-{channel_name}", "weight")

    elif len(rgb.shape) == 2:
        dataframe.add_channel(check_datatype(rgb), "L", brightness=True, signal=True)
        overlight_idx = np.where(rgb >= max_value*0.99)
        if len(overlight_idx) > 0:
            weight = np.ones(rgb.shape)*params["weight"]
            weight[overlight_idx] = 0
            dataframe.add_channel(weight, f"weight-L", weight=True)
            dataframe.add_channel_link("L", f"weight-L", "weight")
    else:
        # unknown shape!
        pass
    yield dataframe
