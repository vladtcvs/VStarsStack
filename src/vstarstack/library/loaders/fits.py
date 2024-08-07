"""Reading FITS files"""
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

import numpy as np
from astropy.io import fits

import vstarstack.library.data

def readfits(filename: str):
    """Read fits image file"""

    images = fits.open(filename)

    for plane in images:
        tags = {}
        for key in plane.header:
            val = str(plane.header[key])
            tags[key] = val

        shape = plane.data.shape
        if len(shape) == 2:
            original = plane.data.reshape((1, shape[0], shape[1]))
        else:
            original = plane.data
        shape = original.shape

        params = {
            "w": shape[2],
            "h": shape[1],
        }
        if "DATE-OBS" in tags:
            params["UTC"] = tags["DATE-OBS"]

        dataframe = vstarstack.library.data.DataFrame(params, tags)

        exptime = plane.header["EXPTIME"]

        slice_names = []

        weight_channel_name = "weight"
        weight = np.ones((shape[1], shape[2]))*exptime
        dataframe.add_channel(weight, weight_channel_name, weight=True)

        if shape[0] == 1:
            if "FILTER" in plane.header:
                channel_name = plane.header["FILTER"].strip()
            else:
                channel_name = "Y"
            slice_names.append(channel_name)
        elif shape[0] == 3:
            slice_names.append('R')
            slice_names.append('G')
            slice_names.append('B')
        else:
            print("Unknown image format, skip")
            yield None

        for i, slice_name in enumerate(slice_names):
            dataframe.add_channel(original[i, :, :], slice_name, brightness=True, signal=True)
            dataframe.add_channel_link(slice_name, weight_channel_name, "weight")

        yield dataframe
