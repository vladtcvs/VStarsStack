"""Reading FITS files"""
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

import logging
from astropy.io import fits

import vstarstack.library.data
logger = logging.getLogger(__name__)

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

        if "EXPTIME" in plane.header:
            params["exposure"] = float(plane.header["EXPTIME"])
        else:
            params["exposure"] = 1

        if "GAIN" in plane.header:
            params["gain"] = float(plane.header["GAIN"])
        else:
            params["gain"] = 1

        if "CCD-TEMP" in plane.header:
            params["temperature"] = float(plane.header["CCD-TEMP"])

        slice_names = []

        params["weight"] = params["exposure"]*params["gain"]
        dataframe = vstarstack.library.data.DataFrame(params, tags)

        if shape[0] == 1:
            if "FILTER" in plane.header:
                channel_name = plane.header["FILTER"].strip()
            else:
                channel_name = "L"
            slice_names.append(channel_name)
        elif shape[0] == 3:
            slice_names.append('R')
            slice_names.append('G')
            slice_names.append('B')
        else:
            logger.error(f"Unknown image format with shape {shape}, skip")
            yield None

        for i, slice_name in enumerate(slice_names):
            dataframe.add_channel(original[i, :, :], slice_name, brightness=True, signal=True)

        yield dataframe
