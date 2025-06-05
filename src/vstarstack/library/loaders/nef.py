"""Reading NEF image files"""
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

import rawpy
import exifread

import vstarstack.library.data


def readnef(filename: str):
    """Read NEF file"""
    img = rawpy.imread(filename)
    image = img.raw_image_visible
    pattern = img.raw_pattern
    color_desc = img.color_desc

    pattern = [pattern[0,0], pattern[0,1], pattern[1,0], pattern[1,1]]
    bayer = "bayer_2_2_" + "".join([color_desc.decode('ascii')[index] for index in pattern])

    with open(filename, 'rb') as file:
        tags = exifread.process_file(file)

    params = {
        "w": image.data.shape[1],
        "h": image.data.shape[0],
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

    printable_tags = {}
    for tag_name in tags:
        printable_tags[tag_name] = tags[tag_name].printable

    dataframe = vstarstack.library.data.DataFrame(params, printable_tags)
    dataframe.add_channel(image, "raw", encoded=True, brightness=True, signal=True)
    dataframe.add_parameter(bayer, "format")
    yield dataframe
