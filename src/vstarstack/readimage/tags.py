"""Read tags from EXIF"""
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

import exifread

tags_names = {
    "shutter": [("EXIF ExposureTime", 0)],
    "iso": [("EXIF ISOSpeedRatings", 0), ("MakerNote ISOSetting", 1)],
}


def read_tags(filename):
    """Read EXIF tags from file"""
    with open(filename, 'rb') as file:
        tags = exifread.process_file(file)

    res = {}
    for tag_name, tag in tags_names.items():
        for name, variant_id in tags_names[tag_name]:
            if name in tags:
                res[tag_name] = float(tag.values[variant_id])
                break

    print(res)
    return res
