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

import cv2
import vstarstack.library.data
import vstarstack.library.common

def blur(dataframe : vstarstack.library.data.DataFrame, size : int):
    """Gaussian blur"""

    for channel in dataframe.get_channels():
        image, opts = dataframe.get_channel(channel)
        if opts["encoded"]:
            continue
        if opts["weight"]:
            continue

        w_channel = dataframe.links["weight"][channel]
        weight, _ = dataframe.get_channel(w_channel)

        image = cv2.GaussianBlur(image, (size, size), 0)
        weight = cv2.GaussianBlur(weight, (size, size), 0)

        dataframe.replace_channel(image, channel)
        dataframe.replace_channel(weight, w_channel)

    return dataframe

class BlurredSource(vstarstack.library.common.IImageSource):
    """Blurred image source"""
    def __init__(self, src : vstarstack.library.common.IImageSource, size : int):
        if size % 2 == 0:
            size += 1
        self.size = size
        self.src = src

    def items(self) -> vstarstack.library.data.DataFrame:
        """Take next element"""
        for dataframe in self.src.items():
            dataframe = blur(dataframe, self.size)
            yield dataframe
