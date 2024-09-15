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

import vstarstack.library.data
import vstarstack.library.common
import vstarstack.library.image_process.normalize
import vstarstack.library.merge.simple_add
from vstarstack.library.data import DataFrame

def mean(images : vstarstack.library.common.IImageSource) -> DataFrame:
    """Just mean of images"""
    cnt = 0
    for _ in images.items():
        cnt += 1
    image = vstarstack.library.merge.simple_add.simple_add(images)
    for channel in image.channels:
        layer, opts = image.get_channel(channel)
        if not image.get_channel_option(channel, "brightness"):
            continue
        layer /= cnt
        image.replace_channel(layer, channel, **opts)
    return image
