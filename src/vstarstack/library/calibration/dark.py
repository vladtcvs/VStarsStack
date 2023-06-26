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

import vstarstack.library.common
import vstarstack.library.data
import vstarstack.library.merging

def remove_dark(dataframe : vstarstack.library.data.DataFrame,
                dark : vstarstack.library.data.DataFrame):
    """Remove dark from image"""
    for channel in dataframe.get_channels():
        image, opts = dataframe.get_channel(channel)
        if not opts["brightness"]:
            continue

        if channel in dark.get_channels():
            image = image - dark.get_channel(channel)[0]

        dataframe.replace_channel(image, channel)
    return dataframe

def prepare_darks(images : vstarstack.library.common.IImageSource
                  ) -> vstarstack.library.data.DataFrame:
    """Build dark frame"""
    return vstarstack.library.merging.mean(images)
