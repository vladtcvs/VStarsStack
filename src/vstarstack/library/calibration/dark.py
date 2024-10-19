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
import vstarstack.library.merge.simple_mean

def remove_dark(dataframe : vstarstack.library.data.DataFrame,
                dark : vstarstack.library.data.DataFrame):
    """Remove dark from image"""
    dark_channel_name = None
    if "L" in dark.get_channels():
        dark_channel_name = "L"
    else:
        for channel in dark.get_channels():
            if dark.get_channel_option(channel, "brightness"):
                dark_channel_name = channel
                break

    if dark_channel_name is None:
        print("Can not find brightness channel, skip")
        return None

    for channel in dataframe.get_channels():
        image, opts = dataframe.get_channel(channel)
        if not opts["brightness"]:
            print(f"Skipping {channel}")
            continue

        if channel in dark.get_channels():
            dark_layer, _ = dark.get_channel(channel)
        else:
            dark_layer, _ = dark.get_channel(dark_channel_name)

        image = image - dark_layer
        dataframe.replace_channel(image, channel, **opts)
    return dataframe

def prepare_darks(images : vstarstack.library.common.IImageSource
                  ) -> vstarstack.library.data.DataFrame:
    """Build dark frame"""
    return vstarstack.library.merge.simple_mean.mean(images)
