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

import vstarstack.library.data
import vstarstack.library.merging
import vstarstack.library.image_process.normalize

def flatten(dataframe : vstarstack.library.data.DataFrame,
            flat : vstarstack.library.data.DataFrame):
    """Apply flattening"""
    for channel in dataframe.get_channels():
        image, opts = dataframe.get_channel(channel)
        if not opts["brightness"]:
            continue

        if channel in flat.get_channels():
            image = image / flat.get_channel(channel)[0]

        dataframe.replace_channel(image, channel)
    return dataframe

def prepare_flat_simple(fnames : list[str]) -> vstarstack.library.data.DataFrame:
    """Prepare flat files for processing"""
    flat = vstarstack.library.merging.simple_add(fnames)
    return vstarstack.library.image_process.normalize.normalize(flat)

def prepare_flat_sky(fnames : list[str], sigma_k=1) -> vstarstack.library.data.DataFrame:
    """Generate flat image"""
    flat = vstarstack.library.merging.sigma_clip(fnames, sigma_k, 5)
    return vstarstack.library.image_process.normalize.normalize(flat)
