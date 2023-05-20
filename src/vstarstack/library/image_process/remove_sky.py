"""Remove sky from the image"""
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

import vstarstack.library.sky_model.gradient
import vstarstack.library.sky_model.isoline
import vstarstack.library.sky_model.quadratic

import vstarstack.library.common
import vstarstack.library.data

def remove_sky_with_model(dataframe : vstarstack.library.data.DataFrame, model):
    """Remove sky from the image"""
    for channel in dataframe.get_channels():

        image, opts = dataframe.get_channel(channel)
        if not opts["brightness"]:
            continue

        sky = model(image)
        result = image - sky
        dataframe.add_channel(result, channel, **opts)

    return dataframe

def remove_sky(dataframe : vstarstack.library.data.DataFrame, model_name : str):
    """Remove sky from the image"""
    if model_name == "gradient":
        model = vstarstack.library.sky_model.gradient.model
    elif model_name == "isoline":
        model = vstarstack.library.sky_model.isoline.model
    elif model_name == "quadratic":
        model = vstarstack.library.sky_model.quadratic.model
    else:
        raise Exception("Unknown model")
    remove_sky_with_model(dataframe, model)
