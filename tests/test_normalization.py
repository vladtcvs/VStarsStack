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

import os
import tempfile
import numpy as np
from vstarstack.library.data import DataFrame
from vstarstack.library.image_process.normalize import normalize

def test_1():
    dataframe = DataFrame()

    layer = np.ones((100,100))
    weight = np.ones((100,100))
    dataframe.add_channel(layer, "light", brightness=True)
    dataframe.add_channel(weight, "weight", weight=True)
    dataframe.add_channel_link("light", "weight", "weight")

    dataframe2 = normalize(dataframe)
    layer_normed, opts = dataframe2.get_channel("light")
    assert opts["normed"] == True
    assert layer_normed[0,0] == 1

    layer[0,0] = 2
    assert layer_normed[0,0] == 1

def test_2():
    dataframe = DataFrame()

    layer = np.ones((100,100))*2
    weight = np.ones((100,100))*2
    dataframe.add_channel(layer, "light", brightness=True)
    dataframe.add_channel(weight, "weight", weight=True)
    dataframe.add_channel_link("light", "weight", "weight")

    dataframe2 = normalize(dataframe)
    layer_normed, opts = dataframe2.get_channel("light")
    assert opts["normed"] == True
    assert layer_normed[0,0] == 1

    weight_normed, opts = dataframe2.get_channel("weight")
    assert weight_normed[0,0] == 2

    layer[0,0] = 2
    assert layer_normed[0,0] == 1
