#
# Copyright (c) 2024 Vladislav Tsendrovskii
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
import numpy as np
from vstarstack.library.data import DataFrame
from vstarstack.library.image_process.nanmean_filter import *

def test_1():
    df = DataFrame()

    layer = np.ones((100,100))
    
    layer[0,0] = np.nan
    df.add_channel(layer, "light", brightness=True, signal=True)
    fixed = filter_df(df, 1, False)
    fixed_layer = fixed.get_channel("light")[0]
    assert fixed_layer[0,0] == 1

def test_2():
    df = DataFrame()

    layer = np.zeros((100,100))
    
    layer[:,:] = np.nan
    layer[0::2,0::2] = 1
    df.add_channel(layer, "light", brightness=True, signal=True)
    fixed = filter_df(df, 1, False)
    fixed_layer = fixed.get_channel("light")[0]
    assert np.amax(fixed_layer) == 1
    assert np.amin(fixed_layer) == 1
