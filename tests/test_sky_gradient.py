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

import os
import numpy as np
import matplotlib.pyplot as plt

import vstarstack.library.loaders.classic
import vstarstack.library.image_process.remove_sky

dir_path = os.path.dirname(os.path.realpath(__file__))


def test_gradient_1():
    fname = os.path.join(dir_path, "gradient.png")
    df = next(vstarstack.library.loaders.classic.readjpeg(fname))
    layer = df.get_channel("L")[0]
    maxv = np.amax(layer)

    vstarstack.library.image_process.remove_sky.remove_sky(df, "gradient")
    layer = df.get_channel("L")[0]
    maxvf = np.amax(layer)

    assert maxvf <= maxv / 10
