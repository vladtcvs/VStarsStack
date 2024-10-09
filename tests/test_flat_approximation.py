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
import vstarstack.library.calibration.flat
import vstarstack.library.loaders.classic
import os
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

def test_flat():
    dfg = vstarstack.library.loaders.classic.readjpeg(os.path.join(dir_path, 'flat/flat.jpg'))
    df = next(dfg)
    flat,_ = df.get_channel('L')
    approx = vstarstack.library.calibration.flat.approximate_flat_image(df.copy())
    flat_approx,_ = approx.get_channel('L')
    assert flat_approx.shape == flat.shape
    rel = flat_approx / flat
    assert np.average(rel) < 1.05
    assert np.average(rel) > 0.95
    assert np.std(rel) < 0.05
