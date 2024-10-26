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

import numpy as np
import vstarstack.library.common
import vstarstack.library.calibration.dark
from vstarstack.library.data import DataFrame

def test_prepare_dark_1():
    image = np.zeros((100,100))
    df = DataFrame({"exposure" : 1, "gain" : 1, "temperature" : 0}, {})
    df.add_channel(image, "L", brightness=True, signal=True)
    src = vstarstack.library.common.ListImageSource([df])
    darks = vstarstack.library.calibration.dark.prepare_darks(src, 0, 1)
    assert len(darks) == 1
