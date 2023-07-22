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

import numpy as np
from vstarstack.library.data import DataFrame
from vstarstack.library.image_process.cut import cut

def test_cut_1():
    h = 200
    w = 200
    orig = DataFrame(params={"w" : w, "h" : h})
    cutted = cut(orig, 50, 100, 100, 150)
    assert cutted.params["w"] == 50
    assert cutted.params["h"] == 50
    assert cutted.params["center_offset_x"] == -25
    assert cutted.params["center_offset_y"] == 25

def test_cut_2():
    h = 200
    w = 200
    orig = DataFrame(params={"w" : w,
                             "h" : h,
                             "center_offset_x" : 25,
                             "center_offset_y" : -25,})
    cutted = cut(orig, 50, 100, 100, 150)
    assert cutted.params["w"] == 50
    assert cutted.params["h"] == 50
    assert cutted.params["center_offset_x"] == 0
    assert cutted.params["center_offset_y"] == 0
