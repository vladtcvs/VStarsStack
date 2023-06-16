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

import math
from vstarstack.library.projection.orthographic import Projection

def equal(val1, val2):
    """Test that values are same"""
    return abs(val1 - val2) < 1e-6

def test_1():
    # sphere
    proj = Projection(100, 100, 50, 50, 0, 0)
    lat, lon = proj.project(50, 50)
    print(lat, lon)
    assert equal(lon, 0)
    assert equal(lat, 0)

def test_2():
    proj = Projection(100, 100, 50, 50, 0, 0)
    lat, lon = proj.project(50, 75)
    print(lat, lon)
    assert equal(lon, math.pi/2)
    assert equal(lat, 0)

def test_3():
    proj = Projection(100, 100, 50, 50, 0, 0)
    lat, lon = proj.project(50, 25)
    print(lat, lon)
    assert equal(lon, -math.pi/2)
    assert equal(lat, 0)

def test_4():
    proj = Projection(100, 100, 50, 50, 0, 0)
    lat, lon = proj.project(75, 50)
    print(lat, lon)
    assert equal(lon, 0)
    assert equal(lat, -math.pi/2)

def test_5():
    proj = Projection(100, 100, 50, 50, 0, 0)
    lat, lon = proj.project(75, 75)
    print(lat, lon)
    assert lon is None
    assert lat is None
