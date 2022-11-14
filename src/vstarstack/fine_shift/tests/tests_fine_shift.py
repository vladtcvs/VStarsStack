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

import vstarstack.fine_shift.image_wave

N = 2000
dh = 0.1

def test_identity():
    wave = vstarstack.fine_shift.image_wave.ImageWave(10, 10, 2, 2)
    x,y = wave.interpolate(0,0)
    assert x==0 and y==0
    x,y = wave.interpolate(10,10)
    assert x==10 and y==10

def test_approximate_identity():
    wave = vstarstack.fine_shift.image_wave.ImageWave(10, 10, 2, 2)
    targets = [(5,5)]
    measured = [(5,5)]
    wave.approximate(targets, measured, N, dh)
    x,y = wave.interpolate(5,5)

    assert abs(x-5) < 5e-3
    assert abs(y-5) < 5e-3

def test_approximate_single():
    wave = vstarstack.fine_shift.image_wave.ImageWave(10.0, 10.0, 2, 2)
    targets = [(5.0,5.0)]
    measured = [(5.2,5.0)]
    wave.approximate(targets, measured, N, dh)
    x,y = wave.interpolate(5.2,5.0)

    assert abs(x-5.0) < 5e-3
    assert abs(y-5.0) < 5e-3

def test_approximate_parabola():
    wave = vstarstack.fine_shift.image_wave.ImageWave(10, 10, 3, 3)
    targets = [(5,0),(5,5),(5,10)]
    measured = [(5,0),(5.1,5),(5,10)]
    wave.approximate(targets, measured, N, dh)
    x,y = wave.interpolate(5.1,5)

    assert abs(x-5) < 5e-3
    assert abs(y-5) < 5e-3

def test_approximate_parabola2():
    wave = vstarstack.fine_shift.image_wave.ImageWave(10, 10, 3, 3)
    targets = [(5,0),(5,5),(5,10)]
    measured = [(5,0),(5.1,5.1),(5,10)]
    wave.approximate(targets, measured, N, dh)
    x,y = wave.interpolate(5.1,5.1)

    assert abs(x-5) < 5e-3
    assert abs(y-5) < 5e-3


def test_approximate_parabola_long():
    wave = vstarstack.fine_shift.image_wave.ImageWave(10.0, 10.0, 30, 30)
    targets = [(5,0),(5,5),(5,10)]
    measured = [(5,0),(5.1,5.1),(5,10)]
    wave.approximate(targets, measured, N, dh)
    x,y = wave.interpolate(5.1,5.1)

    assert abs(x-5) < 5e-3
    assert abs(y-5) < 5e-3

def test_serialize():
    wave = vstarstack.fine_shift.image_wave.ImageWave(10, 10, 3, 3)
    data = wave.data()
    assert type(data["w"]) == float
    assert type(data["h"]) == float
    assert type(data["Nw"]) == int
    assert type(data["Nh"]) == int
    assert data["w"] == 10
    assert data["h"] == 10
    assert data["Nw"] == 3
    assert data["Nh"] == 3
    assert len(data["data"]) == 3*3*2
