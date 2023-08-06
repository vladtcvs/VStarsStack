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

from vstarstack.library.loaders.classic import readjpeg
import vstarstack.library.fine_shift.image_wave

dir_path = os.path.dirname(os.path.realpath(__file__))
N = 200
dh = 0.1


def test_identity():
    wave = vstarstack.library.fine_shift.image_wave.ImageWave(10, 10, 2, 2, 0.01)
    x, y = wave.interpolate(0, 0)
    assert x == 0 and y == 0
    x, y = wave.interpolate(10, 10)
    assert x == 10 and y == 10


def test_approximate_identity():
    wave = vstarstack.library.fine_shift.image_wave.ImageWave(10, 10, 2, 2, 0.01)
    targets = [(5, 5)]
    points = [(5, 5)]
    wave.approximate_by_targets(targets, points, N, dh)
    x, y = wave.interpolate(5, 5)

    assert abs(x-5) < 5e-3
    assert abs(y-5) < 5e-3


def test_approximate_single():
    wave = vstarstack.library.fine_shift.image_wave.ImageWave(10, 10, 2, 2, 0.01)
    targets = [(5.0, 5.0)]
    points = [(5.2, 5.0)]
    wave.approximate_by_targets(targets, points, N, dh)
    x, y = wave.interpolate(5.2, 5.0)

    assert abs(x-5.0) < 5e-3
    assert abs(y-5.0) < 5e-3


def test_approximate_parabola1():
    wave = vstarstack.library.fine_shift.image_wave.ImageWave(10, 10, 3, 3, 0.01)
    targets = [(5, 0), (5, 5), (5, 10)]
    points = [(5, 0), (5.1, 5), (5, 10)]
    wave.approximate_by_targets(targets, points, N, dh)
    x, y = wave.interpolate(5.1, 5)

    assert abs(x-5) < 5e-3
    assert abs(y-5) < 5e-3


def test_approximate_parabola2():
    wave = vstarstack.library.fine_shift.image_wave.ImageWave(10, 10, 3, 3, 0.01)
    targets = [(5, 0), (5, 5), (5, 10)]
    measured = [(5, 0), (5.1, 5.1), (5, 10)]
    wave.approximate_by_targets(targets, measured, N, dh)
    x, y = wave.interpolate(5.1, 5.1)

    assert abs(x-5) < 5e-3
    assert abs(y-5) < 5e-3


def test_approximate_parabola_long():
    wave = vstarstack.library.fine_shift.image_wave.ImageWave(10, 10, 30, 30, 0.001)
    targets = [(5, 0), (5, 5), (5, 10)]
    measured = [(5, 0), (5.1, 5.1), (5, 10)]
    wave.approximate_by_targets(targets, measured, N, dh)
    x, y = wave.interpolate(5.1, 5.1)

    assert abs(x-5) < 5e-3
    assert abs(y-5) < 5e-3

def compare_shift_array(array, reference):
    assert len(array) == len(reference)
    print(array, reference)
    for i, v in enumerate(reference):
        assert abs(v - array[i]) < 1e-3

def test_fine_shift_by_correlation1():
    df1 = next(readjpeg(os.path.join(dir_path, "fine_shift/image1.png")))
    df2 = next(readjpeg(os.path.join(dir_path, "fine_shift/image2.png")))

    w = df1.params["w"]
    h = df1.params["h"]
    wave = vstarstack.library.fine_shift.image_wave.ImageWave(w, h, 2, 2, 0)

    image1 = df1.get_channel("L")[0].astype('double')
    image2 = df2.get_channel("L")[0].astype('double')
    wave.approximate_by_correlation(image1, image2, 0, 0.2)
    data = wave.data()
    print(data)
    compare_shift_array(data["data"], [-1, 0, -1, 0, -1, 0, -1, 0])

test_fine_shift_by_correlation1()

def test_serialize():
    wave = vstarstack.library.fine_shift.image_wave.ImageWave(10, 10, 3, 3, 0.01)
    data = wave.data()
    assert type(data["w"]) == int
    assert type(data["h"]) == int
    assert type(data["Nw"]) == int
    assert type(data["Nh"]) == int
    assert data["w"] == 10
    assert data["h"] == 10
    assert data["Nw"] == 3
    assert data["Nh"] == 3
    assert len(data["data"]) == 3*3*2


def test_deserialize():
    wave = vstarstack.library.fine_shift.image_wave.ImageWave(10, 10, 2, 2, 0.01)
    targets = [(5.0, 5.0)]
    measured = [(5.2, 5.0)]
    wave.approximate_by_targets(targets, measured, N, dh)
    x, y = wave.interpolate(5.2, 5.0)

    assert abs(x-5.0) < 5e-3
    assert abs(y-5.0) < 5e-3

    data = wave.data()
    wave2 = vstarstack.library.fine_shift.image_wave.ImageWave.from_data(data)

    assert wave2 is not None

    wave2.approximate_by_targets(targets, measured, N, dh)
    x, y = wave2.interpolate(5.2, 5.0)

    assert abs(x-5.0) < 5e-3
    assert abs(y-5.0) < 5e-3
