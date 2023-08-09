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
import numpy as np

from vstarstack.library.loaders.classic import readjpeg
import vstarstack.library.fine_shift.image_wave
from vstarstack.library.fine_shift.image_wave import ImageWave

dir_path = os.path.dirname(os.path.realpath(__file__))
N = 200
dh = 0.1

def test_identity():
    wave = ImageWave(10, 10, 2, 2, 0.01)
    x, y = wave.interpolate(0, 0)
    assert x == 0 and y == 0
    x, y = wave.interpolate(10, 10)
    assert x == 10 and y == 10


def test_approximate_identity():
    wave = ImageWave(10, 10, 2, 2, 0.01)
    targets = [(5, 5)]
    points = [(5, 5)]
    wave.approximate_by_targets(targets, points, N, dh)
    x, y = wave.interpolate(5, 5)

    assert abs(x-5) < 5e-3
    assert abs(y-5) < 5e-3


def test_approximate_single():
    wave = ImageWave(10, 10, 2, 2, 0.01)
    targets = [(5.0, 5.0)]
    points = [(5.2, 5.0)]
    wave.approximate_by_targets(targets, points, N, dh)
    x, y = wave.interpolate(5.2, 5.0)

    assert abs(x-5.0) < 5e-3
    assert abs(y-5.0) < 5e-3


def test_approximate_parabola1():
    wave = ImageWave(10, 10, 3, 3, 0.01)
    targets = [(5, 0), (5, 5), (5, 10)]
    points = [(5, 0), (5.1, 5), (5, 10)]
    wave.approximate_by_targets(targets, points, N, dh)
    x, y = wave.interpolate(5.1, 5)

    assert abs(x-5) < 5e-3
    assert abs(y-5) < 5e-3


def test_approximate_parabola2():
    wave = ImageWave(10, 10, 3, 3, 0.01)
    targets = [(5, 0), (5, 5), (5, 10)]
    measured = [(5, 0), (5.1, 5.1), (5, 10)]
    wave.approximate_by_targets(targets, measured, N, dh)
    x, y = wave.interpolate(5.1, 5.1)

    assert abs(x-5) < 5e-3
    assert abs(y-5) < 5e-3


def test_approximate_parabola_long():
    wave = ImageWave(10, 10, 30, 30, 0.01)
    targets = [(5, 0), (5, 5), (5, 10)]
    measured = [(5, 0), (5.1, 5.1), (5, 10)]
    wave.approximate_by_targets(targets, measured, N, dh)
    x, y = wave.interpolate(5.1, 5.1)

    assert abs(x-5) < 5e-3
    assert abs(y-5) < 5e-3


def test_serialize():
    wave = ImageWave(10, 10, 3, 3, 0.01)
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
    wave = ImageWave(10, 10, 2, 2, 0.01)
    targets = [(5.0, 5.0)]
    measured = [(5.2, 5.0)]
    wave.approximate_by_targets(targets, measured, N, dh)
    x, y = wave.interpolate(5.2, 5.0)

    assert abs(x-5.0) < 5e-3
    assert abs(y-5.0) < 5e-3

    data = wave.data()
    wave2 = ImageWave.from_data(data)

    assert wave2 is not None

    wave2.approximate_by_targets(targets, measured, N, dh)
    x, y = wave2.interpolate(5.2, 5.0)

    assert abs(x-5.0) < 5e-3
    assert abs(y-5.0) < 5e-3

def compare_shift_array(array, reference):
    assert len(array) == len(reference)
    print(array, reference)
    for i, v in enumerate(reference):
        assert abs(v - array[i]) < 1e-3

def test_correlation1():
    df1 = next(readjpeg(os.path.join(dir_path, "fine_shift/image1.png")))
    image1 = df1.get_channel("L")[0].astype('double')
    correlation = vstarstack.library.fine_shift.image_wave.image_correlation(image1, image1)
    assert correlation == 1

def test_correlation2():
    df1 = next(readjpeg(os.path.join(dir_path, "fine_shift/image1.png")))
    df2 = next(readjpeg(os.path.join(dir_path, "fine_shift/image2.png")))

    image1 = df1.get_channel("L")[0].astype('double')
    image2 = df2.get_channel("L")[0].astype('double')

    image1_moved = np.zeros(image1.shape)
    for y in range(image1.shape[0]):
        for x in range(1,image1.shape[1]):
            image1_moved[y,x] = image1[y,x-1]
        image1_moved[y,0] = np.nan

    correlation = vstarstack.library.fine_shift.image_wave.image_correlation(image1_moved, image2)
    assert correlation == 1

def test_shift_image1():
    df1 = next(readjpeg(os.path.join(dir_path, "fine_shift/image1.png")))

    image1 = df1.get_channel("L")[0].astype('double')
    wave = ImageWave(image1.shape[1], image1.shape[0], 2, 2, 0.01)
    image2 = wave.apply_shift(image1, 1)
    correlation = vstarstack.library.fine_shift.image_wave.image_correlation(image1, image2)
    assert correlation == 1

def test_approximate_by_correlation1():
    df1 = next(readjpeg(os.path.join(dir_path, "fine_shift/image1.png")))
    df2 = next(readjpeg(os.path.join(dir_path, "fine_shift/image2.png")))

    image = df1.get_channel("L")[0].astype('double')
    image_ref = df2.get_channel("L")[0].astype('double')
    wave = ImageWave.find_shift_array(image, None, image_ref, None, 5, 3, 4)
    image_shifted = wave.apply_shift(image, 1)
    correlation = vstarstack.library.fine_shift.image_wave.image_correlation(image_shifted, image_ref)
    assert correlation > 1 - 1e-4
