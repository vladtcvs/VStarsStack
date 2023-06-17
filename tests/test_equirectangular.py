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

import vstarstack.library.projection.equirectangular

thr = 1e-6

def test_center_f():
    w = 1500
    h = 1000

    x = 750
    y = 500

    lon_expected = 0
    lat_expected = 0

    proj = vstarstack.library.projection.equirectangular.Projection(w, h)
    lat, lon = proj.project(y, x)
    assert abs(lat - lat_expected) < thr
    assert abs(lon - lon_expected) < thr


def test_center_r():
    w = 1500
    h = 1000

    lon = 0
    lat = 0

    x_expected = 750
    y_expected = 500

    proj = vstarstack.library.projection.equirectangular.Projection(w, h)
    y, x = proj.reverse(lat, lon)
    assert abs(y - y_expected) < thr
    assert abs(x - x_expected) < thr


def test_left_f():
    w = 1500
    h = 1000

    x = 0
    y = 500

    lon_expected = math.pi
    lat_expected = 0

    proj = vstarstack.library.projection.equirectangular.Projection(w, h)
    lat, lon = proj.project(y, x)
    assert abs(lat - lat_expected) < thr
    assert abs(lon - lon_expected) < thr


def test_left_r():
    w = 1500
    h = 1000

    x_expected = 0
    y_expected = 500

    lon = math.pi
    lat = 0

    proj = vstarstack.library.projection.equirectangular.Projection(w, h)
    y, x = proj.reverse(lat, lon)

    assert abs(y - y_expected) < thr
    assert abs(x - x_expected) < thr


def test_top_f():
    w = 1500
    h = 1000

    x = 750
    y = 0

    lon_expected = 0
    lat_expected = math.pi/2

    proj = vstarstack.library.projection.equirectangular.Projection(w, h)
    lat, lon = proj.project(y, x)
    assert abs(lat - lat_expected) < thr
    assert abs(lon - lon_expected) < thr


def test_top_r():
    w = 1500
    h = 1000

    x_expected = 750
    y_expected = 0

    lon = 0
    lat = math.pi/2

    proj = vstarstack.library.projection.equirectangular.Projection(w, h)
    y, x = proj.reverse(lat, lon)

    assert abs(y - y_expected) < thr
    assert abs(x - x_expected) < thr
