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

import sys
import math

import vstarstack.projection.perspective

thr = 1e-6

def test_center_f():
    W = 15
    H = 10
    F = 1000
    w = 1500
    h = 1000

    x = 750
    y = 500

    lon_expected = 0
    lat_expected = 0

    proj = vstarstack.projection.perspective.Projection(W, H, F, w, h)
    lat, lon = proj.project(y, x)
    assert abs(lat - lat_expected) < thr
    assert abs(lon - lon_expected) < thr

def test_center_r():
    W = 15
    H = 10
    F = 1000
    w = 1500
    h = 1000

    lon = 0
    lat = 0

    x_expected = 750
    y_expected = 500

    proj = vstarstack.projection.perspective.Projection(W, H, F, w, h)
    y, x = proj.reverse(lat, lon)
    assert abs(y - y_expected) < thr
    assert abs(x - x_expected) < thr

def test_left_f():
    W = 15
    H = 10
    F = 1000
    w = 1500
    h = 1000

    x = 0
    y = 500

    lon_expected = math.atan(W/2/F)
    lat_expected = 0

    proj = vstarstack.projection.perspective.Projection(W, H, F, w, h)
    lat, lon = proj.project(y, x)
    assert abs(lat - lat_expected) < thr
    assert abs(lon - lon_expected) < thr

def test_left_r():
    W = 15
    H = 10
    F = 1000
    w = 1500
    h = 1000

    x_expected = 0
    y_expected = 500

    lon = math.atan(W/2/F)
    lat = 0

    proj = vstarstack.projection.perspective.Projection(W, H, F, w, h)
    y, x = proj.reverse(lat, lon)

    assert abs(y - y_expected) < thr
    assert abs(x - x_expected) < thr

def test_top_f():
    W = 15
    H = 10
    F = 1000
    w = 1500
    h = 1000

    x = 750
    y = 0

    lon_expected = 0
    lat_expected = math.atan(H/2/F)

    proj = vstarstack.projection.perspective.Projection(W, H, F, w, h)
    lat, lon = proj.project(y, x)
    assert abs(lat - lat_expected) < thr
    assert abs(lon - lon_expected) < thr

def test_top_r():
    W = 15
    H = 10
    F = 1000
    w = 1500
    h = 1000

    x_expected = 750
    y_expected = 0

    lon = 0
    lat = math.atan(H/2/F)

    proj = vstarstack.projection.perspective.Projection(W, H, F, w, h)
    y, x = proj.reverse(lat, lon)

    assert abs(y - y_expected) < thr
    assert abs(x - x_expected) < thr