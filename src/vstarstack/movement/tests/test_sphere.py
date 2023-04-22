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

import vstarstack.projection.perspective
import sphere
import sys
import math
sys.path.append('../')
sys.path.append('../../projection')


thr = 1e-6
pixthr = 1e-3

W = 15
H = 10
F = 1000
w = 1500
h = 1000


def test_no_rotation_forward():
    s1lat1 = 0
    s1lon1 = 0

    s1lat2 = 0
    s1lon2 = 0

    s2lat1 = 0
    s2lon1 = 1

    s2lat2 = 0
    s2lon2 = 1

    s1pos1 = (s1lat1, s1lon1)
    s1pos2 = (s1lat2, s1lon2)

    s2pos1 = (s2lat1, s2lon1)
    s2pos2 = (s2lat2, s2lon2)

    positions = [(h/2, w/2)]

    proj = vstarstack.projection.perspective.Projection(W, H, F, w, h)
    movement = sphere.Movement.build(s1pos1, s2pos1, s1pos2, s2pos2)

    shifted = movement.apply(positions, proj)
    assert len(shifted) == 1
    assert abs(shifted[0][0] - h/2) < pixthr
    assert abs(shifted[0][1] - w/2) < pixthr


def test_no_rotation_reverse():
    s1lat1 = 0
    s1lon1 = 0

    s1lat2 = 0
    s1lon2 = 0

    s2lat1 = 0
    s2lon1 = 1

    s2lat2 = 0
    s2lon2 = 1

    s1pos1 = (s1lat1, s1lon1)
    s1pos2 = (s1lat2, s1lon2)

    s2pos1 = (s2lat1, s2lon1)
    s2pos2 = (s2lat2, s2lon2)

    positions = [(h/2, w/2)]

    proj = vstarstack.projection.perspective.Projection(W, H, F, w, h)
    movement = sphere.Movement.build(s1pos1, s2pos1, s1pos2, s2pos2)

    shifted = movement.reverse(positions, proj)
    assert len(shifted) == 1
    assert abs(shifted[0][0] - h/2) < pixthr
    assert abs(shifted[0][1] - w/2) < pixthr


def test_lon_rotation_forward():
    dlon = 0.1

    s1lat1 = 0
    s1lon1 = 0

    s1lat2 = 0
    s1lon2 = dlon

    s2lat1 = 0
    s2lon1 = 1

    s2lat2 = 0
    s2lon2 = 1 + dlon

    s1pos1 = (s1lat1, s1lon1)
    s1pos2 = (s1lat2, s1lon2)

    s2pos1 = (s2lat1, s2lon1)
    s2pos2 = (s2lat2, s2lon2)

    x = w/2
    y = h/2

    x_moved_expected = x - w * F/W*math.tan(dlon)
    y_moved_expected = h/2
    positions = [(y, x)]

    proj = vstarstack.projection.perspective.Projection(W, H, F, w, h)
    movement = sphere.Movement.build(s1pos1, s2pos1, s1pos2, s2pos2)

    shifted = movement.apply(positions, proj)
    assert len(shifted) == 1
    assert abs(shifted[0][0] - y_moved_expected) < pixthr
    assert abs(shifted[0][1] - x_moved_expected) < pixthr


def test_lon_rotation_reverse():
    dlon = 0.1

    s1lat1 = 0
    s1lon1 = 0

    s1lat2 = 0
    s1lon2 = dlon

    s2lat1 = 0
    s2lon1 = 1

    s2lat2 = 0
    s2lon2 = 1 + dlon

    s1pos1 = (s1lat1, s1lon1)
    s1pos2 = (s1lat2, s1lon2)

    s2pos1 = (s2lat1, s2lon1)
    s2pos2 = (s2lat2, s2lon2)

    x = w/2
    y = h/2

    x_moved_expected = x + w * F/W*math.tan(dlon)
    y_moved_expected = h/2
    positions = [(y, x)]

    proj = vstarstack.projection.perspective.Projection(W, H, F, w, h)
    movement = sphere.Movement.build(s1pos1, s2pos1, s1pos2, s2pos2)

    shifted = movement.reverse(positions, proj)
    assert len(shifted) == 1
    assert abs(shifted[0][0] - y_moved_expected) < pixthr
    assert abs(shifted[0][1] - x_moved_expected) < pixthr


def test_neglon_rotation_forward():
    dlon = -0.1

    s1lat1 = 0
    s1lon1 = 0

    s1lat2 = 0
    s1lon2 = dlon

    s2lat1 = 0
    s2lon1 = 1

    s2lat2 = 0
    s2lon2 = 1 + dlon

    s1pos1 = (s1lat1, s1lon1)
    s1pos2 = (s1lat2, s1lon2)

    s2pos1 = (s2lat1, s2lon1)
    s2pos2 = (s2lat2, s2lon2)

    x = w/2
    y = h/2

    x_moved_expected = x - w * F/W*math.tan(dlon)
    y_moved_expected = h/2
    positions = [(y, x)]

    proj = vstarstack.projection.perspective.Projection(W, H, F, w, h)
    movement = sphere.Movement.build(s1pos1, s2pos1, s1pos2, s2pos2)

    shifted = movement.apply(positions, proj)
    assert len(shifted) == 1
    assert abs(shifted[0][0] - y_moved_expected) < pixthr
    assert abs(shifted[0][1] - x_moved_expected) < pixthr
