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

import vstarstack.library.projection.perspective as perspective
from vstarstack.library.movement import flat

thr = 1e-6
pixthr = 1e-3

W = 15
H = 10
F = 1000
w = 1500
h = 1000


def test_no_movement_forward():
    # star1 on frame1
    s1y1 = h/2
    s1x1 = w/2

    # star1 on frame2
    s1y2 = h/2
    s1x2 = w/2

    # star2 on frame1
    s2y1 = h/2
    s2x1 = w/2+1

    # star2 on frame2
    s2y2 = h/2
    s2x2 = w/2+1

    s1pos1 = (s1y1, s1x1)
    s1pos2 = (s1y2, s1x2)

    s2pos1 = (s2y1, s2x1)
    s2pos2 = (s2y2, s2x2)

    positions = [(h/2, w/2)]

    proj = perspective.Projection(W, H, F, w, h)
    movement = flat.Movement.build(s1pos1, s2pos1, s1pos2, s2pos2)

    shifted = movement.apply(positions, proj)
    assert len(shifted) == 1
    assert abs(shifted[0][0] - h/2) < pixthr
    assert abs(shifted[0][1] - w/2) < pixthr

def test_no_movement_reverse():
    # star1 on frame1
    s1y1 = h/2
    s1x1 = w/2

    # star1 on frame2
    s1y2 = h/2
    s1x2 = w/2

    # star2 on frame1
    s2y1 = h/2
    s2x1 = w/2+1

    # star2 on frame2
    s2y2 = h/2
    s2x2 = w/2+1

    s1pos1 = (s1y1, s1x1)
    s1pos2 = (s1y2, s1x2)

    s2pos1 = (s2y1, s2x1)
    s2pos2 = (s2y2, s2x2)

    positions = [(h/2, w/2)]

    proj = perspective.Projection(W, H, F, w, h)
    movement = flat.Movement.build(s1pos1, s2pos1, s1pos2, s2pos2)

    shifted = movement.reverse(positions, proj)
    assert len(shifted) == 1
    assert abs(shifted[0][0] - h/2) < pixthr
    assert abs(shifted[0][1] - w/2) < pixthr

def test_rotation_forward():
    # star1 on frame1
    s1y1 = h/2
    s1x1 = w/2

    # star1 on frame2
    s1y2 = h/2
    s1x2 = w/2

    # star2 on frame1
    s2y1 = h/2
    s2x1 = w/2+1

    # star2 on frame2
    s2y2 = h/2+1
    s2x2 = w/2

    s1pos1 = (s1y1, s1x1)
    s1pos2 = (s1y2, s1x2)

    s2pos1 = (s2y1, s2x1)
    s2pos2 = (s2y2, s2x2)

    positions = [(h/2, w/2), (h/2, w/2+1), (h/2+1, w/2)]

    proj = perspective.Projection(W, H, F, w, h)
    movement = flat.Movement.build(s1pos1, s2pos1, s1pos2, s2pos2)

    shifted = movement.apply(positions, proj)
    assert len(shifted) == 3
    assert abs(shifted[0][0] - h/2) < pixthr
    assert abs(shifted[0][1] - w/2) < pixthr
    assert abs(shifted[1][0] - (h/2+1)) < pixthr
    assert abs(shifted[1][1] - w/2) < pixthr
    assert abs(shifted[2][0] - h/2) < pixthr
    assert abs(shifted[2][1] - (w/2-1)) < pixthr

def test_rotation_reverse():
    # star1 on frame1
    s1y1 = h/2
    s1x1 = w/2

    # star1 on frame2
    s1y2 = h/2
    s1x2 = w/2

    # star2 on frame1
    s2y1 = h/2
    s2x1 = w/2+1

    # star2 on frame2
    s2y2 = h/2+1
    s2x2 = w/2

    s1pos1 = (s1y1, s1x1)
    s1pos2 = (s1y2, s1x2)

    s2pos1 = (s2y1, s2x1)
    s2pos2 = (s2y2, s2x2)

    positions = [(h/2, w/2), (h/2, w/2+1), (h/2+1, w/2)]

    proj = perspective.Projection(W, H, F, w, h)
    movement = flat.Movement.build(s1pos1, s2pos1, s1pos2, s2pos2)

    shifted = movement.reverse(positions, proj)
    assert len(shifted) == 3
    assert abs(shifted[0][0] - h/2) < pixthr
    assert abs(shifted[0][1] - w/2) < pixthr
    assert abs(shifted[1][0] - (h/2-1)) < pixthr
    assert abs(shifted[1][1] - w/2) < pixthr
    assert abs(shifted[2][0] - h/2) < pixthr
    assert abs(shifted[2][1] - (w/2+1)) < pixthr

def test_shift_forward():
    # star1 on frame1
    s1y1 = h/2
    s1x1 = w/2

    # star1 on frame2
    s1y2 = h/2
    s1x2 = w/2+1

    # star2 on frame1
    s2y1 = h/2
    s2x1 = w/2+1

    # star2 on frame2
    s2y2 = h/2
    s2x2 = w/2+2

    s1pos1 = (s1y1, s1x1)
    s1pos2 = (s1y2, s1x2)

    s2pos1 = (s2y1, s2x1)
    s2pos2 = (s2y2, s2x2)

    positions = [(h/2, w/2), (h/2, w/2+1), (h/2+1, w/2)]

    proj = perspective.Projection(W, H, F, w, h)
    movement = flat.Movement.build(s1pos1, s2pos1, s1pos2, s2pos2)

    shifted = movement.apply(positions, proj)
    assert len(shifted) == 3
    assert abs(shifted[0][0] - h/2) < pixthr
    assert abs(shifted[0][1] - (w/2+1)) < pixthr
    assert abs(shifted[1][0] - h/2) < pixthr
    assert abs(shifted[1][1] - (w/2+2)) < pixthr
    assert abs(shifted[2][0] - (h/2+1)) < pixthr
    assert abs(shifted[2][1] - (w/2+1)) < pixthr

def test_shift_reverse():
    # star1 on frame1
    s1y1 = h/2
    s1x1 = w/2

    # star1 on frame2
    s1y2 = h/2
    s1x2 = w/2+1

    # star2 on frame1
    s2y1 = h/2
    s2x1 = w/2+1

    # star2 on frame2
    s2y2 = h/2
    s2x2 = w/2+2

    s1pos1 = (s1y1, s1x1)
    s1pos2 = (s1y2, s1x2)

    s2pos1 = (s2y1, s2x1)
    s2pos2 = (s2y2, s2x2)

    positions = [(h/2, w/2), (h/2, w/2+1), (h/2+1, w/2)]

    proj = perspective.Projection(W, H, F, w, h)
    movement = flat.Movement.build(s1pos1, s2pos1, s1pos2, s2pos2)

    shifted = movement.reverse(positions, proj)
    assert len(shifted) == 3
    assert abs(shifted[0][0] - h/2) < pixthr
    assert abs(shifted[0][1] - (w/2-1)) < pixthr
    assert abs(shifted[1][0] - h/2) < pixthr
    assert abs(shifted[1][1] - (w/2)) < pixthr
    assert abs(shifted[2][0] - (h/2+1)) < pixthr
    assert abs(shifted[2][1] - (w/2-1)) < pixthr

def test_rotate_shift_forward():
    # star1 on frame1
    s1y1 = h/2
    s1x1 = w/2

    # star1 on frame2
    s1y2 = h/2
    s1x2 = w/2+1

    # star2 on frame1
    s2y1 = h/2
    s2x1 = w/2+1

    # star2 on frame2
    s2y2 = h/2+1
    s2x2 = w/2+1

    s1pos1 = (s1y1, s1x1)
    s1pos2 = (s1y2, s1x2)

    s2pos1 = (s2y1, s2x1)
    s2pos2 = (s2y2, s2x2)

    positions = [(h/2, w/2), (h/2, w/2+1)]

    proj = perspective.Projection(W, H, F, w, h)
    movement = flat.Movement.build(s1pos1, s2pos1, s1pos2, s2pos2)

    shifted = movement.apply(positions, proj)
    assert len(shifted) == 2
    assert abs(shifted[0][0] - h/2) < pixthr
    assert abs(shifted[0][1] - (w/2+1)) < pixthr
    assert abs(shifted[1][0] - (h/2+1)) < pixthr
    assert abs(shifted[1][1] - (w/2+1)) < pixthr
