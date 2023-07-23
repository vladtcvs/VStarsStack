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

import math
import vstarstack.library.projection.perspective as perspective
from vstarstack.library.movement import flat

thr = 1e-6
pixthr = 1e-3

W = 15
H = 10
F = 1000
w = 1500
h = 1000

def compare_points(point1, point2):
    assert abs(point1[0] - point2[0]) < pixthr
    assert abs(point1[1] - point2[1]) < pixthr

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

def test_multiply_1():
    # star1 on frame1
    s1y1 = h/2
    s1x1 = w/2

    # star1 on frame2
    s1y2 = h/2
    s1x2 = w/2+1

    # star1 on frame3
    s1y3 = h/2
    s1x3 = w/2+2

    # star2 on frame1
    s2y1 = h/2+1
    s2x1 = w/2+1

    # star2 on frame2
    s2y2 = h/2+1
    s2x2 = w/2+2

    # star2 on frame3
    s2y3 = h/2+1
    s2x3 = w/2+3

    s1pos1 = (s1y1, s1x1)
    s1pos2 = (s1y2, s1x2)
    s1pos3 = (s1y3, s1x3)

    s2pos1 = (s2y1, s2x1)
    s2pos2 = (s2y2, s2x2)
    s2pos3 = (s2y3, s2x3)

    movement1 = flat.Movement.build(s1pos1, s2pos1, s1pos2, s2pos2)
    movement2 = flat.Movement.build(s1pos2, s2pos2, s1pos3, s2pos3)

    movement = movement2 * movement1

    proj = perspective.Projection(W, H, F, w, h)
    positions = [s1pos1, s2pos1]
    shifted = movement.apply(positions, proj)
    assert len(shifted) == 2

    compare_points(shifted[0], s1pos3)
    compare_points(shifted[1], s2pos3)

def test_multiply_2():
    # star1 on frame1
    s1y1 = h/2
    s1x1 = w/2

    # star1 on frame2
    s1y2 = h/2
    s1x2 = w/2

    # star1 on frame3
    s1y3 = h/2
    s1x3 = w/2

    # star2 on frame1
    s2y1 = h/2
    s2x1 = w/2+1

    # star2 on frame2
    s2y2 = h/2+1
    s2x2 = w/2

    # star2 on frame3
    s2y3 = h/2-1
    s2x3 = w/2

    s1pos1 = (s1y1, s1x1)
    s1pos2 = (s1y2, s1x2)
    s1pos3 = (s1y3, s1x3)

    s2pos1 = (s2y1, s2x1)
    s2pos2 = (s2y2, s2x2)
    s2pos3 = (s2y3, s2x3)

    movement1 = flat.Movement.build(s1pos1, s2pos1, s1pos2, s2pos2)
    movement2 = flat.Movement.build(s1pos2, s2pos2, s1pos3, s2pos3)

    movement = movement2 * movement1

    proj = perspective.Projection(W, H, F, w, h)
    positions = [s1pos1, s2pos1]
    shifted = movement.apply(positions, proj)
    assert len(shifted) == 2

    compare_points(shifted[0], s1pos3)
    compare_points(shifted[1], s2pos3)

def test_multiply_3():
    # star1 on frame1
    s1y1 = h/2
    s1x1 = w/2

    # star1 on frame2
    s1y2 = h/2
    s1x2 = w/2+1

    # star1 on frame3
    s1y3 = h/2+1
    s1x3 = w/2+1

    # star2 on frame1
    s2y1 = h/2
    s2x1 = w/2+1

    # star2 on frame2
    s2y2 = h/2+1
    s2x2 = w/2+1

    # star2 on frame3
    s2y3 = h/2+1
    s2x3 = w/2

    s1pos1 = (s1y1, s1x1)
    s1pos2 = (s1y2, s1x2)
    s1pos3 = (s1y3, s1x3)

    s2pos1 = (s2y1, s2x1)
    s2pos2 = (s2y2, s2x2)
    s2pos3 = (s2y3, s2x3)

    movement1 = flat.Movement.build(s1pos1, s2pos1, s1pos2, s2pos2)
    movement2 = flat.Movement.build(s1pos2, s2pos2, s1pos3, s2pos3)

    movement = movement2 * movement1

    proj = perspective.Projection(W, H, F, w, h)
    positions = [s1pos1, s2pos1]
    shifted = movement.apply(positions, proj)
    assert len(shifted) == 2

    compare_points(shifted[0], s1pos3)
    compare_points(shifted[1], s2pos3)

def test_multiply_4():
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
    s2y2 = h/2 + 1/2**0.5
    s2x2 = w/2 + 1/2**0.5 + 1

    s1pos1 = (s1y1, s1x1)
    s1pos2 = (s1y2, s1x2)

    s2pos1 = (s2y1, s2x1)
    s2pos2 = (s2y2, s2x2)

    movement1 = flat.Movement(0, -h/2, -w/2)
    movement2 = flat.Movement(math.pi/4, 0, 0)
    movement3 = flat.Movement(0, 0, 1)
    movement4 = flat.Movement(0, h/2, w/2)

    movement = movement4 * movement3 * movement2 * movement1

    proj = perspective.Projection(W, H, F, w, h)
    positions = [s1pos1, s2pos1]
    shifted = movement.apply(positions, proj)
    assert len(shifted) == 2

    compare_points(shifted[0], s1pos2)
    compare_points(shifted[1], s2pos2)

def test_inverse():
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

    positions = [s1pos1, s2pos1]

    proj = perspective.Projection(W, H, F, w, h)
    movement = flat.Movement.build(s1pos1, s2pos1, s1pos2, s2pos2)
    rev_movement = movement.inverse()
    mov1 = movement * rev_movement
    mov2 = rev_movement * movement

    shifted1 = mov1.apply(positions, proj)
    shifted2 = mov2.apply(positions, proj)
    assert len(shifted1) == 2
    assert len(shifted2) == 2
    compare_points(shifted1[0], s1pos1)
    compare_points(shifted1[1], s2pos1)
    compare_points(shifted2[0], s1pos1)
    compare_points(shifted2[1], s2pos1)
