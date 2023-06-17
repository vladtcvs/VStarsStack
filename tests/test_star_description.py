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
import vstarstack.library.stars.describe as describe

thr = 1e-6

def assert_descriptor(desc : describe.DescriptorItem, expected_desc):
    eid1, eangle1, ers1, eid2, eangle2, ers2, eangle12 = expected_desc
    assert desc.id1 == eid1
    assert desc.id2 == eid2
    assert abs(desc.angle1 - eangle1) < thr
    assert abs(desc.angle2 - eangle2) < thr
    assert abs(desc.relative_size1 - ers1) < thr
    assert abs(desc.relative_size2 - ers2) < thr
    assert abs(desc.angle12 - eangle12) < thr

def test_angles_1():
    vec1 = describe.dirvec(0, 0)
    vec2 = describe.dirvec(0, 1)
    angle = describe.angle_between_vectors(vec1, vec2)
    assert abs(angle - 1) < thr

def test_angles_2():
    vec1 = describe.dirvec(0, 0)
    vec2 = describe.dirvec(1, 0)
    angle = describe.angle_between_vectors(vec1, vec2)
    assert abs(angle - 1) < thr

def test_angles_3():
    star1 = {"lon" : 0, "lat" : 0}
    star2 = {"lon" : 1e-3, "lat" : 0}
    star3 = {"lon" : 0, "lat" : 1e-3}
    angle12, angle13, angle123 = describe.star_triangle(star1, star2, star3)
    assert abs(angle12 - 1e-3) < thr
    assert abs(angle13 - 1e-3) < thr
    assert abs(angle123 - math.pi/2) < 1e-4

def test_angles_4():
    star1 = {"lon" : 0, "lat" : 0}
    star2 = {"lon" : 0, "lat" : 1e-3}
    star3 = {"lon" : 1e-3, "lat" : 0}
    angle12, angle13, angle123 = describe.star_triangle(star1, star2, star3)
    assert abs(angle12 - 1e-3) < thr
    assert abs(angle13 - 1e-3) < thr
    assert abs(angle123 + math.pi/2) < 1e-4

def test_descriptor_1():
    star1 = {"id" : 1, "size" : 1, "lon" : 0, "lat" : 0}
    star2 = {"id" : 2, "size" : 2, "lon" : 1e-3, "lat" : 0}
    star3 = {"id" : 3, "size" : 3, "lon" : 0, "lat" : 1e-3}
    desc = describe.build_descriptor_angled(star1, [star2, star3])
    assert len(desc.items) == 1
    assert_descriptor(desc.items[0], (2, 1e-3, 2, 3, 1e-3, 3, -math.pi/2))

def test_descriptors_1():
    stars = [
        {
            "id" : 1,
            "lon" : 0,
            "lat" : 0,
            "size" : 1
        },
        {
            "id" : 2,
            "lon" : 1e-3,
            "lat" : 0,
            "size" : 2
        },
        {
            "id" : 3,
            "lon" : 0,
            "lat" : 2e-3,
            "size" : 3
        },
    ]

    descriptors = describe.build_descriptors(stars, True, None)
    assert len(descriptors) == 3

    assert len(descriptors[0].items) == 1
    assert_descriptor(descriptors[0].items[0],
                      (2, 1e-3, 2, 3, 2e-3, 3, -math.pi/2))

    assert len(descriptors[1].items) == 1
    assert_descriptor(descriptors[1].items[0],
                      (1, 1e-3, 0.5, 3, (2e-3**2+1e-3**2)**0.5, 1.5, math.atan(2)))

    assert len(descriptors[2].items) == 1
    assert_descriptor(descriptors[2].items[0],
                      (1, 2e-3, 1/3, 2, (2e-3**2+1e-3**2)**0.5, 2/3, -math.atan(1/2)))
