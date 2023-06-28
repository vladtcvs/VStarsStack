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
import json

import vstarstack.library.movement.find_shift
from vstarstack.library.movement.sphere import Movement

thr = 1e-6

def test_cluster_movements_1():
    clusters = [
        {
            "frame1": {
                "lon": 0,
                "lat": 0,
            },
            "frame2": {
                "lon": 0,
                "lat": 0,
            },
        },
        {
            "frame1": {
                "lon": 1e-3,
                "lat": 0,
            },
            "frame2": {
                "lon": 1e-3,
                "lat": 0,
            }
        }
    ]

    movements = vstarstack.library.movement.find_shift.build_movements(Movement, clusters)
    movement = movements["frame1"]["frame2"]
    ser = movement.serialize()
    assert ser == "{\"rot\": [0.0, 0.0, 0.0, 1.0]}"

def test_cluster_movements_2():
    angle = 1
    clusters = [
        {
            "frame1": {
                "lon": 0,
                "lat": 0,
            },
            "frame2": {
                "lon": angle,
                "lat": 0,
            },
        },
        {
            "frame1": {
                "lon": 1e-3,
                "lat": 0,
            },
            "frame2": {
                "lon": angle+1e-3,
                "lat": 0,
            }
        }
    ]

    movements = vstarstack.library.movement.find_shift.build_movements(Movement, clusters)
    movement = movements["frame2"]["frame1"]
    ser = movement.serialize()
    quat = json.loads(ser)
    assert abs(quat["rot"][0] - 0) < thr
    assert abs(quat["rot"][1] - 0) < thr
    assert abs(quat["rot"][2] - math.sin(angle/2)) < thr
    assert abs(quat["rot"][3] - math.cos(angle/2)) < thr

def test_cluster_movements_3():
    clusters = [
        {
            "frame1": {
                "lon": 0,
                "lat": 0,
            },
            "frame2": {
                "lon": 0,
                "lat": -1,
            },
        },
        {
            "frame1": {
                "lon": 1,
                "lat": 0,
            },
            "frame2": {
                "lon": 1.2368643351396535,
                "lat": -0.4719777676633856,
            }
        }
    ]

    movements = vstarstack.library.movement.find_shift.build_movements(Movement, clusters)
    movement = movements["frame2"]["frame1"]
    ser = movement.serialize()
    quat = json.loads(ser)
    assert abs(quat["rot"][0] - 0) < thr
    assert abs(quat["rot"][1] - math.sin(1/2)) < thr
    assert abs(quat["rot"][2] - 0) < thr
    assert abs(quat["rot"][3] - math.cos(1/2)) < thr
