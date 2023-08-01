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

import numpy as np
from scipy.spatial.transform import Rotation

from vstarstack.library.movement.sphere import Movement
import vstarstack.library.movement.find_shift

thr = 1e-6

def compare_movements(mov1, mov2):
    rv1 = mov1.as_rotvec()
    rv2 = mov2.as_rotvec()
    drv = rv1 - rv2
    assert (abs(drv) < (thr, thr, thr)).all()

def test_1():
    clusters = [
        {
            "image1" : {
                "lat" : 0,
                "lon" : 0,
            },
            "image2" : {
                "lat" : 0,
                "lon" : 0,
            }
        },
        {
            "image1" : {
                "lat" : 0,
                "lon" : 1,
            },
            "image2" : {
                "lat" : 0,
                "lon" : 1,
            }
        },
    ]
    movements, errors = vstarstack.library.movement.find_shift.build_movements(Movement, clusters)
    assert len(errors) == 0
    assert "image1" in movements
    assert "image2" in movements
    assert "image1" in movements["image1"]
    assert "image1" in movements["image2"]
    assert "image2" in movements["image1"]
    assert "image2" in movements["image2"]

    mov = movements["image1"]["image2"]
    compare_movements(mov.rot, Rotation.identity())
    mov = movements["image2"]["image1"]
    compare_movements(mov.rot, Rotation.identity())

def test_2():
    clusters = [
        {
            "image1" : {
                "lat" : 0,
                "lon" : 0,
            },
            "image2" : {
                "lat" : 0,
                "lon" : 0.1,
            },
            "image3" : {
                "lat" : 0,
                "lon" : 0.2,
            }
        },
        {
            "image1" : {
                "lat" : 0,
                "lon" : 1,
            },
            "image2" : {
                "lat" : 0,
                "lon" : 1.1,
            },
            "image3" : {
                "lat" : 0,
                "lon" : 1.2,
            }
        },
    ]
    movements, errors = vstarstack.library.movement.find_shift.build_movements(Movement, clusters)
    assert len(errors) == 0
    assert "image1" in movements
    assert "image2" in movements
    assert "image3" in movements
    
    mov = movements["image1"]["image2"]
    compare_movements(mov.rot, Rotation.from_rotvec((0, 0, -0.1)))
    mov = movements["image2"]["image1"]
    compare_movements(mov.rot, Rotation.from_rotvec((0, 0, 0.1)))
    mov = movements["image1"]["image3"]
    compare_movements(mov.rot, Rotation.from_rotvec((0, 0, -0.2)))
    mov = movements["image3"]["image2"]
    compare_movements(mov.rot, Rotation.from_rotvec((0, 0, 0.1)))


def test_3():
    clusters = [
        {
            "image1" : {
                "lat" : 0,
                "lon" : 0,
            },
            "image2" : {
                "lat" : 0,
                "lon" : 0.1,
            },
            "image3" : {
                "lat" : 0,
                "lon" : 0.2,
            },
            "image4" : {
                "lat" : 0,
                "lon" : 0.3,
            },
        },
        {
            "image1" : {
                "lat" : 0,
                "lon" : 1,
            },
            "image2" : {
                "lat" : 0,
                "lon" : 1.1,
            },
            "image3" : {
                "lat" : 0,
                "lon" : 1.2,
            },
            "image4" : {
                "lat" : 0,
                "lon" : 1.3,
            }
        },
    ]
    movements, errors = vstarstack.library.movement.find_shift.build_movements(Movement, clusters)
    assert len(errors) == 0
    assert "image1" in movements
    assert "image2" in movements
    assert "image3" in movements
    assert "image4" in movements
    
    movements["image1"].pop("image1")
    movements["image1"].pop("image3")
    movements["image1"].pop("image4")

    movements["image2"].pop("image1")
    movements["image2"].pop("image2")
    movements["image2"].pop("image4")

    movements["image3"].pop("image1")
    movements["image3"].pop("image2")
    movements["image3"].pop("image3")

    movements["image4"].pop("image1")
    movements["image4"].pop("image2")
    movements["image4"].pop("image3")
    movements["image4"].pop("image4")

    movements = vstarstack.library.movement.find_shift.complete_movements(Movement, movements, True)
    identity = Movement.identity()
    
    for name1 in movements:
        compare_movements(movements[name1][name1].rot, identity.rot)
        compare_movements(movements[name1][name1].rev, identity.rot)
        for name2 in movements[name1]:
            mov12 = movements[name1][name2]
            mov21 = movements[name2][name1]
            compare_movements(mov12.rot, mov21.rev)
            compare_movements(mov12.rev, mov21.rot)

def test_4():
    clusters = [
        {
            "image1" : {
                "lat" : 0,
                "lon" : 0,
            },
            "image2" : {
                "lat" : 0,
                "lon" : 0.1,
            },
            "image3" : {
                "lat" : 0,
                "lon" : 0.2,
            },
            "image4" : {
                "lat" : 0,
                "lon" : 0.3,
            },
        },
        {
            "image1" : {
                "lat" : 0,
                "lon" : 1,
            },
            "image2" : {
                "lat" : 0,
                "lon" : 1.1,
            },
            "image3" : {
                "lat" : 0,
                "lon" : 1.2,
            },
            "image4" : {
                "lat" : 0,
                "lon" : 1.3,
            }
        },
    ]
    movements, errors = vstarstack.library.movement.find_shift.build_movements(Movement, clusters)
    assert len(errors) == 0
    assert "image1" in movements
    assert "image2" in movements
    assert "image3" in movements
    assert "image4" in movements
    
    orig14 = movements["image1"]["image4"]
    orig41 = movements["image4"]["image1"]

    movements["image1"].pop("image1")
    movements["image1"].pop("image3")
    movements["image1"].pop("image4")

    movements["image2"].pop("image1")
    movements["image2"].pop("image2")
    movements["image2"].pop("image4")

    movements["image3"].pop("image1")
    movements["image3"].pop("image2")
    movements["image3"].pop("image3")

    movements["image4"].pop("image1")
    movements["image4"].pop("image2")
    movements["image4"].pop("image3")
    movements["image4"].pop("image4")

    movements = vstarstack.library.movement.find_shift.complete_movements(Movement, movements, True)

    compare_movements(orig14.rot, movements["image1"]["image4"].rot)
    compare_movements(orig41.rot, movements["image4"]["image1"].rot)

