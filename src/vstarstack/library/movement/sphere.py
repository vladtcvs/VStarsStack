"""Spherical rotations"""
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

import logging
import math
import json
from typing import Any
import numpy as np

from scipy.spatial.transform import Rotation

import vstarstack.library.movement.basic_movement
from vstarstack.library.movement.movements import SphereMovement

logger = logging.getLogger(__name__)

def p2vec(pos):
    """Lon,lat -> x,y,z"""
    lon = pos[0]
    lat = pos[1]
    return np.array([math.cos(lon)*math.cos(lat), math.sin(lon)*math.cos(lat), math.sin(lat)])


def vecmul(vec1, vec2):
    """Vector cross multiplication"""
    return np.cross(vec1, vec2)


def scmul(vec1, vec2):
    """Vector scalar multiplication"""
    return np.dot(vec1, vec2)


def vecangle(vec1, vec2):
    """Angle between vectors"""
    dir1 = np.linalg.norm(vec1)
    dir2 = np.linalg.norm(vec2)
    return np.arccos(np.clip(np.dot(vec1/dir1, vec2/dir2), -1.0, 1.0))


class Movement(vstarstack.library.movement.basic_movement.Movement):
    """Class of spherical movements"""

    def apply(self, positions : np.ndarray, input_proj, output_proj):
        """Apply movement in (x,y) coordinates"""
        return self.mov.apply_forward(positions, input_proj, output_proj)

    def apply_lonlat(self, positions : np.ndarray):
        """Apply movement in (lon,lat) coordinates"""
        return self.mov.apply_forward_lonlat(positions)

    def reverse(self, positions : np.ndarray, input_proj, output_proj):
        """Apply reverse movement in (x,y) coordinates"""
        return self.mov.apply_reverse(positions, input_proj, output_proj)

    def reverse_lonlat(self, positions : np.ndarray):
        """Apply reverse movement in (lon,lat) coordinates"""
        return self.mov.apply_reverse_lonlat(positions)

    def __init__(self, rot):
        self.rot = rot
        self.rev = rot.inv()
        q = self.rot.as_quat()
        self.mov = SphereMovement(q[3], q[0], q[1], q[2])

    def __reduce__(self) -> str | tuple[Any, ...]:
        return (self.__class__, (self.rot, ))

    def magnitude(self):
        """Magnitude of movement"""
        rotvec = self.rot.as_rotvec()
        angle = np.linalg.norm(rotvec)
        return angle**2

    def serialize(self):
        """Store movement to string"""
        quaternion = self.rot.as_quat()
        return json.dumps({"rot": list(quaternion)})

    @staticmethod
    def deserialize(ser):
        """Restore movement from string"""
        desc = json.loads(ser)
        quaternion = desc["rot"]
        rot = Rotation.from_quat(np.array(quaternion))
        return Movement(rot)

    # identity
    @staticmethod
    def identity():
        """Build identity movement"""
        rot = Rotation.identity()
        return Movement(rot)

    # move pi1, pi2 to p1, p2
    @staticmethod
    def build(point1_from, point2_from, point1_to, point2_to, debug=False):
        """Build movement by 2 pairs of points"""
        v1_from = p2vec(point1_from)
        v2_from = p2vec(point2_from)
        v1_to = p2vec(point1_to)
        v2_to = p2vec(point2_to)

        dangle_from = vecangle(v1_from, v2_from)
        dangle_to = vecangle(v1_to, v2_to)
        assert abs(dangle_from - dangle_to) < 1*math.pi/180

        # build rotation1, which moves v1_from to v1_to
        axis1 = vecmul(v1_from, v1_to)
        angle1 = vecangle(v1_from, v1_to)

        if angle1 == 0:
            rot1 = Rotation.from_rotvec([0, 0, 0])
        else:
            axis1 = axis1 / np.linalg.norm(axis1)
            rot1 = Rotation.from_rotvec(angle1 * axis1)

        # intermidiate position of v1 and v2 vectors
        # v1_int must be equal to v1_to
        v1_int = rot1.apply([v1_from])[0]
        v2_int = rot1.apply([v2_from])[0]

        v1_int = v1_int / np.linalg.norm(v1_int)
        v2_int = v2_int / np.linalg.norm(v2_int)

        logger.debug(f"angle1 = {angle1}")
        assert (vecangle(v1_int, v1_to) < 1e-5)

        # build rotation2, which moves v2_int to v2_to
        axis2 = v1_to

        v2proj_int = scmul(axis2, v2_int)
        v2proj_to = scmul(axis2, v2_to)

        v2ort_int = v2_int - axis2 * v2proj_int
        v2ort_to = v2_to - axis2 * v2proj_to

        angle2 = vecangle(v2ort_int, v2ort_to)

        ort = vecmul(v2ort_int, v2ort_to)
        if scmul(ort, axis2) < 0:
            angle2 = -angle2

        if angle2 == 0:
            rot2 = Rotation.from_rotvec([0, 0, 0])
        else:
            rot2 = Rotation.from_rotvec(angle2 * axis2)

        v1_res = rot2.apply([v1_int])[0]
        v2_res = rot2.apply([v2_int])[0]

        assert (vecangle(v1_res, v1_to) < 1*math.pi/180)
        assert (vecangle(v2_res, v2_to) < 1*math.pi/180)

        rot = rot2 * rot1

        v1_res = rot.apply([v1_from])[0]
        v2_res = rot.apply([v2_from])[0]

        assert (vecangle(v1_res, v1_to) < 0.7*math.pi/180)
        assert (vecangle(v2_res, v2_to) < 0.7*math.pi/180)

        return Movement(rot)

    @staticmethod
    def average(transformations, percent=100):
        """Average of multiple movements"""
        axises = np.zeros((len(transformations), 3))
        for i, transformation in enumerate(transformations):
            rotvec = transformation.rot.as_rotvec()
            axises[i, 0:3] = rotvec

        logger.debug(f"axises {axises}")
        if percent == 100:
            # Use all vectors
            rotvec = np.average(axises, axis=0)
        else:
            # Use only specified percent of nearest to average
            rotvec = np.average(axises, axis=0)
            distances = []
            for i in range(len(transformations)):
                daxis = axises[i] - rotvec
                distance = np.sum(daxis*daxis)**0.5
                distances.append((distance, axises[i]))
                # print(dl)
            distances.sort(key=lambda item: item[0])
            num = max(1, math.ceil(percent * len(distances) / 100))
            distances = distances[:num]
            rotvec = np.zeros((3,))
            for _, axis in distances:
                rotvec += axis
            rotvec /= len(distances)
        rot = Rotation.from_rotvec(rotvec)
        transformation = Movement(rot)
        return transformation

    def __mul__(self, other):
        """Multiply movements"""
        rot1 = self.rot
        rot2 = other.rot
        return Movement(rot1 * rot2)

    def inverse(self):
        return Movement(self.rev)
