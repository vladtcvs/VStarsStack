"""Flat movements"""
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
import json
from typing import List
import numpy as np

import vstarstack.library.common
from vstarstack.library.common import norm
import vstarstack.library.movement.basic_movement

class Movement(vstarstack.library.movement.basic_movement.Movement):
    """Flat movements of plane"""

    def apply(self, positions : list, proj) -> List:
        """Apply movement"""
        npositions = []
        for y, x in positions:
            nx = x * math.cos(self.a) - y * math.sin(self.a) + self.dx
            ny = y * math.cos(self.a) + x * math.sin(self.a) + self.dy
            npositions.append((ny, nx))

        return npositions

    def reverse(self, positions : list, proj) -> List:
        """Apply reverse movement"""
        npositions = []
        for y, x in positions:
            nx = (x-self.dx) * math.cos(self.a) + \
                (y-self.dy) * math.sin(self.a)
            ny = (y-self.dy) * math.cos(self.a) - \
                (x-self.dx) * math.sin(self.a)
            npositions.append((ny, nx))

        return npositions

    def magnitude(self) -> float:
        """Magnitude of movement"""
        return self.dx**2 + self.dy**2 + self.a**2

    def __init__(self, angle, dy, dx):
        self.dx = dx
        self.dy = dy
        self.a = angle

    def serialize(self) -> str:
        """Serialize movement"""
        return json.dumps({"dy": self.dy, "dx": self.dx, "angle": self.a*180/math.pi})

    @staticmethod
    def deserialize(ser : str):
        """Deserialize movement"""
        s = json.loads(ser)
        return Movement(s["angle"]*math.pi/180, s["dy"], s["dx"])

    @staticmethod
    def identity():
        return Movement(0, 0, 0)

    @staticmethod
    def build(point1_from, point2_from, point1_to, point2_to, debug=False):
        """Build movement by 2 pairs of points"""
        dir_from = norm((point2_from[0] - point1_from[0], point2_from[1] - point1_from[1]))
        dir_to = norm((point2_to[0] - point1_to[0], point2_to[1] - point1_to[1]))

        cosa = dir_from[0]*dir_to[0] + dir_from[1]*dir_to[1]
        sina = dir_from[1]*dir_to[0] - dir_from[0]*dir_to[1]

        cosa = np.clip(cosa, -1, 1)
        sina = np.clip(sina, -1, 1)

        angle = math.asin(sina)
        if cosa < 0:
            angle = math.pi - angle

        transformation = Movement(angle, 0, 0)
        point1_rotated = transformation.apply([point1_from], None)[0]
        dy = point1_to[0] - point1_rotated[0]
        dx = point1_to[1] - point1_rotated[1]
        return Movement(angle, dy, dx)

    @staticmethod
    def average(transformations : list):
        """average on multiple movements"""
        angles = []
        dxs = []
        dys = []
        for transformation in transformations:
            angles.append(transformation.a)
            dxs.append(transformation.dx)
            dys.append(transformation.dy)
        angle = np.average(angles)
        dy = np.average(dys)
        dx = np.average(dxs)
        transformation = Movement(angle, dy, dx)
        return transformation

    def __mul__(self, other):
        """Multiply movements"""
        angle1 = self.a
        dx1 = self.dx
        dy1 = self.dy

        angle2 = other.a
        dx2 = other.dx
        dy2 = other.dy

        angle = angle1 + angle2
        dx = dx2 * math.cos(angle1) - dy2 * math.sin(angle1) + dx1
        dy = dx2 * math.sin(angle1) + dy2 * math.cos(angle1) + dy1
        return Movement(angle, dy, dx)

    def inverse(self):
        idx = -self.dx * math.cos(self.a) - self.dy * math.sin(self.a)
        idy = -self.dy * math.cos(self.a) + self.dx * math.sin(self.a)
        return Movement(-self.a, idy, idx)
