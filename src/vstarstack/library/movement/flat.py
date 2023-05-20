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
import numpy as np

import vstarstack.library.common
import vstarstack.library.movement.basic_movement

class Movement(vstarstack.library.movement.basic_movement.Movement):
    """Flat movements of plane"""

    def apply(self, positions, proj):
        """Apply movement"""
        npositions = []
        for y, x in positions:
            nx = x * math.cos(self.a) - y * math.sin(self.a) + self.dx
            ny = y * math.cos(self.a) + x * math.sin(self.a) + self.dy
            npositions.append((ny, nx))

        return npositions

    def reverse(self, positions, proj):
        """Apply reverse movement"""
        npositions = []
        for y, x in positions:
            nx = (x-self.dx) * math.cos(self.a) + \
                (y-self.dy) * math.sin(self.a)
            ny = (y-self.dy) * math.cos(self.a) - \
                (x-self.dx) * math.sin(self.a)
            npositions.append((ny, nx))

        return npositions

    def magnitude(self):
        """Magnitude of movement"""
        return self.dx**2 + self.dy**2 + self.a**2

    # move pi1, pi2 to p1, p2
    def __init__(self, angle, dy, dx):
        self.dx = dx
        self.dy = dy
        self.a = angle

    def serialize(self):
        """Serialize movement"""
        return json.dumps({"dy": self.dy, "dx": self.dx, "angle": self.a*180/math.pi})

    @staticmethod
    def deserialize(ser):
        """Deserialize movement"""
        s = json.loads(ser)
        return Movement(s["angle"]*math.pi/180, s["dy"], s["dx"])

    @staticmethod
    def build(pi1, pi2, p1, p2, debug=False):
        """Build movement by 2 pairs of points"""
        diy, dix = vstarstack.library.common.norm((pi2[0] - pi1[0], pi2[1] - pi1[1]))
        dy, dx = vstarstack.library.common.norm((p2[0] - p1[0], p2[1] - p1[1]))

        cosa = diy*dy + dix*dx
        sina = dix*dy - diy*dx

        if cosa > 1:
            cosa = 1
        if cosa < -1:
            cosa = -1

        a = math.asin(sina)

        if cosa < 0:
            a = math.pi - a

        dx = 0
        dy = 0
        transformation = Movement(a, dy, dx)
        ty, tx = transformation.apply([(pi1[0], pi1[1])], None)[0]
        dy = p1[0] - ty
        dx = p1[1] - tx
        return Movement(a, dy, dx)

    @staticmethod
    def average(transformations):
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
