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

class PlanetProjection(object):
    # @param W image width
    # @param H image height
    # @param a planet ellipse major axis
    # @param b planet ellipse minor axis
    # @param angle planet ellipse slope
    # @param rot planet rotation angle
    def __init__(self, W, H, a, b, angle, rot):
        self.W = W
        self.H = H
        self.a = a
        self.b = b
        self.angle = angle
        self.rot = rot
    
    def from_planet_coordinates(self, lon, lat):
        x = self.a/2 * math.sin(lon - self.rot) * math.cos(lat)
        z = -self.b/2 * math.sin(lat)
        X = x*math.cos(self.angle) + z*math.sin(self.angle) + self.W/2
        Y = -x*math.sin(self.angle) + z*math.cos(self.angle) + self.H/2
        return X, Y
    
    def to_planet_coordinates(self, X, Y):
        X = X - self.W/2
        Y = Y - self.H/2
        x = X*math.cos(self.angle) - Y*math.sin(self.angle)
        z = X*math.sin(self.angle) + Y*math.cos(self.angle)
        try:
            lat = -math.asin(z/self.b*2)
            lon = math.asin(x/self.a*2/math.cos(lat))+self.rot
            return lon, lat
        except:
            return None, None

def equal(val, expected):
    return abs(val - expected) < 1e-6

if __name__ == "__main__":
    # perform testing
    
    # sphere
    proj = PlanetProjection(100, 100, 50, 50, 0, 0)
    lon, lat = proj.to_planet_coordinates(50, 50)
    print(lon, lat)
    assert equal(lon, 0)
    assert equal(lat, 0)
    print("test 1 pass")

    lon, lat = proj.to_planet_coordinates(75, 50)
    print(lon, lat)
    assert equal(lon, math.pi/2)
    assert equal(lat, 0)
    print("test 2 pass")

    lon, lat = proj.to_planet_coordinates(25, 50)
    print(lon, lat)
    assert equal(lon, -math.pi/2)
    assert equal(lat, 0)
    print("test 3 pass")

    lon, lat = proj.to_planet_coordinates(50, 75)
    print(lon, lat)
    assert equal(lon, 0)
    assert equal(lat, -math.pi/2)
    print("test 4 pass")

    lon, lat = proj.to_planet_coordinates(75, 75)
    print(lon, lat)
    assert lon is None
    assert lat is None
    print("test 5 pass")
