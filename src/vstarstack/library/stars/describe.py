"""Build stars descriptors for matching stars over different images"""
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
import numpy as np

def project_star(star : dict, proj):
    """Project star to sphere"""
    lat, lon = proj.project(star["y"], star["x"])
    star["lon"] = lon
    star["lat"] = lat
    return star

def dirvec(lat, lon):
    """Build vector from sphere center to (lon,lat)"""
    return np.array([math.cos(lon) * math.cos(lat),
                     math.sin(lon) * math.cos(lat),
                     math.sin(lat)])

def cross_production(vec1, vec2):
    """Cross production of vectors"""
    return np.array([vec1[1]*vec2[2] - vec1[2]*vec2[1],
                     vec1[2]*vec2[0] - vec1[0]*vec2[2],
                     vec1[0]*vec2[1] - vec1[1]*vec2[0]])

def angle_between_vectors(vec1, vec2, normed=False):
    """Find angle between 2 vectors"""
    if not normed:
        vec1 /= np.linalg.norm(vec1)
        vec2 /= np.linalg.norm(vec2)
    cosa = np.sum(vec1*vec2)
    if cosa > 1:
        cosa = 1
    if cosa < -1:
        cosa = -1
    return math.acos(cosa)

def _star2star(pos1, star2):
    """Find angle between 2 stars and direction vector from star1 to star2"""
    lon2 = star2["lon"]
    lat2 = star2["lat"]
    pos2 = dirvec(lat2, lon2)

    diff = pos2 - pos1
    direction = diff / np.linalg.norm(diff)
    angle = angle_between_vectors(pos1, pos2, normed=True)
    return angle, direction

def star_triangle(star1, star2, star3):
    """ Find angles between
        star1 and star2, star1 and star3, and
        angle between vectors star1->star2 and star1->star3
        """
    lon1 = star1["lon"]
    lat1 = star1["lat"]
    pos1 = dirvec(lat1, lon1)
    angle12, dir12 = _star2star(pos1, star2)
    angle13, dir13 = _star2star(pos1, star3)

    cross123 = cross_production(dir12, dir13)
    angle123 = angle_between_vectors(dir12, dir13, normed=True)
    if np.sum(cross123 * pos1) < 0:
        angle123 = -angle123

    return (angle12, angle13, angle123)

class DescriptorItem:
    """Star descriptor item"""
    def __init__(self,
                 angle1, relative_size1,
                 angle2, relative_size2,
                 angle12):

        self.angle1 = angle1
        self.angle2 = angle2
        self.relative_size1 = relative_size1
        self.relative_size2 = relative_size2
        self.angle12 = angle12

    def __float__(self):
        return self.angle1 + self.angle2

    def __lt__(self, other):
        return float(self) < float(other)

    def serialize(self) -> dict:
        """Transform descriptor item to dictionary"""
        return {"angle1"  : self.angle1,
                "rs1"     : self.relative_size1,
                "angle2"  : self.angle2,
                "rs2"     : self.relative_size2,
                "angle12" : self.angle12
                }

    @staticmethod
    def deserialize(array : dict):
        """Load descriptor item from dictionary"""        
        item = DescriptorItem(array["angle1"],
                              array["rs1"],
                              array["angle2"],
                              array["rs2"],
                              array["angle12"])
        return item

    def compare(self, other, *,
                max_distance_angle_diff = 1e-4,
                max_vertex_angle_diff = 1e-4,
                max_relative_size_diff = 1e-2):
        """Compare item with other"""
        dangle1 = abs(self.angle1 - other.angle1) / max_distance_angle_diff
        if dangle1 > 1:
            return np.inf

        dangle2 = abs(self.angle2 - other.angle2) / max_distance_angle_diff
        if dangle2 > 1:
            return np.inf

        dsize1 = abs(self.relative_size1 - other.relative_size1) / max_relative_size_diff
        if dsize1 > 1:
            return np.inf

        dsize2 = abs(self.relative_size2 - other.relative_size2) / max_relative_size_diff
        if dsize1 > 1:
            return np.inf

        dangle12 = abs(self.angle12 - self.angle12) / max_vertex_angle_diff
        if dangle12 > 1:
            return np.inf

        return dangle1 + dangle2 + dsize1 + dsize2 + dangle12

class Descriptor:
    """Star descriptor"""
    def __init__(self, items=None):
        if items is None:
            self.items = []
        else:
            self.items = sorted(items)

    def append(self, item : DescriptorItem):
        """Append DescriptorItem to list"""
        self.items.append(item)
        self.items = sorted(self.items)

    def serialize(self):
        """Transfrom descriptor to saveable object"""
        return [item.serialize() for item in self.items]

    @staticmethod
    def deserialize(items : list):
        """Load descriptor from list of items"""
        return Descriptor([DescriptorItem.deserialize(item) for item in items])

def build_descriptor_angled(star, other_stars) -> Descriptor:
    """Build descriptor for star, use angle between stars in triangle"""
    size = star["size"]
    descriptor = Descriptor()

    for j, other1 in enumerate(other_stars):
        size1 = other1["size"]

        for other2 in other_stars[j+1:]:
            size2 = other2["size"]

            if size1 > size2:
                angle1, angle2, angle12 = star_triangle(star, other1, other2)
                item = DescriptorItem(angle2, size2/size,
                                      angle1, size1/size, angle12)
            else:
                angle2, angle1, angle21 = star_triangle(star, other2, other1)
                item = DescriptorItem(angle1, size1/size,
                                      angle2, size2/size, angle21)
            descriptor.append(item)
    return descriptor

def build_descriptor_distance(star, other_stars) -> Descriptor:
    """Build descriptor for star, don't use angle between stars in triangle"""

    size = star["size"]
    descriptor = Descriptor()

    for other1 in other_stars:
        size1 = other1["size"]

        pos = dirvec(star["lat"], star["lon"])
        angle1, _ = _star2star(pos, other1)
        size1 = other1["size"]

        item = DescriptorItem(angle1, size1/size,
                              angle1, size1/size, 0)
        descriptor.append(item)
    return descriptor

def build_descriptors(stars : list, use_angles : bool, projection = None) -> list[Descriptor]:
    """Build descriptors"""
    descriptors = []
    if projection is not None:
        for index, star in enumerate(stars):
            stars[index] = project_star(star, projection)

    for index, star in enumerate(stars):
        other = [item for i,item in enumerate(stars) if i != index]
        if use_angles:
            descriptor = build_descriptor_angled(star, other)
        else:
            descriptor = build_descriptor_distance(star, other)
        descriptors.append(descriptor)
    return descriptors
