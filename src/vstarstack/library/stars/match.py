"""Match stars by descriptors"""
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

from vstarstack.library.stars import describe

class DescriptorMatcher:
    """Match star descriptors"""
    def __init__(self,
                 min_matched_items : int,
                 max_angle_diff : float,
                 max_vertex_angle_diff : float,
                 max_relative_size_diff : float):
        self.min_matched_items = min_matched_items
        self.max_angle_diff = max_angle_diff
        self.max_vertex_angle_diff = max_vertex_angle_diff
        self.max_relative_size_diff = max_relative_size_diff

    def _build_match_table(self,
                           items1 : list[describe.DescriptorItem],
                           items2 : list[describe.DescriptorItem]):
        match_table = {}
        for i, item1 in enumerate(items1):
            match_table[i] = {}
            for j, item2 in enumerate(items2):
                match = item1.compare(item2,
                                      max_distance_angle_diff=self.max_angle_diff,
                                      max_vertex_angle_diff=self.max_vertex_angle_diff,
                                      max_relative_size_diff=self.max_relative_size_diff)
                if match is not np.inf:
                    match_table[i][j] = match
        return match_table

    def _get_matched_items(self, items1, items2):
        used_second = []
        matches = {}
        match_table = self._build_match_table(items1, items2)
        for i,match in match_table.items():
            if len(match) == 0:
                continue
            minimal = np.inf
            index = None
            for j,match_item in match.items():
                if match_item < minimal and j not in used_second:
                    minimal = match_item
                    index = j
            if index is not None:
                matches[i] = index
                used_second.append(index)
        return matches

    def check_match(self, desc1 : describe.Descriptor, desc2 : describe.Descriptor):
        """Compare 2 descriptors"""
        matches = self._get_matched_items(desc1.items, desc2.items)
        minreq = min([self.min_matched_items, len(desc1.items), len(desc2.items)])
        return len(matches) >= minreq

    def build_match(self,
                    descs1 : list[describe.Descriptor],
                    descs2 : list[describe.Descriptor]) -> dict:
        """Find matches between 2 images"""
        matches = {}
        used_second = []
        for i, desc1 in enumerate(descs1):
            for j, desc2 in enumerate(descs2):
                if j in used_second:
                    continue
                if self.check_match(desc1, desc2):
                    matches[i] = j
                    used_second.append(j)
                    break
        return matches

def build_stars_match_table(matcher : DescriptorMatcher,
                            descs : list[list[describe.Descriptor]]):
    """
    Find stars matches between all images

    Arguments:
    * matcher - descriptor matcher
    * descs   - list of list of descriptors

    Function create stars match table
    """

    matches = {}
    for i, stars1 in enumerate(descs):
        matches[i] = {}
        for j, stars2 in enumerate(descs):
            if i == j:
                continue
            matches[i][j] = matcher.build_match(stars1, stars2)

    return matches
