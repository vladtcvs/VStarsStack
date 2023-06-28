"""Select reference image for shift"""
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

def select_base_image(shifts : dict) -> str:
    """Select image which wouldn't move"""

    name0 = None
    mindistance = None
    max_size_of_cluster = None
    for name in shifts:
        size_of_cluster = len(shifts[name])
        if max_size_of_cluster is not None and size_of_cluster < max_size_of_cluster:
            continue
        if max_size_of_cluster is None or size_of_cluster > max_size_of_cluster:
            max_size_of_cluster = size_of_cluster
            mindistance = None
        distance = 0
        for name2 in shifts[name]:
            shift = shifts[name][name2]
            distance += shift.magnitude()

        if mindistance is None or distance < mindistance:
            mindistance = distance
            name0 = name

    return name0
