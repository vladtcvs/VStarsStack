#
# Copyright (c) 2024 Vladislav Tsendrovskii
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

from vstarstack.library.clusters.clusterization import build_clusters

def test_1():
    match_list = [(0,0,1,0)]
    cluster_list = build_clusters(match_list)
    assert cluster_list is not None
    assert len(cluster_list) == 1
    assert len(cluster_list[0]) == 2
    assert cluster_list[0][0] == (0,0)
    assert cluster_list[0][1] == (1,0)

def test_2():
    match_list = [(0,0,1,0), (1,0,0,0)]
    cluster_list = build_clusters(match_list)
    assert cluster_list is not None
    assert len(cluster_list) == 1
    assert len(cluster_list[0]) == 2
    assert cluster_list[0][0] == (0,0)
    assert cluster_list[0][1] == (1,0)

def test_3():
    match_list = [(0,0,1,0), (0,1,1,1)]
    cluster_list = build_clusters(match_list)
    assert cluster_list is not None
    assert len(cluster_list) == 2
    assert len(cluster_list[0]) == 2
    assert cluster_list[0][0] == (0,0)
    assert cluster_list[0][1] == (1,0)
    assert len(cluster_list[1]) == 2
    assert cluster_list[1][0] == (0,1)
    assert cluster_list[1][1] == (1,1)

def test_4():
    match_list = [(0,0,1,0), (0,1,1,1), (0,0,2,1)]
    cluster_list = build_clusters(match_list)
    assert cluster_list is not None
    assert len(cluster_list) == 2
    assert len(cluster_list[0]) == 3
    assert cluster_list[0][0] == (0,0)
    assert cluster_list[0][1] == (1,0)
    assert cluster_list[0][2] == (2,1)
    assert len(cluster_list[1]) == 2
    assert cluster_list[1][0] == (0,1)
    assert cluster_list[1][1] == (1,1)

    