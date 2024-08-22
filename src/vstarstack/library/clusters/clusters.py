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

import vstarstack.library.clusters.clusterization

def find_clusters_in_match_table(match_table : dict):
    """Build cluster of object from match table"""
    # match_table[image_id_1][image_id_2][star_id_1] = star_id_2
    # match_list = [(image_id_1, star_id_1, image_id_2, star_id_2), (...), ...]
    # cluster_list = [ [(image_id1,star_id_1), (image_id1,star_id_1)),... ], [...], ... ]
    # clusters = [{image_id_1:star_id_1, image_id_2:star_id_2, ...}, {...}, ...]
    match_list = []
    clusters = []
    for image_id_1 in match_table:
        for image_id_2 in match_table[image_id_1]:
            for star_id_1, star_id_2 in match_table[image_id_1][image_id_2].items():
                match_list.append((image_id_1, star_id_1, image_id_2, star_id_2))
    cluster_list = vstarstack.library.clusters.clusterization.build_clusters(match_list)
    for cluster in cluster_list:
        if len(cluster) < 2:
            continue
        vcluster = {}
        for image_id, star_id in cluster:
            if image_id in vcluster:
                # bad cluster            
                break
            vcluster[image_id] = star_id
        else:
            clusters.append(vcluster)
    return clusters
