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

class BadClusterException(Exception):
    """Exception in case of bad cluster"""
    def __init__(self):
        Exception.__init__(self, "Bad cluster")

def _get_new_object(match_table, star_cluster_assignment):
    for image_id1 in match_table:
        for image_id2 in match_table[image_id1]:
            for star_id1 in match_table[image_id1][image_id2]:
                if image_id1 not in star_cluster_assignment or          \
                    star_id1 not in star_cluster_assignment[image_id1]:
                    return image_id1, star_id1
    return None, None

def _propagate_cluster_assignment(image_id : int,
                                  star_id : int,
                                  match_table : dict,
                                  star_cluster_assignment : dict):
    changed = False
    cluster_id = star_cluster_assignment[image_id][star_id]
    for image_id2, matches in match_table[image_id].items():
        if star_id in matches:
            star_id2 = matches[star_id]
            if image_id2 not in star_cluster_assignment:
                star_cluster_assignment[image_id2] = {}
            if star_id2 in star_cluster_assignment[image_id2]:
                if star_cluster_assignment[image_id2][star_id2] != cluster_id:
                    raise BadClusterException()
            else:
                star_cluster_assignment[image_id2][star_id2] = cluster_id
                changed = True
    return changed

def find_clusters_in_match_table(match_table : dict):
    """Build cluster of object from match table"""
    # match_table[image_id_1][image_id_2][star_id_1] = star_id_2
    # star_cluster_assignment[image_id][star_id] = cluster_id
    # clusters = {cluster_id:{image_id_1:star_id_1, image_id_2:star_id_2, ...}, {...}, ...}
    cluster_id = 0
    object_cluster_assignment = {}
    bad_clusters = []
    while True:
        cluster_id += 1
        image_id, star_id = _get_new_object(match_table, object_cluster_assignment)
        if image_id is None:
            break
        if image_id not in object_cluster_assignment:
            object_cluster_assignment[image_id] = {}
        object_cluster_assignment[image_id][star_id] = cluster_id
        try:
            while _propagate_cluster_assignment(image_id,
                                                star_id,
                                                match_table,
                                                object_cluster_assignment):
                pass
        except BadClusterException:
            bad_clusters.append(cluster_id)

    clusters = {}
    for image_id, stars in object_cluster_assignment.items():
        for star_id, cluster_id in stars.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = {}
            clusters[cluster_id][image_id] = star_id

    cluster_list = []
    for cluster_id, cluster in clusters.items():
        if cluster_id not in bad_clusters:
            cluster_list.append(cluster)
    return cluster_list
