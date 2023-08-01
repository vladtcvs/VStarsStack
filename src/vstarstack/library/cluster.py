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

import sys

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

def _process_matches_clusterisation(image1, image2,
                                    match_table : dict,
                                    star_cluster_assignment : dict,
                                    bad_clusters : set):
    changed = False
    for star1, star2 in match_table[image1][image2].items():
        star1 = int(star1)
        if image1 not in star_cluster_assignment:
            star_cluster_assignment[image1] = {}
        if star1 in star_cluster_assignment[image1]:
            cluster_id_1 = star_cluster_assignment[image1][star1]
            if cluster_id_1 in bad_clusters:
                continue
            if image2 not in star_cluster_assignment:
                star_cluster_assignment[image2] = {}
            if star2 in star_cluster_assignment[image2]:
                cluster_id_2 = star_cluster_assignment[image2][star2]
                if cluster_id_2 in bad_clusters:
                    continue
                if cluster_id_1 != cluster_id_2:
                    bad_clusters.add(cluster_id_1)
                    bad_clusters.add(cluster_id_2)
                    print(f"{image1}:{star1} ({cluster_id_1}) <-> {image2}:{star2} ({cluster_id_2})", file=sys.stderr)
                    changed = True
            else:
                star_cluster_assignment[image2][star2] = cluster_id_1
                changed = True
    return changed

def _propagate_cluster_assignment(match_table : dict,
                                  star_cluster_assignment : dict,
                                  bad_clusters : set):
    changed = False
    for image1 in match_table:
        for image2 in match_table[image1]:
            changed = _process_matches_clusterisation(image1,
                                            image2,
                                            match_table,
                                            star_cluster_assignment,
                                            bad_clusters) or changed
            changed = _process_matches_clusterisation(image2,
                                            image1,
                                            match_table,
                                            star_cluster_assignment,
                                            bad_clusters) or changed

    return changed

def _match_table_symmetry(match_table : dict):
    symmetric_match_table = {}
    for image1 in match_table:
        if image1 not in symmetric_match_table:
            symmetric_match_table[image1] = {}
        for image2 in match_table[image1]:
            if image2 not in symmetric_match_table:
                symmetric_match_table[image2] = {}
            if image2 not in symmetric_match_table[image1]:
                symmetric_match_table[image1][image2] = {}
            if image1 not in symmetric_match_table[image2]:
                symmetric_match_table[image2][image1] = {}

    for image1 in match_table:
        for image2 in match_table[image1]:
            for star1 in match_table[image1][image2]:
                star2 = match_table[image1][image2][star1]
                star1 = int(star1)
                symmetric_match_table[image1][image2][star1] = star2
                symmetric_match_table[image2][image1][star2] = star1

    return symmetric_match_table

def find_clusters_in_match_table(match_table : dict):
    """Build cluster of object from match table"""
    # match_table[image_id_1][image_id_2][star_id_1] = star_id_2
    # star_cluster_assignment[image_id][star_id] = cluster_id
    # clusters = {cluster_id:{image_id_1:star_id_1, image_id_2:star_id_2, ...}, {...}, ...}
    cluster_id = 0
    object_cluster_assignment = {}
    bad_clusters = set()
    match_table = _match_table_symmetry(match_table)
    while True:
        cluster_id += 1
        image_id, star_id = _get_new_object(match_table, object_cluster_assignment)
        if image_id is None:
            break
        if image_id not in object_cluster_assignment:
            object_cluster_assignment[image_id] = {}
        object_cluster_assignment[image_id][star_id] = cluster_id
        while _propagate_cluster_assignment(match_table,
                                            object_cluster_assignment,
                                            bad_clusters):
            pass

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
