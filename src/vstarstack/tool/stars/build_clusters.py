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

import json


import vstarstack.tool.usage
import vstarstack.tool.cfg
import vstarstack.tool.common
import vstarstack.library.common
import vstarstack.library.cluster

def _prepare_match_table(match_table):
    mt = {}
    for image_id1 in match_table:
        mt[image_id1] = {}
        for image_id2 in match_table[image_id1]:
            mt[image_id1][image_id2] = {}
            for star_id1 in match_table[image_id1][image_id2]:
                star_id2 = match_table[image_id1][image_id2][star_id1]
                mt[image_id1][image_id2][int(star_id1)] = star_id2
    return mt

def run(project: vstarstack.tool.cfg.Project, argv: list):
    if len(argv) >= 2:
        descs_path = argv[0]
        match_table_f = argv[1]
        cluster_f = argv[2]
    else:
        descs_path = project.config.paths.descs
        match_table_f = project.config.stars.paths.matchfile
        cluster_f = project.config.cluster.path

    with open(match_table_f, encoding='utf8') as f:
        match_table = _prepare_match_table(json.load(f))

    print("Find star index cluster")
    clusters = vstarstack.library.cluster.find_clusters_in_match_table(match_table)
    dclusters = sorted(clusters, key=lambda x : len(x), reverse=True)
    dclusters = [item for item in dclusters if len(item) > 1]
    print("Done")

    stars_files = vstarstack.tool.common.listfiles(descs_path, ".json")
    descs = {}
    for name, fname in stars_files:
        with open(fname, encoding='utf8') as file:
            desc = json.load(file)
        descs[name] = desc

    star_clusters = []
    for cluster in dclusters:
        star_cluster = {}
        for name, star_id in cluster.items():
            star_cluster[name] = descs[name]["points"][star_id]["keypoint"]
        star_clusters.append(star_cluster)

    with open(cluster_f, "w", encoding='utf8') as f:
        json.dump(star_clusters, f, indent=4)
