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

import json

import vstarstack.cfg

import vstarstack.projection.perspective


def run(project: vstarstack.cfg.Project, argv: list):
    infilename = argv[0]
    if len(argv) == 1:
        outfilename = infilename
    else:
        outfilename = argv[1]

    proj = vstarstack.projection.perspective.Projection(project.camera.W,
                                                        project.camera.H,
                                                        project.scope.F,
                                                        project.camera.w,
                                                        project.camera.h)
    with open(infilename) as f:
        clusters = json.load(f)
    for cluster in clusters:
        for name in cluster:
            x = cluster[name]["x"]
            y = cluster[name]["y"]
            lat, lon = proj.project(y, x)
            cluster[name]["lon"] = lon
            cluster[name]["lat"] = lat
    with open(outfilename, "w") as f:
        json.dump(clusters, f, indent=4)
