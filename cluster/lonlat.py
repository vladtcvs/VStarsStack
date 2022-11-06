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
import sys
import json

import cfg

import projection.perspective
import common

def run(argv):
	infilename = argv[0]
	if len(argv) == 1:
		outfilename = infilename
	else:
		outfilename = argv[1]

	proj = projection.perspective.Projection(cfg.camerad["W"], cfg.camerad["H"], cfg.camerad["F"], cfg.camerad["w"], cfg.camerad["h"])
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

