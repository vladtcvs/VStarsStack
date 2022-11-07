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

import vstarstack.cfg
import sys
import json
import math
import os
import numpy as np
import vstarstack.usage

import vstarstack.common
import matplotlib.pyplot as plt

percent = 60

if hasattr(vstarstack.cfg, "use_sphere"):
	if vstarstack.cfg.use_sphere:
		from vstarstack.movement.sphere import Movement
	else:
		from vstarstack.movement.flat import Movement
else:
	pass

def find_shift(star1, star2):
	star1_from = star1[0]
	star1_to = star1[1]
	
	star2_from = star2[0]
	star2_to = star2[1]
	t = Movement.build(star1_from, star2_from, star1_to, star2_to)
	return t

def run(argv):
	if len(argv) > 1:
		clusters_fname = argv[0]
		shifts_fname = argv[1]
	else:
		clusters_fname = vstarstack.cfg.config["cluster"]["path"]
		shifts_fname = vstarstack.cfg.config["paths"]["relative-shifts"]

	with open(clusters_fname) as f:
		clusters = json.load(f)

	print(clusters)
	names = []
	for cluster in clusters:
		for name in cluster:
			if name not in names:
				names.append(name)

	movements = {}

	for name1 in names:
		movements[name1] = {}
		for name2 in names:
			print("%s / %s" % (name1, name2))
			stars = []
			for cluster in clusters:
				if name1 not in cluster:
					continue
				if name2 not in cluster:
					continue
				if vstarstack.cfg.use_sphere:
					star_to = (cluster[name1]["lat"], cluster[name1]["lon"])
					star_from   = (cluster[name2]["lat"], cluster[name2]["lon"])
				else:
					star_to = (cluster[name1]["y"], cluster[name1]["x"])
					star_from   = (cluster[name2]["y"], cluster[name2]["x"])
					
				stars.append((star_from, star_to))

			ts = []

			for i in range(len(stars)-1):
				star1_from, star1_to = stars[i]
				for j in range(i+1, len(stars)):
					star2_from, star2_to = stars[j]
					try:
						t = find_shift((star1_from, star1_to), (star2_from, star2_to))
						ts.append(t)
					except:
						print("Can not find movement")
						continue

			if len(ts) >= 1:
				t = Movement.average(ts, percent)
				movements[name1][name2] = t.serialize()
			else:
				movements[name1][name2] = None
	data = {
		"movements" : movements
	}
	if vstarstack.cfg.use_sphere:
		data["shift_type"] = "sphere"
	else:
		data["shift_type"] = "flat"
	data["format"] = "relative"
	with open(shifts_fname, "w") as f:
		json.dump(data, f, indent=4, ensure_ascii=False)
