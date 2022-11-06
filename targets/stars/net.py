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
import usage
import common
import cfg

def process(argv):
	if len(argv) >= 2:
		jsondir = argv[0]
		out = argv[1]
	else:
		jsondir = cfg.config["stars"]["paths"]["descs"]
		out = cfg.config["stars"]["paths"]["net"]

	imagesfiles = common.listfiles(jsondir, ".json")

	names = []

	images = {}
	for name,filename in imagesfiles:
		names.append(name)
		with open(filename) as f:
			images[name] = json.load(f)

	net = {}
	for name in names:
		net[name] = {}
		image = images[name]
		stars = image["main"]
		for i in range(len(stars)):
			net[name][i] = {}
			for name2 in names:
				m = stars[i]["matches"][name2]
				if m is not None:
					net[name][i][name2] = m

	# find invalid matches
	errors = []
	for name1 in net:
		for i in net[name1]:
			for name2 in net[name1][i]:
				j = net[name1][i][name2]
				if name2 in net and j in net[name2] and name1 in net[name2][j]:
					if net[name2][j][name1] != i:
						errors.append((name1, i, name2, j))
	# remove invalid matches
	for name1, i, name2, j in errors:
		if name2 in net[name1][i]:
			net[name1][i].pop(name2)
		if name1 in net[name2][j]:
			net[name2][j].pop(name1)

	# symmetrise net
	ns = []
	for name1 in net:
		for i in net[name1]:
			for name2 in net[name1][i]:
				j = net[name1][i][name2]
				if name2 not in net or j not in net[name2] or name1 not in net[name2][j]:
					ns.append((name1, i, name2, j))
	for name1, i, name2, j in ns:
		if name1 not in net:
			net[name1] = {}
		if i not in net[name1]:
			net[name1][i] = {}
		if name2 not in net[name1][i]:
			net[name1][i][name2] = j
		
		if name2 not in net:
			net[name2] = {}
		if j not in net[name2]:
			net[name2][j] = {}
		if name1 not in net[name2][j]:
			net[name2][j][name1] = i
		

	with open(out, "w") as f:
		json.dump(net, f, indent=4)

commands = {
	"*" : (process, "Build stars networks", "descs_dir/ net.json"),
}

def run(argv):
	usage.run(argv, "stars net", commands)

