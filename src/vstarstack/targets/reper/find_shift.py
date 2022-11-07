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

from movement_flat import Movement
import sys
import json

def run(argv):
	with open(argv[0]) as f:
		clusters = json.load(f)
	cluster = clusters[0]
	movements = {}
	for name1 in cluster:
		y1 = cluster[name1]["y"]
		x1 = cluster[name1]["x"]
		movements[name1] = {}
		for name2 in cluster:
			y2 = cluster[name2]["y"]
			x2 = cluster[name2]["x"]
			t = Movement(0, y2 - y1, x2 - x1)
			movements[name1][name2] = t.serialize()

	data = {
		"shift_type" : "flat",
		"movements" : movements
	}
	with open(argv[1], "w") as f:
		json.dump(data, f, indent=4)


if __name__ == "__main__":
	run(sys.argv[1:])
