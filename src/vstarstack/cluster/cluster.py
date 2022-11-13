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

import vstarstack.cluster.lonlat
import vstarstack.cluster.find_shift
import vstarstack.usage

commands = {
	"find-shift" : (vstarstack.cluster.find_shift.run, "Find shifts from cluster file", "cluster.json shifts.json"),
	"lonlat"    : (vstarstack.cluster.lonlat.run, "Calculate (lat,lon) from (y,x) in cluster file", "cluster.json [cluster_out.json]"),
}

def run(argv):
	vstarstack.usage.run(argv, "cluster", commands, autohelp=True)
