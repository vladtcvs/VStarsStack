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

import targets.stars.detect
import targets.stars.describe
import targets.stars.match
import targets.stars.net
import targets.stars.cluster
import targets.stars.lonlat
import targets.stars.process
import targets.stars.configure
import sys
import usage

commands = {
	"config"   : (targets.stars.configure.run, "configure stars pipeline"),
	"detect"   : (targets.stars.detect.run, "detect stars"),
	"lonlat"   : (targets.stars.lonlat.run, "fill longitude and latitude"),
	"describe" : (targets.stars.describe.run, "find descriptions for each image"),
	"match"    : (targets.stars.match.run, "match stars between images"),
	"net"      : (targets.stars.net.run, "build net of matching"),
	"cluster"  : (targets.stars.cluster.run, "find matching stars clusters between images"),
	"process"  : (targets.stars.process.run, "run process - all commands above"),
}

def run(argv):
	usage.run(argv, "stars", commands, autohelp=True)

