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

import vstarstack.targets.stars.detect
import vstarstack.targets.stars.describe
import vstarstack.targets.stars.match
import vstarstack.targets.stars.net
import vstarstack.targets.stars.cluster
import vstarstack.targets.stars.lonlat
import vstarstack.targets.stars.process
import vstarstack.targets.stars.configure
import sys
import vstarstack.usage

commands = {
	"config"   : (vstarstack.targets.stars.configure.run, "configure stars pipeline"),
	"detect"   : (vstarstack.targets.stars.detect.run, "detect stars"),
	"lonlat"   : (vstarstack.targets.stars.lonlat.run, "fill longitude and latitude"),
	"describe" : (vstarstack.targets.stars.describe.run, "find descriptions for each image"),
	"match"    : (vstarstack.targets.stars.match.run, "match stars between images"),
	"net"      : (vstarstack.targets.stars.net.run, "build net of matching"),
	"cluster"  : (vstarstack.targets.stars.cluster.run, "find matching stars clusters between images"),
	"process"  : (vstarstack.targets.stars.process.run, "run process - all commands above"),
}

def run(argv):
	usage.run(argv, "stars", commands, autohelp=True)

