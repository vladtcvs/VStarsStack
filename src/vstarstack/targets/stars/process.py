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

import os
import vstarstack.cfg

import vstarstack.targets.stars.detect
import vstarstack.targets.stars.describe
import vstarstack.targets.stars.match
import vstarstack.targets.stars.net
import vstarstack.targets.stars.cluster
import vstarstack.targets.stars.lonlat
import sys
import vstarstack.usage

def process(argv):
	basedir = os.getcwd()
	if len(argv) > 0:
		first_command = argv[0]
		if first_command == "detect":
			level = 0
		elif first_command == "lonlat":
			level = 1
		elif first_command == "describe":
			level = 2
		elif first_command == "match":
			level = 3
		elif first_command == "net":
			level = 4
		elif first_command == "cluster":
			level = 5
	else:
		level = 0

	npydir = os.path.join(basedir, vstarstack.cfg.config["paths"]["npy-fixed"])

	starsdir = os.path.join(basedir, vstarstack.cfg.config["stars"]["paths"]["stars"])
	if not os.path.exists(starsdir):
		os.mkdir(starsdir)

	descdir = os.path.join(basedir, vstarstack.cfg.config["stars"]["paths"]["descs"])	
	if not os.path.exists(descdir):
		os.mkdir(descdir)

	net = os.path.join(basedir, vstarstack.cfg.config["stars"]["paths"]["net"])
	clusters = os.path.join(basedir, vstarstack.cfg.config["cluster"]["path"])

	if level <= 0:
		print("Detect")
		targets.stars.detect.run([npydir, starsdir])
	if level <= 1:
		print("Lonlat")
		targets.stars.lonlat.run([starsdir])
	if level <= 2:
		print("Describe")
		targets.stars.describe.run([starsdir, descdir])
	if level <= 3:
		print("Match")
		targets.stars.match.run([descdir])
	if level <= 4:
		print("Net")
		targets.stars.net.run([descdir, net])
	if level <= 5:
		print("Cluster")
		targets.stars.cluster.run([net, descdir, clusters])

commands = {
	"*" : (process, "Process stars: detect, lonlat, describe, match, net, cluster", "[first_command]"),
}

def run(argv):
	usage.run(argv, "stars process", commands)
