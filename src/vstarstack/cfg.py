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

import sys
import math
import numpy as np
import os
import json
import multiprocessing as mp

import vstarstack.tools.camera
import vstarstack.tools.lens

def getval(config, name, default):
	if name in config:
		return config[name]
	return default

def get_param(name, type, default):
	for arg in sys.argv[2:]:
		if arg[:2] != "--":
			continue
		arg = arg[2:]
		items = arg.split("=")
		if len(items) != 2:
			continue
		if items[0] != name:
			continue
		return type(items[1])
	return default

debug = False
if "DEBUG" in os.environ:
	debug = eval(os.environ["DEBUG"])
	print("Debug = %s" % debug)



cfgdir = os.getcwd()
cfgpath = os.path.join(cfgdir, "project.json")
nthreads = max(int(mp.cpu_count())-1, 1)

if os.path.exists(cfgpath):
	with open(cfgpath) as f:
		config = json.load(f)

	use_sphere = getval(config, "use_sphere", True)
	compress = getval(config, "compress", True)

	if "stars" in config:
		stars      = config["stars"]
		use_angles = getval(stars, "use_angles", True)

	telescope = config["telescope"]

	camera    = vstarstack.tools.camera.Camera(telescope["camera"])
	scope      = vstarstack.tools.lens.Lens(telescope["scope"])

else:
	pass
