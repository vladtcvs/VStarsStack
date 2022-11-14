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

import vstarstack.usage
import os
import json

def list_cfg(argv):
	pass

def dircheck(name):
	if not os.path.isdir(name):
		os.mkdir(name)

def configurate(argv):
	# create project directory
	dircheck(dir)

	# directory for original images (NEF, png, jpg, etc)
	dircheck(dir + "/orig")

	# directory for original images in NPZ format
	dircheck(dir + "/npy-orig")

	# directory for images after pre-processing (remove darks, sky, vignetting, distorsion, etc)
	dircheck(dir + "/npy")

	# directory for images after moving
	dircheck(dir + "/aligned")

	# directory for image descriptors
	dircheck(dir + "/descs")

	config = {
		"use_sphere" : True,
		"compress" : True,
		"paths" : {
			"original"  : "orig",
			"npy-orig"  : "npy-orig",
			"npy-fixed" : "npy",
			"descs"     : "descs",
			"aligned"   : "aligned",
			"output"    : "sum.zip",
		}
	}

	with open(os.path.join(cfgdir, telescope_fname)) as f:
		config["telescope"] = json.load(f)

	with open(dir + "/project.json", "w") as f:
		json.dump(config, f, indent=4, ensure_ascii=False)


commands = {
    "list" : (list_cfg, "list available telescope names", ""),
	"*" : (configurate, "create project", "project_dir telescope_name"),
}

def run(argv):
	usage.run(argv, "project", commands)
