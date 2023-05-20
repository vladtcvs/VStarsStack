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
import json

import vstarstack.tool.usage
import vstarstack.tool.cfg

def dircheck(name):
    """Check existance of dir and create, if required"""
    if not os.path.isdir(name):
        os.mkdir(name)

def _configurate(argv):
    directory = argv[0]

    # create project directory
    dircheck(directory)

    # directory for original images (NEF, png, jpg, etc)
    dircheck(directory + "/orig")

    # directory for original images in NPZ format
    dircheck(directory + "/npy-orig")

    # directory for images after pre-processing (remove darks, sky, vignetting, distorsion, etc)
    dircheck(directory + "/npy")

    # directory for images after moving
    dircheck(directory + "/aligned")

    # directory for image descriptors
    dircheck(directory + "/descs")

    config = {
        "use_sphere": True,
        "compress": True,
        "paths": {
            "original": "orig",
            "npy-orig": "npy-orig",
            "npy-fixed": "npy",
            "descs": "descs",
            "aligned": "aligned",
            "output": "sum.zip",
        },
        "telescope": {
            "camera": {
                "W": 10.0,
                "H": 10.0,
                "w": 1000,
                "h": 1000,
            },
            "scope": {
                "F": 1000.0,
                "D": 100.0,
            },
        }
    }

    with open(directory + "/project.json", "w", encoding="utf8") as file:
        json.dump(config, file, indent=4, ensure_ascii=False)


commands = {
    "create": (_configurate, "create project", "project_dir"),
}

def run(project: vstarstack.tool.cfg.Project, argv: list):
    vstarstack.tool.usage.run(project, argv, "project", commands)
