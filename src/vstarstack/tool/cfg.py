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
import json
import os
import multiprocessing as mp

import vstarstack.tool.devices.camera
import vstarstack.tool.devices.lens

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


DEBUG = False
if "DEBUG" in os.environ:
    DEBUG = os.environ["DEBUG"].lower() == "true"
    print("Debug = {debug}")

nthreads = max(int(mp.cpu_count())-1, 1)


class ConfigException(Exception):
    """Config error exception"""

    def __init__(self, reason):
        Exception.__init__(self, f"Config error: {reason}")


class Project(object):
    def __init__(self, config):
        self.config = config
        self.use_sphere = getval(config, "use_sphere", True)
        self.compress = getval(config, "compress", True)

        if "stars" in config:
            self.stars = config["stars"]
        else:
            self.stars = {}

        telescope = config["telescope"]
        self.camera = vstarstack.tool.devices.camera.Camera(telescope["camera"])
        self.scope = vstarstack.tool.devices.lens.Lens(telescope["scope"])

_PROJECT = None


def get_project(filename=None):
    """Load project file"""
    global _PROJECT

    if filename is None:
        cfgdir = os.getcwd()
        filename = os.path.join(cfgdir, "project.json")

    if _PROJECT is None and os.path.exists(filename):
        with open(filename, encoding='utf8') as f:
            config = json.load(f)
        _PROJECT = Project(config)
    return _PROJECT
