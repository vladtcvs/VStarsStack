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

import vstarstack.tool.config

from vstarstack.tool.configuration import Configuration

def get_param(name, type_of_var, default):
    """Get cmdline parameter --name=value"""
    for arg in sys.argv[1:]:
        if arg[:2] != "--":
            continue
        arg = arg[2:]
        items = arg.split("=")
        if len(items) != 2:
            continue
        if items[0] != name:
            continue
        if type_of_var == bool:
            value = (items[1] == "True")
        else:
            value = type_of_var(items[1])
        return value
    return default


DEBUG = False
if "DEBUG" in os.environ:
    DEBUG = os.environ["DEBUG"].lower() == "true"
    print("Debug = {debug}")

nthreads = max(int(mp.cpu_count())-1, 1)

class Project(object):
    """Holder for configuration"""
    def __init__(self, config_data : dict = None):
        self.config = Configuration(vstarstack.tool.config._module_configuration)
        self.updated = False
        if config_data is not None:
            self.updated = self.config.load_configuration(config_data)

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
        if _PROJECT.updated:
            print("Config updated, saving")
            try:
                with open(filename, "w", encoding='utf8') as f:
                    json.dump(_PROJECT.config.write_configuration(),
                              f, indent=4, ensure_ascii=False)
            except:
                print("Can't update project file")
    return _PROJECT

def store_project(project : Project = None, filename=None):
    """Store project file"""
    if project is None:
        project = _PROJECT
    data = project.config
    data = data.write_configuration()

    if filename is None:
        cfgdir = os.getcwd()
        filename = os.path.join(cfgdir, "project.json")

    with open(filename, "w", encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
