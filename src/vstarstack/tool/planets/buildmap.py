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
import logging

import vstarstack.library.data
import vstarstack.tool.cfg
import vstarstack.tool.usage
import vstarstack.library.common
import vstarstack.library.planetmap
import vstarstack.tool.common

logger = logging.getLogger(__name__)

def _process_file(project : vstarstack.tool.cfg.Project,
                 filename : str,
                 mapname : str):
    raise NotImplementedError()

def _process_path(project : vstarstack.tool.cfg.Project,
                 images_path : str,
                 maps_path : str):
    files = vstarstack.tool.common.listfiles(images_path, ".zip")
    for name, filename in files:
        logger.info(f"Processing {name}")
        out = os.path.join(maps_path, name + ".zip")
        _process_file(project, filename, out)

def run(project : vstarstack.tool.cfg.Project,
        argv : list[str]):
    if len(argv) > 0:
        input_path = argv[0]
        output_path = argv[1]
        if os.path.isdir(input_path):
            _process_path(project, input_path, output_path)
        else:
            _process_file(project, input_path, output_path)
    else:
        _process_path(project,
                     project.config.paths.aligned,
                     project.config.planets.paths.maps)
