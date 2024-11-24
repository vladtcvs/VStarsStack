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
import vstarstack.tool.common
import vstarstack.tool.cfg
logger = logging.getLogger(__name__)


def run(project: vstarstack.tool.cfg.Project, _argv: list):
    orig = project.config.paths.light.npy
    shifted = project.config.paths.aligned

    for path in [orig, shifted]:
        files = vstarstack.tool.common.listfiles(path, ".zip")
        for _, filename in files:
            logger.info(f"Remove {filename}")
            os.remove(filename)
