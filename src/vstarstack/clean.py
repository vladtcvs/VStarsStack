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

import vstarstack.common
import vstarstack.cfg

import os

def run(argv):
    orig    = vstarstack.cfg.config["paths"]["npy-orig"]
    fixed   = vstarstack.cfg.config["paths"]["npy-fixed"]
    shifted = vstarstack.cfg.config["paths"]["aligned"]
    
    for path in [orig, fixed, shifted]:
        files = vstarstack.common.listfiles(path, ".zip")
        for _, filename in files:
            print(filename)
            os.remove(filename)