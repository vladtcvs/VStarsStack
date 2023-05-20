"""Image fixes"""
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

import os
import shutil

import vstarstack.library.common
import vstarstack.tool.cfg
import vstarstack.tool.usage

import vstarstack.tool.image_processing.distorsion
import vstarstack.tool.image_processing.remove_sky
import vstarstack.tool.image_processing.border
import vstarstack.tool.image_processing.normalize

def copy(project: vstarstack.tool.cfg.Project, argv: list):
    """Copy files"""
    if len(argv) > 2:
        orig = argv[0]
        fixed = argv[1]
    else:
        orig = project.config["paths"]["npy-orig"]
        fixed = project.config["paths"]["npy-fixed"]
    files = vstarstack.library.common.listfiles(orig, ".zip")
    for name, fname in files:
        print("Copying ", name)
        fname_out = os.path.join(fixed, name + ".zip")
        shutil.copyfile(fname, fname_out)

commands = {
    "copy": (copy, "just copy images from original to pipeline dir"),
    "distorsion": (vstarstack.tool.image_processing.distorsion.run, "fix distorsion"),
    "remove-sky": (vstarstack.tool.image_processing.remove_sky.run, "remove sky"),
    "border": (vstarstack.tool.image_processing.border.run,     "remove border"),
    "normalize": (vstarstack.tool.image_processing.normalize.run,  "normalize to weight"),
}

def run(project: vstarstack.tool.cfg.Project, argv: list):
    """Run image fix methods"""
    vstarstack.tool.usage.run(project, argv, "image-fix", commands, autohelp=True)
