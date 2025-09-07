"""Image fixes"""
#
# Copyright (c) 2023-2024 Vladislav Tsendrovskii
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
import logging

import vstarstack.library.common
import vstarstack.tool.cfg
import vstarstack.tool.usage

import vstarstack.tool.image_processing.distorsion
import vstarstack.tool.image_processing.remove_sky
import vstarstack.tool.image_processing.border
import vstarstack.tool.image_processing.normalize
import vstarstack.tool.image_processing.blur
import vstarstack.tool.image_processing.drop_unsharp
import vstarstack.tool.image_processing.deconvolution
import vstarstack.tool.image_processing.remove_continuum
import vstarstack.tool.common

logger = logging.getLogger(__name__)

def copy(project: vstarstack.tool.cfg.Project, argv: list):
    """Copy files"""
    orig = argv[0]
    fixed = argv[1]

    files = vstarstack.tool.common.listfiles(orig, ".zip")
    for name, fname in files:
        logger.info(f"Copying {name} to {fixed} dir")
        fname_out = os.path.join(fixed, name + ".zip")
        shutil.copyfile(fname, fname_out)

commands = {
    "copy": (copy, "just copy images from original to pipeline dir", "source/ destination/"),
    "distorsion": (vstarstack.tool.image_processing.distorsion.run, "fix distorsion"),
    "remove-sky": (vstarstack.tool.image_processing.remove_sky.commands, "remove sky"),
    "border": (vstarstack.tool.image_processing.border.run,     "remove border"),
    "normalize": (vstarstack.tool.image_processing.normalize.run,  "normalize to weight"),
    "blur": (vstarstack.tool.image_processing.blur.run,  "gaussian blur"),
    "deconvolution": (vstarstack.tool.image_processing.deconvolution.commands,  "deconvolution"),
    "select-sharp" : (vstarstack.tool.image_processing.drop_unsharp.commands, "select sharp images"),
    "remove-continuum" : (vstarstack.tool.image_processing.remove_continuum.process, "remove continuum", "input.zip Narrow Wide output.zip [coeff]"),
}
