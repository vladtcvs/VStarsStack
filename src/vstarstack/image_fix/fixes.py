"""Image fixes"""
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
import shutil

import vstarstack.common
import vstarstack.cfg
import vstarstack.usage

import vstarstack.image_fix.distorsion
import vstarstack.image_fix.remove_sky
import vstarstack.image_fix.border
import vstarstack.image_fix.normalize
import vstarstack.image_fix.motion_fix
import vstarstack.image_fix.deconvolution
import vstarstack.image_fix.useL

def copy(project: vstarstack.cfg.Project, _argv: list):
    """Copy files"""
    orig = project.config["paths"]["npy-orig"]
    fixed = project.config["paths"]["npy-fixed"]
    files = vstarstack.common.listfiles(orig, ".zip")
    for name, fname in files:
        print("Copying ", name)
        fname_out = os.path.join(fixed, name + ".zip")
        shutil.copyfile(fname, fname_out)


commands = {
    "copy": (copy, "just copy images from original to pipeline dir"),
    "distorsion": (vstarstack.image_fix.distorsion.run, "fix distorsion"),
    "remove-sky": (vstarstack.image_fix.remove_sky.run, "remove sky"),
    "border": (vstarstack.image_fix.border.run,     "remove border"),
    "normalize": (vstarstack.image_fix.normalize.run,  "normalize to weight"),
    "fix-motion": (vstarstack.image_fix.motion_fix.run, "remove motion of image"),
    "deconvolution": (vstarstack.image_fix.deconvolution.run, "deconvolution of image"),
    "useL": (vstarstack.image_fix.useL.run, "use L channel for brightness"),
}


def run(project: vstarstack.cfg.Project, argv: list):
    """Run image fix methods"""
    vstarstack.usage.run(project, argv, "image-fix", commands, autohelp=True)
