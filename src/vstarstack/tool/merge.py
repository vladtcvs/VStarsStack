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

import vstarstack.library.common
import vstarstack.library.data
import vstarstack.library.merging

import vstarstack.tool.cfg
import vstarstack.tool.usage

from vstarstack.library.common import FilesImageSource

def simple_add(project: vstarstack.tool.cfg.Project, argv: list):
    """Calculate simple sum of images"""
    if len(argv) > 0:
        path_images = argv[0]
        out = argv[1]
    else:
        path_images = project.config["paths"]["aligned"]
        out = project.config["paths"]["output"]

    imgs = vstarstack.library.common.listfiles(path_images, ".zip")
    filenames = [img[1] for img in imgs]
    dataframe = vstarstack.library.merging.simple_add(FilesImageSource(filenames))
    dataframe.store(out)

def sigma_clip(project: vstarstack.tool.cfg.Project, argv: list):
    """Calculate sigma clipped sum of images"""
    if len(argv) > 0:
        path_images = argv[0]
        out = argv[1]
        sigma_k = float(argv[2])
        sigma_steps = 3
    else:
        path_images = project.config["paths"]["aligned"]
        out = project.config["paths"]["output"]
        sigma_k = project.config["sigma_clip_coefficient"]
        sigma_steps = project.config["sigma_clip_steps"]

    imgs = vstarstack.library.common.listfiles(path_images, ".zip")
    filenames = [img[1] for img in imgs]
    dataframe = vstarstack.library.merging.sigma_clip(FilesImageSource(filenames),
                                                      sigma_k,
                                                      sigma_steps)
    dataframe.store(out)

commands = {
    "simple": (simple_add, "simple add images"),
    "sigma-clip": (sigma_clip, "add images with sigma clipping"),
}

def run(project: vstarstack.tool.cfg.Project, argv: list):
    vstarstack.tool.usage.run(project, argv, "merge", commands, autohelp=True)
