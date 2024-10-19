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
import vstarstack.library.merge

import vstarstack.library.merge.simple_add
import vstarstack.library.merge.kappa_sigma
import vstarstack.tool.cfg
import vstarstack.tool.usage
import vstarstack.tool.common

from vstarstack.library.common import FilesImageSource

def simple_add(project: vstarstack.tool.cfg.Project, argv: list):
    """Calculate simple sum of images"""
    if len(argv) > 0:
        path_images = argv[0]
        out = argv[1]
    else:
        path_images = project.config.paths.aligned
        out = project.config.paths.light.result

    imgs = vstarstack.tool.common.listfiles(path_images, ".zip")
    filenames = [img[1] for img in imgs]
    dataframe = vstarstack.library.merge.simple_add.simple_add(FilesImageSource(filenames))
    if dataframe is not None:
        vstarstack.tool.common.check_dir_exists(out)
        dataframe.store(out)

def sigma_clip(project: vstarstack.tool.cfg.Project, argv: list):
    """Calculate sigma clipped sum of images"""
    if len(argv) > 0:
        path_images = argv[0]
        out = argv[1]
        kappa1 = float(argv[2])
        kappa2 = kappa1
        sigma_steps = 1
    else:
        path_images = project.config.paths.aligned
        out = project.config.paths.light.result
        kappa1 = project.config.merge.sigma_clip_coefficient_begin
        kappa2 = project.config.merge.sigma_clip_coefficient_end
        sigma_steps = project.config.merge.sigma_clip_steps

    imgs = vstarstack.tool.common.listfiles(path_images, ".zip")
    filenames = [img[1] for img in imgs]
    dataframe = vstarstack.library.merge.kappa_sigma.kappa_sigma(FilesImageSource(filenames),
                                                      kappa1,
                                                      kappa2,
                                                      sigma_steps)
    vstarstack.tool.common.check_dir_exists(out)
    dataframe.store(out)

commands = {
    "simple": (simple_add, "simple add images"),
    "sigma-clip": (sigma_clip, "add images with sigma clipping"),
}
