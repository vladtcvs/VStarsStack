"""Read source image files to internal format"""
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
import multiprocessing as mp
import logging

import vstarstack.tool.cfg
import vstarstack.tool.usage
import vstarstack.library.loaders.nef
import vstarstack.library.loaders.classic
import vstarstack.library.loaders.ser
import vstarstack.library.loaders.yuv
import vstarstack.library.loaders.video
import vstarstack.library.loaders.fits
import vstarstack.tool.common

from vstarstack.library.projection import ProjectionType
from vstarstack.library.projection.tools import add_description

logger = logging.getLogger(__name__)

def _work(reader,
          project: vstarstack.tool.cfg.Project,
          input_file: str,
          output_dir: str,
          output_name: str,
          mode: str):
    if mode == "dark" or mode == "flat":
        projection = ProjectionType.NoneProjection
        params = {}
    else:
        projection = ProjectionType.Perspective
        params = {
            "F": project.config.telescope.scope.F,
            "kh": project.config.telescope.camera.pixel_H / 1000,
            "kw": project.config.telescope.camera.pixel_W / 1000,
        }

    logger.info(f"Processing file: {input_file}")
    for frame_id, dataframe in enumerate(reader(input_file)):
        outfname = os.path.join(output_dir, f"{output_name}_{frame_id:06}.zip")
        add_description(dataframe, projection, **params)
        img_format = project.config.telescope.camera.format
        if img_format != "COPY":
            dataframe.add_parameter(img_format, "format")
        vstarstack.tool.common.check_dir_exists(outfname)
        dataframe.store(outfname)

def _process_file(reader, project: vstarstack.tool.cfg.Project, argv : list, mode : str):
    input_file = argv[0]
    output_dir = argv[1]
    output_name = argv[2]
    _work(reader, project, input_file, output_dir, output_name, mode)

def _process_path(reader, exts, project: vstarstack.tool.cfg.Project, argv : list, mode : str):
    """Process all files in directory"""
    input_dir = argv[0]
    output_dir = argv[1]
    files = []
    for ext in exts:
        files += vstarstack.tool.common.listfiles(input_dir, ext, recursive=True)

    with mp.Pool(vstarstack.tool.cfg.nthreads) as pool:
        args = [(reader, project, filename, output_dir, name, mode) for name, filename in files]
        pool.starmap(_work, args)

def _process_read(reader, exts, project: vstarstack.tool.cfg.Project, argv : list):
    mode = vstarstack.tool.cfg.get_param("mode", str, "light")
    if len(argv) > 0:
        source_path = argv[0]
        if os.path.isdir(source_path):
            _process_path(reader, exts, project, argv, mode)
        else:
            _process_file(reader, project, argv, mode)
    else:
        if mode == "light":
            _process_path(reader, exts, project, [
                        project.config.paths.light.original,
                        project.config.paths.light.npy,
                    ], mode)
        elif mode == "flat":
            _process_path(reader, exts, project, [
                        project.config.paths.flat.original,
                        project.config.paths.flat.npy,
                    ], mode)
        elif mode == "dark":
            _process_path(reader, exts, project, [
                        project.config.paths.dark.original,
                        project.config.paths.dark.npy,
                    ], mode)

def _read_nef(project: vstarstack.tool.cfg.Project, argv: list):
    reader = vstarstack.library.loaders.nef.readnef
    _process_read(reader, [".nef"], project, argv)

def _read_classic(project: vstarstack.tool.cfg.Project, argv: list):
    reader = vstarstack.library.loaders.classic.readjpeg
    _process_read(reader, [".jpg", ".png", ".tiff"], project, argv)

def _read_ser(project: vstarstack.tool.cfg.Project, argv: list):
    reader = vstarstack.library.loaders.ser.readser
    _process_read(reader, [".ser"], project, argv)

def _read_yuv(project: vstarstack.tool.cfg.Project, argv: list):
    reader = vstarstack.library.loaders.yuv.readyuv
    _process_read(reader, [".yuv"], project, argv)

def _read_fits(project: vstarstack.tool.cfg.Project, argv: list):
    reader = vstarstack.library.loaders.fits.readfits
    _process_read(reader, [".fits"], project, argv)

def _read_video(project: vstarstack.tool.cfg.Project, argv: list):
    reader = vstarstack.library.loaders.video.read_video
    _process_read(reader, [".avi"], project, argv)

commands = {
    "nef": (_read_nef, "read Nikon NEF"),
    "classic": (_read_classic, "read usual images (JPG, PNG, TIFF)"),
    "ser": (_read_ser, "read SER"),
    "yuv": (_read_yuv, "read YUV images"),
    "fits": (_read_fits, "read FITS images"),
    "video": (_read_video, "read VIDEO images"),
}
