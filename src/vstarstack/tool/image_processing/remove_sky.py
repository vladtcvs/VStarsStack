"""Remove sky from the image"""
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

import vstarstack.tool.usage
import vstarstack.tool.cfg
import vstarstack.library.common
import vstarstack.library.data
import vstarstack.library.image_process.remove_sky
import vstarstack.tool.common

logger = logging.getLogger(__name__)

def remove_sky(name, infname, outfname, model):
    """Remove sky from file"""
    logger.info(f"Processing {name}")

    img = vstarstack.library.data.DataFrame.load(infname)
    vstarstack.library.image_process.remove_sky.remove_sky(img, model)
    vstarstack.tool.common.check_dir_exists(outfname)
    img.store(outfname)


def process_file(argv, model_name):
    """Remove sky from single file"""
    infname = argv[0]
    outfname = argv[1]
    name = os.path.splitext(os.path.basename(infname))[0]
    remove_sky(name, infname, outfname, model_name)


def process_dir(argv, model_name):
    """Remove sky from all files in directory"""
    inpath = argv[0]
    outpath = argv[1]
    files = vstarstack.tool.common.listfiles(inpath, ".zip")
    with mp.Pool(vstarstack.tool.cfg.nthreads) as pool:
        pool.starmap(remove_sky, [(name, fname, os.path.join(
            outpath, name + ".zip"), model_name) for name, fname in files])


def process(project: vstarstack.tool.cfg.Project, argv: list, model_name : str):
    """Process file(s) in path"""
    if len(argv) > 0:
        if os.path.isdir(argv[0]):
            process_dir(argv, model_name)
        else:
            process_file(argv, model_name)
    else:
        process_dir([project.config.paths.light.npy,
                     project.config.paths.light.npy], model_name)

commands = {
    "isoline": (lambda project, argv: process(project, argv, "isoline"),
                "use isoline model"),
    "gradient": (lambda project, argv: process(project, argv, "gradient"),
                 "use gradient model"),
    "quadratic": (lambda project, argv: process(project, argv, "quadratic"),
                  "use quadratic gradient model"),
}
