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

import vstarstack.tool.cfg
import vstarstack.tool.usage

import vstarstack.library.common
import vstarstack.library.image_process.distorsion
import vstarstack.library.data

def dedistorsion(distorsion : vstarstack.library.image_process.distorsion.Distorsion,
                 name : str,
                 infname : str,
                 outfname : str):
    """Remove distorsion"""
    print(f"Processing {name}")
    dataframe = vstarstack.library.data.DataFrame.load(infname)
    dataframe = vstarstack.library.image_process.distorsion.fix_distorsion(dataframe, distorsion)
    dataframe.store(outfname)

def _process_file(distorsion : vstarstack.library.image_process.distorsion.Distorsion,
                 argv: list):
    infname = argv[0]
    outfname = argv[1]
    name = os.path.splitext(os.path.basename(infname))[0]
    dedistorsion(distorsion, name, infname, outfname)

def _process_dir(distorsion : vstarstack.library.image_process.distorsion.Distorsion,
                argv: list):
    inpath = argv[0]
    outpath = argv[1]
    files = vstarstack.library.common.listfiles(inpath, ".zip")
    with mp.Pool(vstarstack.tool.cfg.nthreads) as pool:
        pool.starmap(dedistorsion, [(distorsion, name, fname, os.path.join(
            outpath, name + ".zip")) for name, fname in files])

def _process(project: vstarstack.tool.cfg.Project, argv: list):
    a = project.distorsion["a"]
    b = project.distorsion["b"]
    c = project.distorsion["c"]

    distorsion = vstarstack.library.image_process.distorsion.Distorsion(a, b, c)
    if len(argv) > 0:
        if os.path.isdir(argv[0]):
            _process_dir(distorsion, argv)
        else:
            _process_file(distorsion, argv)
    else:
        _process_dir(distorsion, [project.config["paths"]["npy-fixed"],
                              project.config["paths"]["npy-fixed"]])

commands = {
    "*":  (_process, "Remove distrosion", "(input.file output.file | [input/ output/])"),
}

def run(project: vstarstack.tool.cfg.Project, argv: list):
    vstarstack.tool.usage.run(project, argv, "image-process distorsion", commands)
