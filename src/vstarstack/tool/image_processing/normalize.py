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
import multiprocessing as mp

import vstarstack.tool.usage
import vstarstack.tool.cfg
import vstarstack.library.common
import vstarstack.library.data
import vstarstack.library.image_process.normalize
import vstarstack.tool.common

def normalize(name, infname, outfname):
    """Normalize image"""
    print(name)
    img = vstarstack.library.data.DataFrame.load(infname)
    img = vstarstack.library.image_process.normalize.normalize(img)
    vstarstack.tool.common.check_dir_exists(outfname)
    img.store(outfname)

def _process_file(argv):
    infname = argv[0]
    if len(argv) > 1:
        outfname = argv[1]
    else:
        outfname = infname
    name = os.path.splitext(os.path.basename(infname))[0]
    normalize(name, infname, outfname)


def _process_dir(argv):
    inpath = argv[0]
    if len(argv) > 1:
        outpath = argv[1]
    else:
        outpath = inpath
    files = vstarstack.tool.common.listfiles(inpath, ".zip")
    with mp.Pool(vstarstack.tool.cfg.nthreads) as pool:
        pool.starmap(normalize, [(name, fname, os.path.join(
            outpath, name + ".zip")) for name, fname in files])

def _process(project: vstarstack.tool.cfg.Project, argv: list):
    if len(argv) > 0:
        if os.path.isdir(argv[0]):
            _process_dir(argv)
        else:
            _process_file(argv)
    else:
        _process_dir([project.config.paths.light.npy,
                      project.config.paths.light.npy])

def run(project: vstarstack.tool.cfg.Project, argv: list):
    _process(project, argv)
