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
import vstarstack.library.data
import vstarstack.library.image_process.blur

import vstarstack.tool.common

def blur(name, fname, outname, size):
    print(name)

    img = vstarstack.library.data.DataFrame.load(fname)
    img = vstarstack.library.image_process.blur.blur(img, size)
    vstarstack.tool.common.check_dir_exists(outname)
    img.store(outname)

def process_file(argv):
    infile = argv[0]
    outfile = argv[1]
    size = int(argv[2])
    name = os.path.splitext(os.path.basename(infile))[0]
    blur(name, infile, outfile, size)


def process_dir(argv):
    inpath = argv[0]
    outpath = argv[1]
    size = int(argv[2])
    files = vstarstack.tool.common.listfiles(inpath, ".zip")
    args = [(name, fname, os.path.join(outpath, name + ".zip"), size) for name, fname in files]
    with mp.Pool(vstarstack.tool.cfg.nthreads) as pool:
        pool.starmap(blur, args)

def run(project: vstarstack.tool.cfg.Project, argv: list):
    if len(argv) > 0:
        if os.path.isdir(argv[0]):
            process_dir(argv)
        else:
            process_file(argv)
    else:
        process_dir([project.config.paths.npy_fixed,
                     project.config.paths.npy_fixed])
