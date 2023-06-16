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
import vstarstack.library.image_process.border


def border(name, fname, outname, bw_left, bw_top, bw_right, bw_bottom):
    print(name)

    img = vstarstack.library.data.DataFrame.load(fname)
    img = vstarstack.library.image_process.border.border(img,
                                                         bw_left, bw_top,
                                                         bw_right, bw_bottom)
    img.store(outname)

def process_file(argv):
    infile = argv[0]
    outfile = argv[1]
    bbw = argv[2:]
    if len(bbw) >= 4:
        brd_left = int(bbw[0])
        brd_top = int(bbw[1])
        brd_right = int(bbw[2])
        brd_bottom = int(bbw[3])
    elif len(bbw) > 0:
        brd_left = int(bbw[0])
        brd_top = int(bbw[0])
        brd_right = int(bbw[0])
        brd_bottom = int(bbw[0])

    name = os.path.splitext(os.path.basename(infile))[0]

    border(name, infile, outfile, brd_left, brd_top, brd_right, brd_bottom)


def process_dir(argv):
    inpath = argv[0]
    outpath = argv[1]
    bbw = argv[2:]
    if len(bbw) >= 4:
        brd_left = int(bbw[0])
        brd_top = int(bbw[1])
        brd_right = int(bbw[2])
        brd_bottom = int(bbw[3])
    elif len(bbw) > 0:
        brd_left = int(bbw[0])
        brd_top = int(bbw[0])
        brd_right = int(bbw[0])
        brd_bottom = int(bbw[0])

    files = vstarstack.library.common.listfiles(inpath, ".zip")
    with mp.Pool(vstarstack.tool.cfg.nthreads) as pool:
        pool.starmap(border, [(name, fname, os.path.join(outpath, name + ".zip"),
                 brd_left, brd_top, brd_right, brd_bottom) for name, fname in files])


def process(project: vstarstack.tool.cfg.Project, argv: list):
    if len(argv) > 0:
        if os.path.isdir(argv[0]):
            process_dir(argv)
        else:
            process_file(argv)
    else:
        process_dir([project.config["paths"]["npy-fixed"],
                     project.config["paths"]["npy-fixed"]])


commands = {
    "*": (process, "remove border", "(input.zip output.zip | [input/ output/])"),
}


def run(project: vstarstack.tool.cfg.Project, argv: list):
    vstarstack.tool.usage.run(project, argv, "image-process border", commands)
