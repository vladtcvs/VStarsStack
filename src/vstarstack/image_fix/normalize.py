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

import vstarstack.usage
import os
import vstarstack.common
import vstarstack.data
import vstarstack.cfg
import numpy as np

import multiprocessing as mp
ncpu = max(int(mp.cpu_count())-1, 1)


def normalize(name, infname, outfname):
    print(name)
    img = vstarstack.data.DataFrame.load(infname)

    for channel in img.get_channels():
        image, opts = img.get_channel(channel)
        if "normalized" in opts and opts["normalized"]:
            continue
        if opts["weight"]:
            continue
        if opts["encoded"]:
            continue
        if channel not in img.links["weight"]:
            continue
        weight, _ = img.get_channel(img.links["weight"][channel])
        image = image / weight
        image[np.where(weight == 0)] = 0
        opts["normalized"] = True
        img.add_channel(image, channel, **opts)

    img.store(outfname)


def process_file(argv):
    infname = argv[0]
    outfname = argv[1]
    name = os.path.splitext(os.path.basename(infname))[0]
    normalize(name, infname, outfname)


def process_dir(argv):
    inpath = argv[0]
    outpath = argv[1]
    files = vstarstack.common.listfiles(inpath, ".zip")
    pool = mp.Pool(ncpu)
    pool.starmap(normalize, [(name, fname, os.path.join(
        outpath, name + ".zip")) for name, fname in files])
    pool.close()


def process(project: vstarstack.cfg.Project, argv: list):
    if len(argv) > 0:
        if os.path.isdir(argv[0]):
            process_dir(argv)
        else:
            process_file(argv)
    else:
        process_dir([project.config["paths"]["npy-fixed"],
                     project.config["paths"]["npy-fixed"]])


def run(project: vstarstack.cfg.Project, argv: list):
    process(project, argv)
