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

import vstarstack.sky_model.gradient
import vstarstack.sky_model.gauss
import vstarstack.sky_model.isoline
import vstarstack.sky_model.quadratic

import vstarstack.usage
import vstarstack.common
import vstarstack.cfg
import vstarstack.data


def remove_sky(project, name, infname, outfname, model):
    """Remove sky from file"""
    print(name)

    img = vstarstack.data.DataFrame.load(infname)

    for channel in img.get_channels():

        image, opts = img.get_channel(channel)
        if not opts["brightness"]:
            continue

        print(f"\t{channel}")
        sky = model(project, image)

        result = image - sky
        img.add_channel(result, channel, **opts)

    img.store(outfname)


def process_file(project, argv, model):
    """Remove sky from single file"""
    infname = argv[0]
    outfname = argv[1]
    name = os.path.splitext(os.path.basename(infname))[0]
    remove_sky(project, name, infname, outfname, model)


def process_dir(project, argv, model):
    """Remove sky from all files in directory"""
    inpath = argv[0]
    outpath = argv[1]
    files = vstarstack.common.listfiles(inpath, ".zip")
    with mp.Pool(vstarstack.cfg.nthreads) as pool:
        pool.starmap(remove_sky, [(project, name, fname, os.path.join(
            outpath, name + ".zip"), model) for name, fname in files])


def process(project: vstarstack.cfg.Project, argv: list, model):
    """Process file(s) in path"""
    if len(argv) > 0:
        if os.path.isdir(argv[0]):
            process_dir(project, argv, model)
        else:
            process_file(project, argv, model)
    else:
        process_dir(project, [project.config["paths"]["npy-fixed"],
                              project.config["paths"]["npy-fixed"]], model)


commands = {
    "isoline": (lambda project, argv: process(project, argv,
                                              vstarstack.sky_model.isoline.model),
                "use isoline model"),
    "gauss": (lambda project, argv: process(project, argv,
                                            vstarstack.sky_model.gauss.model),
              "use gauss blur model"),
    "gradient": (lambda project, argv: process(project, argv,
                                               vstarstack.sky_model.gradient.model),
                 "use gradient model"),
    "quadratic": (lambda project, argv: process(project, argv,
                                                vstarstack.sky_model.quadratic.model),
                  "use quadratic gradient model"),
}


def run(project: vstarstack.cfg.Project, argv: list):
    """Run removing of sky"""
    vstarstack.usage.run(
        project, argv, "image-fix remove-sky", commands, autohelp=True)
