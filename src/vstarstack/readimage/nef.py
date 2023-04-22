"""Reading NEF image files"""
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

import rawpy
import numpy as np

import vstarstack.cfg
import vstarstack.common
import vstarstack.data
import vstarstack.usage

import vstarstack.readimage.tags


def readnef(project: vstarstack.cfg.Project, filename: str, output: str):
    """Read NEF file"""
    img = rawpy.imread(filename)
    image = img.raw_image_visible

    tags = vstarstack.readimage.tags.read_tags(filename)

    params = {
        "w": image.data.shape[1],
        "h": image.data.shape[0],
        "projection": "perspective",
        "perspective_F": project.scope.F,
        "perspective_kh": project.camera.kh,
        "perspective_kw": project.camera.kw,
        "format": project.camera.format,
    }

    exptime = tags["shutter"]*tags["iso"]

    dataframe = vstarstack.data.DataFrame(params, tags)

    weight = np.ones(image.data.shape)*exptime

    dataframe.add_channel(image, "raw", encoded=True)
    dataframe.add_channel(weight, "weight")
    dataframe.add_channel_link("raw", "weight", "weight")

    dataframe.store(output)


def work(project: vstarstack.cfg.Project, input_file: str, output_file: str):
    """Process file"""
    print(input_file)
    readnef(project, input_file, output_file)


def process_file(project: vstarstack.cfg.Project, argv: list):
    """Process single file"""
    input_file = argv[0]
    output_file = argv[1]
    work(project, input_file, output_file)


def process_path(project: vstarstack.cfg.Project, argv: list):
    """Process all files in directory"""
    input_dir = argv[0]
    output_dir = argv[1]
    files = vstarstack.common.listfiles(input_dir, ".nef")
    ncpu = max(int(mp.cpu_count())-1, 1)
    with mp.Pool(ncpu) as pool:
        pool.starmap(work, [(project, filename, os.path.join(
            output_dir, name + ".zip")) for name, filename in files])


def process(project: vstarstack.cfg.Project, argv: list):
    """Process reading files"""
    if len(argv) > 0:
        source_path = argv[0]
        if os.path.isdir(source_path):
            process_path(project, argv)
        else:
            process_file(project, argv)
    else:
        process_path(project, [project.config["paths"]
                     ["original"], project.config["paths"]["npy-orig"]])


commands = {
    "*": (process, "read NEF to npy", "(input.NEF output.zip| [original/ npy/])"),
}


def run(project: vstarstack.cfg.Project, argv: list):
    """Run reading NEF files"""
    vstarstack.usage.run(project, argv, "readimage nef",
                         commands, autohelp=False)
