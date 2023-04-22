"""Reading common image files: jpg/png/tiff"""
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
import numpy as np

from PIL import Image
import vstarstack.cfg
import vstarstack.common
import vstarstack.usage
import vstarstack.data

import vstarstack.readimage.tags


def readjpeg(project: vstarstack.cfg.Project, fname: str):
    """Read single image (jpg, png, tiff) file"""
    rgb = np.asarray(Image.open(fname)).astype(np.float32)
    shape = rgb.shape
    shape = (shape[0], shape[1])

    tags = vstarstack.readimage.tags.read_tags(fname)
    params = {
        "w": shape[1],
        "h": shape[0],
        "projection": "perspective",
        "perspective_F": project.scope.F,
        "perspective_kh": project.camera.kh,
        "perspective_kw": project.camera.kw,
        "format": project.camera.format,
    }

    try:
        exposure = tags["shutter"]*tags["iso"]
    except KeyError as _:
        exposure = 1

    weight = np.ones((shape[0], shape[1]))*exposure

    dataframe = vstarstack.data.DataFrame(params, tags)
    dataframe.add_channel(weight, "weight", weight=True)

    if len(rgb.shape) == 3:
        dataframe.add_channel(rgb[:, :, 0], "R", brightness=True)
        dataframe.add_channel(rgb[:, :, 1], "G", brightness=True)
        dataframe.add_channel(rgb[:, :, 2], "B", brightness=True)
        dataframe.add_channel_link("R", "weight", "weight")
        dataframe.add_channel_link("G", "weight", "weight")
        dataframe.add_channel_link("B", "weight", "weight")
    elif len(rgb.shape) == 2:
        dataframe.add_channel(
            rgb[:, :], "Y", weight_name="weight", brightness=True)
        dataframe.add_channel_link("Y", "weight", "weight")
    else:
        # unknown shape!
        pass
    return dataframe


def process_file(project: vstarstack.cfg.Project, argv: list):
    """Process single file"""
    fname = argv[0]
    output = argv[1]
    dataframe = readjpeg(project, fname)
    dataframe.store(output)


def process_path(project: vstarstack.cfg.Project, argv: list):
    """Process all files in directory"""
    input_dir = argv[0]
    output_dir = argv[1]

    jpgs = vstarstack.common.listfiles(input_dir, ".jpg")
    pngs = vstarstack.common.listfiles(input_dir, ".png")
    tiffs = vstarstack.common.listfiles(input_dir, ".tiff")
    files = jpgs + pngs + tiffs
    for name, fname in files:
        print(name)
        process_file(project, (fname, os.path.join(output_dir, name + '.zip')))


def process(project: vstarstack.cfg.Project, argv: list):
    """Process: read image files to npy"""
    if len(argv) > 0:
        source = argv[0]
        if os.path.isdir(source):
            process_path(project, argv)
        else:
            process_file(project, argv)
    else:
        process_path(project, [project.config["paths"]
                     ["original"], project.config["paths"]["npy-orig"]])


commands = {
    "*": (process, "read JPEG to npy", "(input.jpg output.zip | [original/ npy/])"),
}


def run(project: vstarstack.cfg.Project, argv: list):
    """Run reading jpg/png/tiff"""
    vstarstack.usage.run(project, argv, "readimage jpeg",
                         commands, autohelp=False)
