"""Reading FITS files"""
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
import numpy as np

from astropy.io import fits

import vstarstack.cfg
import vstarstack.common
import vstarstack.usage
import vstarstack.data


def process_file(project: vstarstack.cfg.Project, argv: list):
    """Read fits image file"""
    fname = argv[0]
    output = argv[1]
    name = argv[2]

    print(f"Opening {fname}")
    try:
        images = fits.open(fname)
    except Exception as excpt:
        print(f"Error reading file: {excpt}")
        return

    for plane_id in range(1):
        print(plane_id)
        image = images[plane_id]

        tags = {}
        for key in image.header:
            val = str(image.header[key])
            tags[key] = val

        shape = image.data.shape
        if len(shape) == 2:
            original = image.data.reshape((1, shape[0], shape[1]))
        else:
            original = image.data
        shape = original.shape

        params = {
            "w": shape[2],
            "h": shape[1],
            "projection": "perspective",
            "perspective_F": project.scope.F,
            "perspective_kh": project.camera.kh,
            "perspective_kw": project.camera.kw,
            "format": project.camera.format,
        }

        dataframe = vstarstack.data.DataFrame(params, tags)

        exptime = image.header["EXPTIME"]

        slice_names = []

        weight_channel_name = "weight"
        weight = np.ones((shape[1], shape[2]))*exptime
        dataframe.add_channel(weight, weight_channel_name, weight=True)

        if shape[0] == 1:
            if "FILTER" in image.header:
                channel_name = image.header["FILTER"].strip()
            else:
                channel_name = "Y"
            slice_names.append(channel_name)
        elif shape[0] == 3:
            slice_names.append('R')
            slice_names.append('G')
            slice_names.append('B')
        else:
            print("Unknown image format, skip")
            return

        for i, slice_name in enumerate(slice_names):
            dataframe.add_channel(
                original[i, :, :], slice_name, brightness=True)
            dataframe.add_channel_link(
                slice_name, weight_channel_name, "weight")

        framename = os.path.join(output, f"{name}.zip")
        dataframe.store(framename)


def process_path(project: vstarstack.cfg.Project, argv: list):
    """Read all fits files in directory"""
    input_dir = argv[0]
    output_dir = argv[1]

    files = vstarstack.common.listfiles(input_dir, ".fits")
    for name, fname in files:
        print(name)
        process_file(project, (fname, output_dir, name))


def process(project: vstarstack.cfg.Project, argv: list):
    """Read all source fits files"""
    if len(argv) > 0:
        source = argv[0]
        destination = argv[1]
        if os.path.isdir(source):
            process_path(project, (source, destination))
        else:
            name = os.path.splitext(os.path.basename(source))[0]
            process_file(project, (source, destination, name))
    else:
        process_path(project, [project.config["paths"]
                     ["original"], project.config["paths"]["npy-orig"]])


commands = {
    "*": (process, "read FITS to npy", "(input.fits output/ | [original/ npy/])"),
}


def run(project: vstarstack.cfg.Project, argv: list):
    """run reading fits files"""
    vstarstack.usage.run(project, argv, "readimage fits",
                         commands, autohelp=False)
