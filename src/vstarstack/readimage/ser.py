"""Read SER images"""
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
import math
import numpy as np

import vstarstack.cfg
import vstarstack.common
import vstarstack.usage
import vstarstack.data

import vstarstack.readimage.tags


def serread(file, integer_size, little_endian):
    """Read integer from SER file"""
    block = list(file.read(integer_size))
    if little_endian:
        block = block[::-1]
    val = 0
    for i in range(integer_size):
        val *= 256
        val += block[i]
    return val


def serread4(file):
    """Read 4-byte integer"""
    return serread(file, 4, True)


def pixelread(file, bpp, little_endian, colorid):
    """Read single pixel form SER"""
    if colorid == 0:
        return serread(file, bpp, little_endian)
    if colorid == 100:
        return np.array([serread(file, bpp, little_endian),
                         serread(file, bpp, little_endian),
                         serread(file, bpp, little_endian)])
    if colorid == 101:
        return np.array([serread(file, bpp, little_endian),
                         serread(file, bpp, little_endian),
                         serread(file, bpp, little_endian)])


def read_to_npy(file, bpp, little_endian, shape):
    """Read block of SER file"""
    num = 1
    for _, dim in enumerate(shape):
        num *= dim
    num_b = bpp * num
    block = np.array(list(file.read(num_b)), dtype=np.uint32)
    block = block.reshape((num, bpp))
    for i in range(bpp):
        if little_endian:
            block[:, i] *= 2**(8*i)
        else:
            block[:, i] *= 2**(8*(bpp-i))
    block = np.sum(block, axis=1)
    block = block.reshape(shape)
    return block


def readser(project: vstarstack.cfg.Project, fname: str):
    """Read SER file"""
    with open(fname, "rb") as file:
        fileid = file.read(14)
        if fileid != b'LUCAM-RECORDER':
            print("Invalid header, skipping")
            return
        _luid = serread4(file)
        colorid = serread4(file)
        le16bit = serread4(file)
        width = serread4(file)
        height = serread4(file)
        depth = serread4(file)
        bpp = (int)(math.ceil(depth / 8))
        frames = serread4(file)
        observer = file.read(40).decode('utf8')
        instrume = file.read(40).decode('utf8')
        telescope = file.read(40).decode('utf8')
        datetime = serread(file, 8, True)
        datetime_utc = serread(file, 8, True)

        if colorid == 0:
            shape = (height, width, 1)
            channels = ["Y"]
        elif colorid == 100:
            shape = (height, width, 3)
            channels = ["R", "G", "B"]
        elif colorid == 101:
            shape = (height, width, 3)
            channels = ["B", "G", "R"]
        else:
            print(f"Unsupported colorid = {colorid}")
            return

        tags = {
            "depth": depth,
            "observer": observer,
            "instrument": instrume,
            "telescope": telescope,
            "dateTime": datetime,
            "dateTimeUTC": datetime_utc,
        }

        params = {
            "w": width,
            "h": height,
            "projection": "perspective",
            "perspective_F": project.scope.F,
            "perspective_kh": project.camera.kh,
            "perspective_kw": project.camera.kw,
            "format": project.camera.format,
        }

        for frame_id in range(frames):
            print(f"\tprocessing frame {frame_id}")
            frame = read_to_npy(file, bpp, le16bit, shape)
            dataframe = vstarstack.data.DataFrame(params, tags)
            exptime = 1
            weight = np.ones(frame.data.shape)*exptime
            index = 0
            for index, channel in enumerate(channels):
                dataframe.add_channel(frame[:, :, index], channel)
                dataframe.add_channel(weight, "weight-"+channel)
                dataframe.add_channel_link(
                    channel, "weight-"+channel, "weight")
            yield frame_id, dataframe


def process_file(project: vstarstack.cfg.Project, argv: list):
    """Read single SER file"""
    fname = argv[0]
    output = argv[1]
    name = argv[2]

    for frame_id, dataframe in readser(project, fname):
        framename = os.path.join(output, f"{name}_{frame_id:06}.zip")
        dataframe.store(framename)


def process_path(project: vstarstack.cfg.Project, argv: list):
    """Read all SER files in directory"""
    input_dir = argv[0]
    output = argv[1]

    files = vstarstack.common.listfiles(input_dir, ".ser")
    for name, fname in files:
        print(name)
        process_file(project, (fname, output, name))


def process(project: vstarstack.cfg.Project, argv: list):
    """Process path with SER file(s)"""
    if len(argv) > 0:
        input_path = argv[0]
        output_path = argv[1]
        if os.path.isdir(input_path):
            process_path(project, (input_path, output_path))
        else:
            name = os.path.splitext(os.path.basename(input_path))[0]
            process_file(project, (input_path, output_path, name))
    else:
        process_path(project, [project.config["paths"]
                     ["original"], project.config["paths"]["npy-orig"]])


commands = {
    "*": (process, "read SER to npy", "(input.ser output/ | [original/ npy/])"),
}


def run(project: vstarstack.cfg.Project, argv: list):
    """Run SER file reading"""
    vstarstack.usage.run(project, argv, "readimage ser",
                         commands, autohelp=False)
