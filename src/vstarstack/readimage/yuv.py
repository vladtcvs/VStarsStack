"""Read YUV video to npy frames"""
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

import vstarstack.cfg
import vstarstack.common
import vstarstack.usage
import vstarstack.data
import vstarstack.readimage.tags


def readyuv(project: vstarstack.cfg.Project, fname: str, width: int, height: int):
    """Read YUV file"""
    frame_len = int(width*height*2)
    shape = (int(height*2), width)

    with open(fname, "rb") as files:
        tags = {
            "depth": 8,
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

        frame_id = 0
        while True:
            frame = files.read(frame_len)
            if not frame:
                break
            yuv = np.frombuffer(frame, dtype=np.uint8)
            try:
                yuv = yuv.reshape(shape)
                print(f"\tprocessing frame {frame_id}")

                dataframe = vstarstack.data.DataFrame(params, tags)
                exptime = 1
                weight = np.ones(frame.data.shape)*exptime

                dataframe.add_channel(yuv, "raw", encoded=True)
                dataframe.add_channel(weight, "weight")
                dataframe.add_channel_link("raw", "weight", "weight")
                yield frame_id, dataframe
                frame_id += 1
            except Exception as _:
                break


def process_file(project: vstarstack.cfg.Project, argv: list):
    """Process single YUV file"""
    fname = argv[0]
    output = argv[1]
    name = argv[2]

    width = project.camera.w
    height = project.camera.h

    for frame_id, dataframe in readyuv(project, fname, width, height):
        framename = os.path.join(output, f"{name}_{frame_id:05}.zip")
        dataframe.store(framename)


def process_path(project: vstarstack.cfg.Project, argv: list):
    """Process all files in directory"""
    input_dir = argv[0]
    output_dir = argv[1]

    files = vstarstack.common.listfiles(input_dir, ".yuv")
    for name, fname in files:
        print(name)
        process_file(project, (fname, output_dir, name))


def process(project: vstarstack.cfg.Project, argv: list):
    """Process path with YUV file(s)"""
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
    "*": (process, "read SER to npy", "(input.yuv output/ | [original/ npy/])"),
}


def run(project: vstarstack.cfg.Project, argv: list):
    """Run reading YUV"""
    vstarstack.usage.run(project, argv, "readimage yuv",
                         commands, autohelp=False)
