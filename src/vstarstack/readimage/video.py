"""Read video source file"""
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
import cv2

import vstarstack.cfg
import vstarstack.common
import vstarstack.usage
import vstarstack.data

import vstarstack.readimage.tags


def read_video(project: vstarstack.cfg.Project, fname: str):
    """Read frames from video file"""
    vidcap = cv2.VideoCapture(fname)
    frame_id = 0

    # vidcap.set(cv2.CAP_PROP_FORMAT, -1)
    while True:
        success, frame = vidcap.read()
        if not success:
            break

        tags = {
            "depth": 8,
        }

        params = {
            "w": frame.shape[1],
            "h": frame.shape[0],
            "projection": "perspective",
            "perspective_F": project.scope.F,
            "perspective_kh": project.camera.kh,
            "perspective_kw": project.camera.kw,
            "format": project.camera.format,
        }

        print(f"\tprocessing frame {frame_id}")

        exptime = 1
        weight = np.ones((frame.shape[0], frame.shape[1]))*exptime

        dataframe = vstarstack.data.DataFrame(params, tags)
        dataframe.add_channel(frame[:, :, 0], "R")
        dataframe.add_channel(frame[:, :, 1], "G")
        dataframe.add_channel(frame[:, :, 2], "B")
        dataframe.add_channel(weight, "weight")
        dataframe.add_channel_link("R", "weight", "weight")
        dataframe.add_channel_link("G", "weight", "weight")
        dataframe.add_channel_link("B", "weight", "weight")
        yield frame_id, dataframe
        frame_id += 1


def process_file(project: vstarstack.cfg.Project, argv: list):
    """Process single file"""
    fname = argv[0]
    output = argv[1]
    name = argv[2]

    for frame_id, dataframe in read_video(project, fname):
        framename = os.path.join(output, f"{name}_{frame_id:05}")
        dataframe.store(framename)


def process_path(project: vstarstack.cfg.Project, argv: list):
    """Process all videeo files in directory"""
    input_dir = argv[0]
    output_dir = argv[1]

    files = vstarstack.common.listfiles(input_dir)
    for name, fname in files:
        print(name)
        process_file(project, (fname, output_dir, name))


def process(project: vstarstack.cfg.Project, argv: list):
    """Process video file(s) in path"""
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
    "*": (process, "read Video to npy", "(input.video output/ | [original/ npy/])"),
}


def run(project: vstarstack.cfg.Project, argv: list):
    """Read video file to npy"""
    vstarstack.usage.run(project, argv, "readimage video",
                         commands, autohelp=False)
