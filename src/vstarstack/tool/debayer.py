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

import vstarstack.library.data
import vstarstack.library.debayer.bayer
import vstarstack.library.debayer.yuv422

import vstarstack.tool.cfg
import vstarstack.tool.usage
import vstarstack.library.common

def _process_file(default_format, fname, output):
    dataframe = vstarstack.library.data.DataFrame.load(fname)
    if "format" in dataframe.params:
        mode = dataframe.params["format"]
    else:
        mode = default_format

    print(f"Mode = {mode}")
    if mode[:5] == "bayer":
        mask_desc = mode[5:9]
        mask = vstarstack.library.debayer.bayer.generate_mask(mask_desc)
        dataframe = vstarstack.library.debayer.bayer.debayer_dataframe(dataframe, mask, "raw")
    elif mode == "yuv422":
        dataframe = vstarstack.library.debayer.yuv422.yuv_422_dataframe(dataframe, "raw")
    else:
        return
    dataframe.store(output)

def _process_path(default_format, input_path, output_path):
    files = vstarstack.library.common.listfiles(input_path, ".zip")
    for name, fname in files:
        print(name)
        _process_file(default_format, fname, os.path.join(output_path, name + ".zip"))

def _process(project: vstarstack.tool.cfg.Project, argv: list):
    default_format = project.camera.format
    if len(argv) > 0:
        input_path = argv[0]
        output_path = argv[1]
        if os.path.isdir(input_path):
            _process_path(default_format, input_path, output_path)
        else:
            _process_file(default_format, input_path, output_path)
    else:
        _process_path(default_format,
                      project.config["paths"]["npy-orig"],
                      project.config["paths"]["npy-fixed"])

def run(project: vstarstack.tool.cfg.Project, argv: list):
    _process(project, argv)