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

import vstarstack.library.data
import vstarstack.library.debayer.bayer
import vstarstack.library.debayer.yuv422

import vstarstack.tool.common
import vstarstack.tool.cfg
import vstarstack.tool.usage
import vstarstack.library.common

from vstarstack.library.debayer.bayer import DebayerMethod

nthreads = vstarstack.tool.cfg.nthreads

def _process_file(name, default_format, fname, output):
    print(name)
    dataframe = vstarstack.library.data.DataFrame.load(fname)
    if "format" in dataframe.params:
        mode = dataframe.params["format"]
    else:
        mode = default_format

    print(f"Mode = {mode}")
    if mode[:6] == "bayer_":
        mask_desc = mode[6:]
        mask = vstarstack.library.debayer.bayer.generate_mask(mask_desc)
        method = vstarstack.tool.cfg.get_param("method", str, "SUBSAMPLE")
        if method == "SUBSAMPLE":
            method = DebayerMethod.SUBSAMPLE
        elif method == "CFA":
            method = DebayerMethod.CFA
        elif method == "INTERPOLATE":
            method = DebayerMethod.INTERPOLATE

        dataframe = vstarstack.library.debayer.bayer.debayer_dataframe(dataframe, mask, "raw", method)
    elif mode == "yuv_422":
        dataframe = vstarstack.library.debayer.yuv422.yuv_422_dataframe(dataframe, "raw")
    else:
        return
    vstarstack.tool.common.check_dir_exists(output)
    dataframe.store(output)

def _process_file_wrapper(arg):
    _process_file(*arg)

def _process_path(default_format, input_path, output_path):
    files = vstarstack.tool.common.listfiles(input_path, ".zip")
    with mp.Pool(nthreads) as pool:
        args = [(name,
                 default_format,
                 fname,
                 os.path.join(output_path, name + ".zip"))
                 for name, fname in files]
        for _ in pool.imap_unordered(_process_file_wrapper, args):
            pass

def run(project: vstarstack.tool.cfg.Project, argv: list):
    default_format = project.config.telescope.camera.format
    if len(argv) > 0:
        input_path = argv[0]
        output_path = argv[1]
        if os.path.isdir(input_path):
            _process_path(default_format, input_path, output_path)
        else:
            _process_file(input_path, default_format, input_path, output_path)
    else:
        _process_path(default_format,
                      project.config.paths.light.npy,
                      project.config.paths.light.npy)
