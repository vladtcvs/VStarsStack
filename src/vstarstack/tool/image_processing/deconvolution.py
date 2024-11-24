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
import multiprocessing as mp
import skimage.restoration

import vstarstack.tool.common
import vstarstack.tool.cfg
import vstarstack.library.data

def deconvolution(df : vstarstack.library.data.DataFrame, psf_df : vstarstack.library.data.DataFrame, strength : int, method : str):
    for channel in df.get_channels():
        image, opts = df.get_channel(channel)
        if not opts["brightness"]:
            continue
        psf,_ = psf_df.get_channel(channel)
        if psf is None:
            continue
        if method == "RL":
            norm = np.amax(image)
            deconvolved = skimage.restoration.richardson_lucy(image/norm, psf, num_iter=strength)*norm
            deconvolved[np.where(np.isnan(deconvolved))] = 0
        else:
            raise Exception(f"Unknown method {method}")
        df.replace_channel(deconvolved, channel, **opts)
    return df

def _process_file(input : str, psf : vstarstack.library.data.DataFrame, output : str, strength : int, method : str):
    df = vstarstack.library.data.DataFrame.load(input)
    deconvolution(df, psf, strength, method)
    vstarstack.tool.common.check_dir_exists(output)
    df.store(output)

def _process_single_file(input : str, psf_fname : str, output : str, strength : int, method : str):
    psf = vstarstack.library.data.DataFrame.load(psf_fname)
    _process_file(input, psf, output, strength, method)

def _process_dir(inputs : str, psf_fname : str, outputs : str, strength : int, method : str):
    files = vstarstack.tool.common.listfiles(inputs, ".zip")
    psf = vstarstack.library.data.DataFrame.load(psf_fname)
    with mp.Pool(vstarstack.tool.cfg.nthreads) as pool:
        pool.starmap(_process_file, [(fname, psf,
                                      os.path.join(outputs, name + ".zip"),
                                      strength,
                                      method) for name, fname in files])

def _process(project : vstarstack.tool.cfg.Project, argv : list[str], method : str):
    """Deconvolution"""
    inputs = argv[0]
    psf_fname = argv[1]
    outputs = argv[2]
    strength = int(argv[3])

    if os.path.isdir(inputs):
        _process_dir(inputs, psf_fname, outputs, strength, method)
    else:
        _process_single_file(inputs, psf_fname, outputs, strength, method)

commands = {
    "rl": (lambda project, argv : _process(project, argv, "RL"),  "Richardson-Lucy deconvolution", "inputs/ psf.zip outputs/ <num_steps>"),
}