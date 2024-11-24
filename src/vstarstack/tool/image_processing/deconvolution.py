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

def deconvolution(df : vstarstack.library.data.DataFrame, psf_df : vstarstack.library.data.DataFrame, argv : list, method : str):
    for channel in df.get_channels():
        image, opts = df.get_channel(channel)
        if not opts["brightness"]:
            continue
        psf,_ = psf_df.get_channel(channel)
        if psf is None:
            continue
        norm = np.amax(image)
        image = image / norm
        if method == "RL":
            num_steps = int(argv[0])
            deconvolved = skimage.restoration.richardson_lucy(image, psf, num_iter=num_steps)
        elif method == "Wiener":
            balance = float(argv[0])
            deconvolved = skimage.restoration.wiener(image, psf, balance)
        else:
            raise Exception(f"Unknown method {method}")
        deconvolved[np.where(np.isnan(deconvolved))] = 0
        deconvolved = deconvolved * norm
        df.replace_channel(deconvolved, channel, **opts)
    return df

def _process_file(input : str, psf : vstarstack.library.data.DataFrame, output : str, argv : list, method : str):
    df = vstarstack.library.data.DataFrame.load(input)
    deconvolution(df, psf, argv, method)
    vstarstack.tool.common.check_dir_exists(output)
    df.store(output)

def _process_single_file(input : str, psf_fname : str, output : str, argv : list, method : str):
    psf = vstarstack.library.data.DataFrame.load(psf_fname)
    _process_file(input, psf, output, argv, method)

def _process_dir(inputs : str, psf_fname : str, outputs : str, argv : list, method : str):
    files = vstarstack.tool.common.listfiles(inputs, ".zip")
    psf = vstarstack.library.data.DataFrame.load(psf_fname)
    with mp.Pool(vstarstack.tool.cfg.nthreads) as pool:
        pool.starmap(_process_file, [(fname, psf,
                                      os.path.join(outputs, name + ".zip"),
                                      argv,
                                      method) for name, fname in files])

def _process(project : vstarstack.tool.cfg.Project, argv : list[str], method : str):
    """Deconvolution"""
    inputs = argv[0]
    psf_fname = argv[1]
    outputs = argv[2]
    argv = argv[3:]

    if os.path.isdir(inputs):
        _process_dir(inputs, psf_fname, outputs, argv, method)
    else:
        _process_single_file(inputs, psf_fname, outputs, argv, method)

commands = {
    "rl": (lambda project, argv : _process(project, argv, "RL"),  "Richardson-Lucy deconvolution", "inputs/ psf.zip outputs/ <num_steps>"),
    "wiener": (lambda project, argv : _process(project, argv, "Wiener"),  "Wiener deconvolution", "inputs/ psf.zip outputs/ <balance>"),
}
