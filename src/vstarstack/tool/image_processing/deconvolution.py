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

def deconvolution(df : vstarstack.library.data.DataFrame, psf_df : vstarstack.library.data.DataFrame, strength : int):
    for channel in df.get_channels():
        image, opts = df.get_channel(channel)
        if not opts["brightness"]:
            continue
        psf,_ = psf_df.get_channel(channel)
        if psf is None:
            continue
        norm = np.amax(image)
        deconvolved_RL = skimage.restoration.richardson_lucy(image/norm, psf, num_iter=strength)*norm
        deconvolved_RL[np.where(np.isnan(deconvolved_RL))] = 0
        df.replace_channel(deconvolved_RL, channel, **opts)
    return df

def _process_file(input : str, psf : vstarstack.library.data.DataFrame, output : str, strength : int):
    df = vstarstack.library.data.DataFrame.load(input)
    deconvolution(df, psf, strength)
    df.store(output)

def _process_single_file(input : str, psf_fname : str, output : str, strength : int):
    psf = vstarstack.library.data.DataFrame.load(psf_fname)
    _process_file(input, psf, output, strength)

def _process_dir(inputs : str, psf_fname : str, outputs : str, strength : int):
    files = vstarstack.tool.common.listfiles(inputs, ".zip")
    psf = vstarstack.library.data.DataFrame.load(psf_fname)
    with mp.Pool(vstarstack.tool.cfg.nthreads) as pool:
        pool.starmap(_process_file, [(fname, psf,
                                      os.path.join(outputs, name + ".zip"),
                                      strength) for name, fname in files])

def run(project : vstarstack.tool.cfg.Project, argv : list[str]):
    """Deconvolution"""
    inputs = argv[0]
    psf_fname = argv[1]
    outputs = argv[2]
    strength = int(argv[3])

    if os.path.isdir(inputs):
        _process_dir(inputs, psf_fname, outputs, strength)
    else:
        _process_single_file(inputs, psf_fname, outputs, strength)
