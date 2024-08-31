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

def deconvolution(df : vstarstack.library.data.DataFrame, strength : int):
    psf = np.zeros((3, 3))

    psf[1, 1] = 1

    psf[0, 1] = 1/strength
    psf[2, 1] = 1/strength
    psf[1, 0] = 1/strength
    psf[1, 2] = 1/strength

    psf[0, 0] = 1/strength/1.41
    psf[0, 2] = 1/strength/1.41
    psf[2, 0] = 1/strength/1.41
    psf[2, 0] = 1/strength/1.41

    psf = psf / np.sum(psf)

    for channel in df.get_channels():
        image, opts = df.get_channel(channel)
        if not opts["brightness"]:
            continue
        norm = np.amax(image)
        deconvolved_RL = skimage.restoration.richardson_lucy(image/norm, psf, num_iter=100)*norm
        deconvolved_RL[np.where(np.isnan(deconvolved_RL))] = 0
        df.replace_channel(deconvolved_RL, channel)
    return df

def _process_file(input : str, output : str, strength : int):
    df = vstarstack.library.data.DataFrame.load(input)
    deconvolution(df, strength)
    df.store(output)

def _process_dir(inputs : str, outputs : str, strength : int):
    files = vstarstack.tool.common.listfiles(inputs, ".zip")
    with mp.Pool(vstarstack.tool.cfg.nthreads) as pool:
        pool.starmap(_process_file, [(fname,
                                      os.path.join(outputs, name + ".zip"),
                                      strength) for name, fname in files])

def run(project : vstarstack.tool.cfg.Project, argv : list[str]):
    """Deconvolution"""
    if len(argv) >= 3:
        inputs = argv[0]
        strength = int(argv[1])
        outputs = argv[2]
    else:
        inputs = project.config.paths.npy_fixed
        outputs = project.config.paths.npy_fixed
        strength = int(argv[0])

    if os.path.isdir(inputs):
        _process_dir(inputs, outputs, strength)
    else:
        _process_file(inputs, outputs, strength)
