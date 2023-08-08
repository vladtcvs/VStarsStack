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
import cv2
import numpy as np
import multiprocessing as mp
from skimage.restoration import richardson_lucy

import vstarstack.tool.common
import vstarstack.tool.cfg
import vstarstack.library.data

def deconvolution(df : vstarstack.library.data.DataFrame, radius : int):
    psf = np.zeros((2*radius+1, 2*radius+1))
    cv2.circle(psf, (radius, radius), radius, 1, -1)
    psf = psf / sum(psf)
    for channel in df.get_channels():
        image, opts = df.get_channel(channel)
        if not opts["brightness"]:
            continue
        norm = np.amax(image)
        deconvolved_RL = richardson_lucy(image/norm, psf, num_iter=30)*norm
        deconvolved_RL[np.where(np.isnan(deconvolved_RL))] = 0
        df.replace_channel(deconvolved_RL, channel)
    return df

def _process_file(input : str, output : str, radius : int):
    df = vstarstack.library.data.DataFrame.load(input)
    deconvolution(df, radius)
    df.store(output)

def _process_dir(inputs : str, outputs : str, radius : int):
    files = vstarstack.tool.common.listfiles(inputs, ".zip")
    with mp.Pool(vstarstack.tool.cfg.nthreads) as pool:
        pool.starmap(_process_file, [(fname,
                                      os.path.join(outputs, name + ".zip"),
                                      radius) for name, fname in files])

def run(project : vstarstack.tool.cfg.Project, argv : list[str]):
    """Deconvolution"""
    if len(argv) >= 3:
        inputs = argv[0]
        radius = int(argv[1])
        outputs = argv[2]
    else:
        inputs = project.config.paths.npy_fixed
        outputs = project.config.paths.npy_fixed
        radius = int(argv[0])

    if os.path.isdir(inputs):
        _process_dir(inputs, outputs, radius)
    else:
        _process_file(inputs, outputs, radius)
