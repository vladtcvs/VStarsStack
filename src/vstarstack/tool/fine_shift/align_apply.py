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
import json
import multiprocessing as mp

from vstarstack.library.fine_movement.aligner import Aligner
import vstarstack.tool.usage
import vstarstack.tool.cfg
import vstarstack.tool.configuration
import vstarstack.library.data
import vstarstack.library.common

import vstarstack.tool.common

ncpu = vstarstack.tool.cfg.nthreads

def align_file(project : vstarstack.tool.cfg.Project,
               name : str,
               input_image_f : str,
               desc_f : str,
               output_image_f : str,
               subpixels : int):
    """Apply alignment to each file"""
    print(f"{name}: {input_image_f} : {desc_f} -> {output_image_f} [{subpixels}]")

    if not os.path.exists(input_image_f):
        return
    if not os.path.exists(desc_f):
        return

    with open(desc_f, encoding='utf8') as f:
        aligner = Aligner.deserialize(json.load(f))

    df = vstarstack.library.data.DataFrame.load(input_image_f)

    # apply alignment to file
    df = aligner.apply_alignment(df, subpixels)
    print(f"{name} - aligned")

    vstarstack.tool.common.check_dir_exists(output_image_f)
    df.store(output_image_f)

def _align_file_wrapper(arg):
    align_file(*arg)

def apply(project: vstarstack.tool.cfg.Project, argv: list):
    if len(argv) >= 3:
        npys = argv[0]
        aligns = argv[1]
        outputs = argv[2]
        if len(argv) >= 4:
            subpixels = int(argv[3])
        else:
            subpixels = 1
    else:
        npys = project.config.paths.npy_fixed
        aligns = project.config.fine_shift.aligns
        subpixels = project.config.fine_shift.subpixels
        outputs = project.config.paths.aligned

    files = vstarstack.tool.common.listfiles(npys, ".zip")
    with mp.Pool(ncpu) as pool:
        args = [(project,
                 name,
                 input_image_f,
                 os.path.join(aligns, name + ".json"),
                 os.path.join(outputs, name + ".zip"),
                 subpixels)
                 for name, input_image_f in files]
        for _ in pool.imap_unordered(_align_file_wrapper, args):
            pass
