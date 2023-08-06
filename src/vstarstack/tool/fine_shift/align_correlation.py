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

from vstarstack.library.fine_shift.fine_shift import Aligner
import vstarstack.tool.usage
import vstarstack.tool.cfg
import vstarstack.tool.configuration
import vstarstack.library.data
import vstarstack.library.common

import vstarstack.tool.common

ncpu = vstarstack.tool.cfg.nthreads

def create_aligner(project: vstarstack.tool.cfg.Project, W: int, H: int):
    """Create aligner for the project"""
    num_steps = project.config.fine_shift.Nsteps
    dh = project.config.fine_shift.dh
    gridW = project.config.fine_shift.gridW
    gridH = project.config.fine_shift.gridH
    spk = project.config.fine_shift.stretchPenaltyCoefficient
    min_points = project.config.fine_shift.points_min_len

    aligner = Aligner(W, H, gridW, gridH, spk, num_steps, min_points, dh)
    return aligner

def align_file(project : vstarstack.tool.cfg.Project,
               name1 : str,
               name2 : str,
               input_image1_f : str,
               input_image2_f : str,
               desc_f : str):
    """Apply alignment to each file"""
    print(f"{name1} -> {name2}")
    if not os.path.exists(input_image1_f):
        return
    if not os.path.exists(input_image2_f):
        return

    df1 = vstarstack.library.data.DataFrame.load(input_image1_f)
    df2 = vstarstack.library.data.DataFrame.load(input_image2_f)
    w = df1.params["w"]
    h = df1.params["h"]
    aligner = create_aligner(project, w, h)

    light1, mask1 = vstarstack.library.common.df_to_light(df1)
    light2, mask2 = vstarstack.library.common.df_to_light(df2)

    # find alignment
    desc = aligner.process_alignment_by_correlation(light1, mask1, light2, mask2)
    print(f"{name1} - align to {name2} found")
    vstarstack.tool.common.check_dir_exists(desc_f)
    with open(desc_f, "w", encoding='utf8') as f:
        json.dump(desc, f, ensure_ascii=False, indent=2)

def _align_file_wrapper(arg):
    align_file(*arg)

def align(project: vstarstack.tool.cfg.Project, argv: list):
    if len(argv) >= 2:
        npys = argv[0]
        aligns = argv[1]
    else:
        npys = project.config.paths.npy_fixed
        aligns = project.config.fine_shift.aligns

    files = vstarstack.tool.common.listfiles(npys, ".zip")
    name0, input_image0_f = files[0]

    with mp.Pool(ncpu) as pool:
        args = [(project,
                 name,
                 name0,
                 input_image_f,
                 input_image0_f,
                 os.path.join(aligns, name + ".json"))
                 for name, input_image_f in files]
        for _ in pool.imap_unordered(_align_file_wrapper, args):
            pass
