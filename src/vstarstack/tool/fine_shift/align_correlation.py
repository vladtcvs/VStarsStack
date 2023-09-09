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

from vstarstack.library.fine_shift.fine_shift import CorrelationAlignedBuilder
import vstarstack.tool.usage
import vstarstack.tool.cfg
import vstarstack.tool.configuration
import vstarstack.library.data
import vstarstack.library.common

import vstarstack.tool.common

ncpu = vstarstack.tool.cfg.nthreads

def create_aligner(project: vstarstack.tool.cfg.Project, W: int, H: int):
    """Create aligner for the project"""
    aligner_factory = CorrelationAlignedBuilder(7, 3, 2)
    return aligner_factory

def align_file(project : vstarstack.tool.cfg.Project,
               name : str,
               name_ref : str,
               input_image_f : str,
               input_image_ref_f : str,
               align_f : str,
               pre_align_f : str | None,
               pre_align_ref_f : str | None):
    """Apply alignment to each file"""
    print(f"{name} -> {name_ref}")
    if not os.path.exists(input_image_f):
        return
    if not os.path.exists(input_image_ref_f):
        return

    df = vstarstack.library.data.DataFrame.load(input_image_f)
    df_ref = vstarstack.library.data.DataFrame.load(input_image_ref_f)
    w = df.params["w"]
    h = df.params["h"]
    aligner_factory = create_aligner(project, w, h)

    light, _ = vstarstack.library.common.df_to_light(df)
    light_ref, _ = vstarstack.library.common.df_to_light(df_ref)

    if pre_align_f is None or not os.path.isfile(pre_align_f):
        pre_align = None
    else:
        with open(pre_align_f, encoding='utf8') as f:
            pre_align = json.load(f)

    if pre_align_ref_f is None or not os.path.isfile(pre_align_ref_f):
        pre_align_ref = None
    else:
        with open(pre_align_ref_f, encoding='utf8') as f:
            pre_align_ref = json.load(f)

    # find alignment
    alignment = aligner_factory.find_alignment(light, pre_align,
                                               light_ref, pre_align_ref, 3)
    print(f"{name} - align to {name_ref} found")
    vstarstack.tool.common.check_dir_exists(align_f)
    with open(align_f, "w", encoding='utf8') as f:
        json.dump(alignment, f, ensure_ascii=False, indent=2)

def _align_file_wrapper(arg):
    align_file(*arg)

def align(project: vstarstack.tool.cfg.Project, argv: list):
    if len(argv) >= 2:
        npys = argv[0]
        aligns = argv[1]
        if len(argv) >= 3:
            pre_aligns = argv[2]
        else:
            pre_aligns = None
    else:
        npys = project.config.paths.npy_fixed
        aligns = project.config.fine_shift.aligns
        pre_aligns = project.config.fine_shift.aligns

    files = vstarstack.tool.common.listfiles(npys, ".zip")
    name0, input_image0_f = files[0]

    with mp.Pool(ncpu) as pool:
        args = [(project,
                 name,
                 name0,
                 input_image_f,
                 input_image0_f,
                 os.path.join(aligns, name + ".json"),
                 os.path.join(pre_aligns, name + ".json") if pre_aligns is not None else None,
                 os.path.join(pre_aligns, name0 + ".json") if pre_aligns is not None else None)
                 for name, input_image_f in files]
        for _ in pool.imap_unordered(_align_file_wrapper, args):
            pass
