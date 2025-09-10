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
import numpy as np
import multiprocessing as mp
import logging

from vstarstack.library.fine_movement.aligner import CorrelationAlignedBuilder
from vstarstack.library.fine_movement.aligner import Aligner
import vstarstack.library.image_process
import vstarstack.library.image_process.togray
import vstarstack.tool.cfg
import vstarstack.library.data

import vstarstack.tool.common

ncpu = vstarstack.tool.cfg.nthreads
logger = logging.getLogger(__name__)

def create_aligner(project: vstarstack.tool.cfg.Project,
                   image_w: int,
                   image_h: int,
                   radius: int,
                   max_shift: int,
                   pixels: int,
                   subpixels: int):
    """Create aligner for the project"""
    aligner_factory = CorrelationAlignedBuilder(image_w, image_h, pixels,
                                                radius, max_shift, subpixels)
    return aligner_factory

def align_file(project : vstarstack.tool.cfg.Project,
               name : str,
               name_ref : str,
               input_image_f : str,
               light_ref : np.ndarray,
               mask_ref : np.ndarray,
               align_f : str,
               pre_align_f : str | None,
               pre_align_ref : Aligner | None):
    """Apply alignment to each file"""
    logger.info(f"{name} -> {name_ref}")
    if not os.path.exists(input_image_f):
        return

    df = vstarstack.library.data.DataFrame.load(input_image_f)

    max_shift = project.config.fine_shift.max_shift
    pixels = project.config.fine_shift.correlation_grid
    area_radius = project.config.fine_shift.area_radius
    logger.info(f"Maximal shift: {max_shift}")
    image_w = light_ref.shape[1]
    image_h = light_ref.shape[0]
    aligner_factory = create_aligner(project,
                                     image_w,
                                     image_h,
                                     area_radius,
                                     max_shift,
                                     pixels,
                                     2)

    light, weight = vstarstack.library.image_process.togray.df_to_gray(df)
    light = np.clip(light, 0, None).astype(np.float32)
    mask = (weight > 0).astype(np.uint8)

    if pre_align_f is None or not os.path.isfile(pre_align_f):
        pre_align = None
    else:
        with open(pre_align_f, encoding='utf8') as f:
            pre_align = Aligner.deserialize(json.load(f))

    # find alignment
    alignment = aligner_factory.find_alignment(light, mask, light_ref, mask_ref,
                                               pre_align, pre_align_ref,
                                               3)
    logger.warning(f"{name} - align to {name_ref} found")
    vstarstack.tool.common.check_dir_exists(align_f)
    with open(align_f, "w", encoding='utf8') as f:
        json.dump(alignment.serialize(), f, ensure_ascii=False, indent=2)

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
        npys = project.config.paths.light.npy
        aligns = project.config.fine_shift.aligns
        pre_aligns = None

    files = vstarstack.tool.common.listfiles(npys, ".zip")
    name0, input_image0_f = files[0]
    logger.info("Loading image 0")
    input_image0 = vstarstack.library.data.DataFrame.load(input_image0_f)
    if input_image0 is None:
        raise Exception("No REFERENCE!")
    light0,weight0 = vstarstack.library.image_process.togray.df_to_gray(input_image0)
    light0 = np.clip(light0, 0, None).astype(np.float32)
    mask0 = (weight0 > 0).astype(np.uint8)

    if vstarstack.tool.cfg.DEBUG:
        import matplotlib.pyplot as plt
        plt.imshow(light0)
        plt.show()

    if pre_aligns is not None:
        fname = os.path.join(pre_aligns, name0 + ".json")
        with open(fname) as f:
            pre_align0 = Aligner.deserialize(json.load(f))
    else:
        pre_align0 = None

    with mp.Pool(ncpu) as pool:
        args = [(project,
                 name,
                 name0,
                 input_image_f,
                 light0,
                 mask0,
                 os.path.join(aligns, name + ".json"),
                 os.path.join(pre_aligns, name + ".json") if pre_aligns is not None else None,
                 pre_align0)
                 for name, input_image_f in files]
        for _ in pool.imap_unordered(_align_file_wrapper, args):
            pass
