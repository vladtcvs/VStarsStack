"""Image shifting"""
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
import math
import json
import logging
import multiprocessing as mp
import numpy as np

import vstarstack.tool.cfg
import vstarstack.tool.usage

from vstarstack.library.movement.sphere import Movement

import vstarstack.library.data
import vstarstack.library.projection.tools
import vstarstack.library.common
import vstarstack.library.movement.select_shift
import vstarstack.library.movement.move_image
from vstarstack.library.projection import ProjectionType

import vstarstack.tool.common

logger = logging.getLogger(__name__)

ncpu = vstarstack.tool.cfg.nthreads
#ncpu = 1

def select_shift(project: vstarstack.tool.cfg.Project, argv: list[str]):
    """Select optimal shift source"""
    if len(argv) >= 2:
        all_shifts = argv[0]
        selected_shift = argv[1]
    else:
        all_shifts = project.config.paths.relative_shifts
        selected_shift = project.config.paths.absolute_shifts

    with open(all_shifts, encoding='utf8') as f:
        serialized = json.load(f)
    shifts = {}
    for name1,shifts1 in serialized.items():
        shifts[name1] = {}
        for name2 in shifts1:
            shifts[name1][name2] = Movement.deserialize(serialized[name1][name2])
    basic_name = vstarstack.library.movement.select_shift.select_base_image(shifts)
    with open(selected_shift, "w", encoding='utf8') as f:
        json.dump(serialized[basic_name], f, ensure_ascii=False, indent=4)

def _make_shift_same_size(name : str, filename : str,
                          shift : Movement | None,
                          outfname : str,
                          interpolate : bool | None):
    if shift is None:
        logger.warning(f"Skip {name}")
        return
    logger.info(f"Processing {name}")
    dataframe = vstarstack.library.data.DataFrame.load(filename)
    logger.info(f"Loaded {name}")
    result = vstarstack.library.movement.move_image.move_dataframe(dataframe, shift, interpolate=interpolate)
    logger.info(f"Transformed {name}")
    logger.info(f"Saving {outfname}")
    vstarstack.tool.common.check_dir_exists(outfname)
    result.store(outfname)
    logger.info(f"Saved {name}")

def _make_shift_extended_size(name : str, filename : str,
                              shift : Movement | None,
                              outfname : str,
                              output_shape : tuple,
                              output_proj : str,
                              output_proj_desc : dict,
                              interpolate : bool | None):
    if shift is None:
        logger.warning(f"Skip {name}")
        return
    logger.info(f"Processing {name}")
    dataframe = vstarstack.library.data.DataFrame.load(filename)
    logger.info(f"Loaded {name}")

    input_proj = vstarstack.library.projection.tools.get_projection(dataframe)
    output_proj = vstarstack.library.projection.tools.build_projection(output_proj, output_proj_desc, output_shape)
    result = vstarstack.library.movement.move_image.move_dataframe(dataframe, shift,
                                                                   input_proj=input_proj,
                                                                   output_proj=output_proj,
                                                                   output_shape=output_shape,
                                                                   interpolate=interpolate)
    logger.info(f"Transformed {name}")
    vstarstack.tool.common.check_dir_exists(outfname)
    result.store(outfname)
    logger.info(f"Saved {name}")

def apply_shift(project: vstarstack.tool.cfg.Project, argv: list[str]):
    """Apply shifts to images"""
    if len(argv) > 0:
        npy_dir = argv[0]
        shifts_fname = argv[1]
        shifted_dir = argv[2]
        interpolate = project.config.shift.interpolate
    else:
        npy_dir = project.config.paths.light.npy
        shifts_fname = project.config.paths.absolute_shifts
        shifted_dir = project.config.paths.aligned
        interpolate = project.config.shift.interpolate

    with open(shifts_fname, encoding='utf8') as file:
        serialized = json.load(file)
    shifts = {}
    for name,shift_ser in serialized.items():
        shifts[name] = Movement.deserialize(shift_ser)

    images = vstarstack.tool.common.listfiles(npy_dir, ".zip")

    args = [(name, filename, shifts[name], os.path.join(shifted_dir, name + ".zip"), interpolate if name in shifts else None) 
            for name, filename in images]
    with mp.Pool(ncpu) as pool:
        pool.starmap(_make_shift_same_size, args)

def _find_shifted_corners(arg):
    filename, name, shifts = arg
    logger.info(f"Estimating size of {name}")
    df = vstarstack.library.data.DataFrame.load(filename)
    input_proj_type, input_proj_desc = vstarstack.library.projection.tools.extract_description(df)
    if input_proj_type != ProjectionType.Perspective:
        logger.error(f"Invalid projection type: {input_proj_type}")
        return (None, None, None, None, None, None, None)

    w = df.get_parameter("w")
    h = df.get_parameter("h")
    input_proj = vstarstack.library.projection.tools.build_projection(ProjectionType.Perspective, input_proj_desc, (h, w))

    out_proj_desc = input_proj_desc

    points = np.array([(0,0),(w,0),(0,h),(w,h)])
    if name not in shifts:
        logger.warning(f"Skip {name} which is not present in shifts")
        return (None, None, None, None, None, None, None)
    shift = shifts[name]
    shifted_points = shift.apply(points.astype('double'), input_proj, input_proj)

    min_x = min(shifted_points[:,0])
    max_x = max(shifted_points[:,0])
    min_y = min(shifted_points[:,1])
    max_y = max(shifted_points[:,1])
    return (min_x, min_y, max_x, max_y, w, h, out_proj_desc)

def _find_extended_perspective(images : list, shifts : dict):
    margin_left = 0
    margin_right = 0
    margin_top = 0
    margin_bottom = 0

    out_proj_desc = {}

    max_w = 0
    max_h = 0

    args = [(filename, name, shifts) for name, filename in images]

    with mp.Pool(vstarstack.tool.cfg.nthreads) as pool:
        corners = pool.map(_find_shifted_corners, args)
        for min_x, min_y, max_x, max_y, w, h, proj_desc in corners:
            if min_x is None:
                continue
            out_proj_desc = proj_desc
            if min_x < 0:
                margin_left = max(margin_left, -min_x)
            if min_y < 0:
                margin_top = max(margin_top, -min_y)
            if max_x > w:
                margin_right = max(margin_right, max_x-w)
            if max_y > h:
                margin_bottom = max(margin_bottom, max_y-h)
            if w > max_w:
                max_w = w
            if h > max_h:
                max_h = h

    margin_w = max(margin_left, margin_right)
    margin_h = max(margin_top, margin_bottom)

    W = max_w + 2*math.ceil(margin_w)
    H = max_h + 2*math.ceil(margin_h)

    out_shape = (H, W)

    return out_shape, ProjectionType.Perspective, out_proj_desc

def apply_shift_extended(project: vstarstack.tool.cfg.Project, argv: list[str]):
    """Apply shifts to images"""
    if len(argv) > 0:
        npy_dir = argv[0]
        shifts_fname = argv[1]
        shifted_dir = argv[2]
        interpolate = project.config.shift.interpolate
    else:
        npy_dir = project.config.paths.light.npy
        shifts_fname = project.config.paths.absolute_shifts
        shifted_dir = project.config.paths.aligned
        interpolate = project.config.shift.interpolate

    with open(shifts_fname, encoding='utf8') as file:
        serialized = json.load(file)
    shifts = {}
    for name,shift_ser in serialized.items():
        shifts[name] = Movement.deserialize(shift_ser)

    images = vstarstack.tool.common.listfiles(npy_dir, ".zip")

    output_shape, output_proj, output_proj_desc = _find_extended_perspective(images, shifts)

    logger.info(f"Resulting size: {output_shape[1]}x{output_shape[0]}")

    args = [(name,
             filename,
             shifts[name] if name in shifts else None,
             os.path.join(shifted_dir, name + ".zip"),
             output_shape,
             output_proj,
             output_proj_desc,
             interpolate,
            )
            for name, filename in images]
    with mp.Pool(ncpu) as pool:
        pool.starmap(_make_shift_extended_size, args)


commands = {
    "select-shift": (select_shift,
                     "Select base image and shift",
                     "shifts.json shift.json"),
    "apply-shift": (apply_shift,
                    "Apply selected shifts",
                    "shift.json npy/ shifted/"),
    "apply-extended-shift": (apply_shift_extended,
                    "Apply selected shifts and save to output with extended size (only perspective projection!)",
                    "shift.json npy/ shifted/"),
}

def run(project: vstarstack.tool.cfg.Project, argv: list[str]):
    """Run image shifting"""
    vstarstack.tool.usage.run(project, argv, "shift", commands)
