"""Image shifting"""
#
# Copyright (c) 2022-2024 Vladislav Tsendrovskii
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
from typing import Tuple
import numpy as np

import vstarstack.library.loaders
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
                              output_proj_type : ProjectionType,
                              output_proj_desc : dict,
                              interpolate : bool | None):
    if shift is None:
        logger.warning(f"Skip {name}")
        return
    logger.info(f"Processing {name}")
    dataframe = vstarstack.library.data.DataFrame.load(filename)
    logger.info(f"Loaded {name}")

    input_proj = vstarstack.library.projection.tools.get_projection(dataframe)
    output_proj = vstarstack.library.projection.tools.build_projection(output_proj_type, output_proj_desc, output_shape)
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

def _find_shifted_corners(input_proj, input_w, input_h, shift = None):
    points = np.array([(0,0),(input_w,0),(0,input_h),(input_w,input_h)])
    lonlats = np.array([input_proj.project(x,y) for x,y in points])
    if shift is not None:
        lonlats = shift.apply_lonlat(lonlats.astype('double'))

    min_lon = min(lonlats[:,0])
    max_lon = max(lonlats[:,0])
    min_lat = min(lonlats[:,1])
    max_lat = max(lonlats[:,1])
    return (min_lon, min_lat), (max_lon, max_lat)

def _find_extended_dimensions(images : list, shifts : dict) -> Tuple[float, float, list]:
    max_w = 0
    max_h = 0
    projections = []

    for name, filename in images:
        logger.info(f"Processing {name}")
        if name not in shifts:
            logger.warning(f"No shift for {name}, skipping")
            continue
        shift = shifts[name]
        df = vstarstack.library.data.DataFrame.load(filename)
        projection_type, projection_desc = vstarstack.library.projection.tools.extract_description(df)
        if projection_type != ProjectionType.Perspective:
            raise ValueError(f"Invalid projection {projection_type}")
        input_w = df.get_parameter("w")
        input_h = df.get_parameter("h")
        input_proj = vstarstack.library.projection.tools.build_projection(projection_type, projection_desc, (input_h, input_w))
        (min_lon, min_lat), (max_lon, max_lat) = _find_shifted_corners(input_proj, input_w, input_h, shift)
        max_w = max([max_w, abs(min_lon), abs(max_lon)])
        max_h = max([max_h, abs(min_lat), abs(max_lat)])
        projections.append((projection_type, projection_desc))

    return max_w, max_h, projections

def _auto_build_best_projection(projections : list) -> Tuple[ProjectionType, dict]:
    if len(projections) == 0:
        return ProjectionType.NoneProjection, {}
    min_kwF = math.inf
    min_khF = math.inf
    max_F = 0
    for projection_type, projection_desc in projections:
        if projection_type != ProjectionType.Perspective:
            raise NotImplementedError("Not implemented")
        F = projection_desc["F"]
        kw = projection_desc["kw"]
        kh = projection_desc["kh"]
        max_F = max(F, max_F)
        min_kwF = min(min_kwF, kw/F)
        min_khF = min(min_khF, kh/F)
    kw = max_F * min_kwF
    kh = max_F * min_khF
    return ProjectionType.Perspective, {"F" : max_F, "kw" : kw, "kh" : kh}

def _calculate_shape(dim_lon, dim_lat, output_proj_type, output_proj_desc) -> Tuple[int, int]:
    proj = vstarstack.library.projection.tools.build_projection(output_proj_type, output_proj_desc, (0,0))
    x1, y1 = proj.reverse(-dim_lon, -dim_lat)
    x2, y2 = proj.reverse(dim_lon, dim_lat)
    w = abs(x2-x1)
    h = abs(y2-y1)
    return (h, w)

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
    output_proj_name = project.config.output.projection_type
    output_proj_desc = project.config.output.projection_desc    

    with open(shifts_fname, encoding='utf8') as file:
        serialized = json.load(file)
    shifts = {}
    for name,shift_ser in serialized.items():
        shifts[name] = Movement.deserialize(shift_ser)

    images = vstarstack.tool.common.listfiles(npy_dir, ".zip")
    dim_lon, dim_lat, projections = _find_extended_dimensions(images, shifts)
    logger.info(f"Resulting dimensions: {dim_lon*180/math.pi:.2f}x{dim_lat*180/math.pi:.2f}")

    if output_proj_name != "COPY":
        if output_proj_name == "perspective":
            output_proj_type = ProjectionType.Perspective
            output_proj_desc = json.loads(output_proj_desc)
        else:
            raise NotImplementedError("Not implemented")
    else:
        output_proj_type, output_proj_desc = _auto_build_best_projection(projections)

    output_shape = _calculate_shape(dim_lon, dim_lat, output_proj_type, output_proj_desc)

    args = [(name,
             filename,
             shifts[name] if name in shifts else None,
             os.path.join(shifted_dir, name + ".zip"),
             output_shape,
             output_proj_type,
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
