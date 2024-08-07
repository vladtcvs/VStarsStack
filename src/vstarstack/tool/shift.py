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
import typing
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

import vstarstack.tool.common

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

def _make_shift_same_size(name : str, filename : str, shift : Movement | None, outfname : str):
    if shift is None:
        print(f"Skip {name}")
        return
    print(f"Processing {name}")
    dataframe = vstarstack.library.data.DataFrame.load(filename)
    print(f"Loaded {name}")
    result = vstarstack.library.movement.move_image.move_dataframe(dataframe, shift)
    print(f"Transformed {name}")
    vstarstack.tool.common.check_dir_exists(outfname)
    result.store(outfname)
    print(f"Saved {name}")

def _make_shift_extended_size(name : str, filename : str,
                              shift : Movement | None,
                              outfname : str,
                              output_shape : tuple,
                              output_proj : str,
                              output_proj_desc : dict):
    if shift is None:
        print(f"Skip {name}")
        return
    print(f"Processing {name}")
    dataframe = vstarstack.library.data.DataFrame.load(filename)
    print(f"Loaded {name}")

    input_proj = vstarstack.library.projection.tools.get_projection(dataframe)
    output_proj = vstarstack.library.projection.tools.build_projection(output_proj, output_proj_desc, output_shape)
    result = vstarstack.library.movement.move_image.move_dataframe(dataframe, shift,
                                                                   input_proj=input_proj,
                                                                   output_proj=output_proj,
                                                                   output_shape=output_shape)
    print(f"Transformed {name}")
    vstarstack.tool.common.check_dir_exists(outfname)
    result.store(outfname)
    print(f"Saved {name}")

def apply_shift(project: vstarstack.tool.cfg.Project, argv: list[str]):
    """Apply shifts to images"""
    if len(argv) > 0:
        npy_dir = argv[0]
        shifts_fname = argv[1]
        shifted_dir = argv[2]
    else:
        npy_dir = project.config.paths.npy_fixed
        shifts_fname = project.config.paths.absolute_shifts
        shifted_dir = project.config.paths.aligned

    with open(shifts_fname, encoding='utf8') as file:
        serialized = json.load(file)
    shifts = {}
    for name,shift_ser in serialized.items():
        shifts[name] = Movement.deserialize(shift_ser)

    images = vstarstack.tool.common.listfiles(npy_dir, ".zip")

    args = [(name, filename, shifts[name] if name in shifts else None, os.path.join(shifted_dir, name + ".zip")) 
            for name, filename in images]
    with mp.Pool(ncpu) as pool:
        pool.starmap(_make_shift_same_size, args)

def _find_extended(images : list, shifts : dict):
    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf

    out_proj_type = None
    out_proj_desc = {}

    for name, filename in images:
        
        df = vstarstack.library.data.DataFrame.load(filename)
        input_proj_type, input_proj_desc = vstarstack.library.projection.tools.extract_description(df)
        w = df.get_parameter("w")
        h = df.get_parameter("h")
        input_proj = vstarstack.library.projection.tools.build_projection(input_proj_type, input_proj_desc, (h, w))

        out_proj_type = input_proj_type
        out_proj_desc = input_proj_desc

        points = np.array([(0,0),(w,0),(0,h),(w,h)])
        
        shift = shifts[name]
        shifted_points = shift.forward(points.astype('double'), input_proj, input_proj)
        min_x = min(min_x, min(shifted_points[:0]))
        max_x = max(max_x, max(shifted_points[:0]))
        min_y = min(min_y, min(shifted_points[:1]))
        max_y = max(max_y, max(shifted_points[:1]))

    W = math.ceil(max_x-min_x)
    H = math.ceil(max_y-min_y)
    out_shape = (H, W)

    return out_shape, out_proj_type, out_proj_desc

def apply_shift_extended(project: vstarstack.tool.cfg.Project, argv: list[str]):
    """Apply shifts to images"""
    if len(argv) > 0:
        npy_dir = argv[0]
        shifts_fname = argv[1]
        shifted_dir = argv[2]
    else:
        npy_dir = project.config.paths.npy_fixed
        shifts_fname = project.config.paths.absolute_shifts
        shifted_dir = project.config.paths.aligned

    with open(shifts_fname, encoding='utf8') as file:
        serialized = json.load(file)
    shifts = {}
    for name,shift_ser in serialized.items():
        shifts[name] = Movement.deserialize(shift_ser)

    images = vstarstack.tool.common.listfiles(npy_dir, ".zip")

    output_shape, output_proj, output_proj_desc = _find_extended(images, shifts)

    args = [(name,
             filename,
             shifts[name] if name in shifts else None,
             os.path.join(shifted_dir, name + ".zip"),
             output_shape,
             output_proj,
             output_proj_desc,
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
    "apply-shift-extended": (apply_shift_extended,
                    "Apply selected shifts and save to output with extended size",
                    "shift.json npy/ shifted/"),
}

def run(project: vstarstack.tool.cfg.Project, argv: list[str]):
    """Run image shifting"""
    vstarstack.tool.usage.run(project, argv, "shift", commands, autohelp=True)
