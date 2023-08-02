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
import json
import multiprocessing as mp

import vstarstack.tool.cfg
import vstarstack.tool.usage

from vstarstack.library.movement.sphere import Movement

import vstarstack.library.data
import vstarstack.library.common
import vstarstack.library.movement.select_shift
import vstarstack.library.movement.move_image

import vstarstack.tool.common

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

def _make_shift(name : str, filename : str, shift : Movement, outfname : str):
    print(f"Processing {name}")
    dataframe = vstarstack.library.data.DataFrame.load(filename)
    result = vstarstack.library.movement.move_image.move_dataframe(dataframe, shift)
    vstarstack.tool.common.check_dir_exists(outfname)
    result.store(outfname)

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
    args = [(name, filename, shifts[name], os.path.join(shifted_dir, name + ".zip")) 
            for name, filename in images]
    with mp.Pool(vstarstack.tool.cfg.nthreads) as pool:
        pool.starmap(_make_shift, args)

commands = {
    "select-shift": (select_shift,
                     "Select base image and shift",
                     "shifts.json shift.json"),
    "apply-shift": (apply_shift,
                    "Apply selected shifts",
                    "shift.json npy/ shifted/"),
}

def run(project: vstarstack.tool.cfg.Project, argv: list[str]):
    """Run image shifting"""
    vstarstack.tool.usage.run(project, argv, "shift", commands, autohelp=True)
