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

import vstarstack.tool.cfg
import vstarstack.tool.usage
import vstarstack.library.common
import vstarstack.library.data
import vstarstack.library.calibration.dark
import vstarstack.library.calibration.flat

def _process_file_flatten(input_fname : str,
                          flat_fname : str,
                          output_fname : str):
    dataframe = vstarstack.library.data.DataFrame.load(input_fname)
    flat = vstarstack.library.data.DataFrame.load(flat_fname)
    result = vstarstack.library.calibration.flat.flatten(dataframe, flat)
    result.store(output_fname)

def _process_file_remove_dark(input_fname : str,
                              dark_fname : str,
                              output_fname : str):
    dataframe = vstarstack.library.data.DataFrame.load(input_fname)
    dark = vstarstack.library.data.DataFrame.load(dark_fname)
    result = vstarstack.library.calibration.dark.remove_dark(dataframe, dark)
    result.store(output_fname)

def _process_dir_flatten(input_path : str,
                         flat_fname : str,
                         output_path : str):
    files = vstarstack.library.common.listfiles(input_path, ".zip")
    for name, filename in files:
        print(f"Processing {name}")
        output_fname = os.path.join(output_path, name + ".zip")
        _process_file_flatten(filename, flat_fname, output_fname)

def _process_dir_remove_dark(input_path : str,
                      dark_fname : str,
                      output_path : str):
    files = vstarstack.library.common.listfiles(input_path, ".zip")
    for name, filename in files:
        print(f"Processing {name}")
        output_fname = os.path.join(output_path, name + ".zip")
        _process_file_remove_dark(filename, dark_fname, output_fname)

def _process_flatten(_project : vstarstack.tool.cfg.Project,
                     argv : list[str]):
    input_path = argv[0]
    flat_fname = argv[1]
    output_path = argv[2]
    if os.path.isdir(input_path):
        _process_dir_flatten(input_path, flat_fname, output_path)
    else:
        _process_file_flatten(input_path, flat_fname, output_path)

def _process_remove_dark(_project : vstarstack.tool.cfg.Project,
                         argv : list[str]):
    input_path = argv[0]
    dark_fname = argv[1]
    output_path = argv[2]
    if os.path.isdir(input_path):
        _process_dir_remove_dark(input_path, dark_fname, output_path)
    else:
        _process_file_remove_dark(input_path, dark_fname, output_path)

def _process_build_dark(_project : vstarstack.tool.cfg.Project,
                        argv : list[str]):
    input_path = argv[0]
    dark_fname = argv[1]
    files = vstarstack.library.common.listfiles(input_path, ".zip")
    darks = [item[1] for item in files]
    src = vstarstack.library.common.FilesImageSource(darks)
    dark = vstarstack.library.calibration.dark.prepare_darks(src)
    dark.store(dark_fname)

def _process_build_flat_simple(_project : vstarstack.tool.cfg.Project,
                               argv : list[str]):
    input_path = argv[0]
    flat_fname = argv[1]
    smooth = vstarstack.tool.cfg.get_param("smooth", int, 31)
    files = vstarstack.library.common.listfiles(input_path, ".zip")
    flats = [item[1] for item in files]
    src = vstarstack.library.common.FilesImageSource(flats)
    flat = vstarstack.library.calibration.flat.prepare_flat_simple(src, smooth)
    flat.store(flat_fname)

def _process_build_flat_sky(_project : vstarstack.tool.cfg.Project,
                              argv : list[str]):
    input_path = argv[0]
    flat_fname = argv[1]
    smooth = vstarstack.tool.cfg.get_param("smooth", int, 31)
    if smooth % 2 == 0:
        smooth += 1
    files = vstarstack.library.common.listfiles(input_path, ".zip")
    flats = [item[1] for item in files]
    src = vstarstack.library.common.FilesImageSource(flats)
    flat = vstarstack.library.calibration.flat.prepare_flat_sky(src, smooth)
    flat.store(flat_fname)

commands = {
    "flatten": (_process_flatten,
                "Flatten image",
                "inputs/ flat.zip outputs/"),
    "remove-dark": (_process_remove_dark,
                   "Substract dark from image",
                   "inputs/ dark.zip outputs/"),
    "build-dark" : (_process_build_dark,
                    "Create dark image",
                    "darks/ dark.zip"),
    "build-flat-simple" : (_process_build_flat_simple,
                           "Create flat image - just sum of flats",
                           "flats/ flat.zip"),
    "build-flat-sky" : (_process_build_flat_sky,
                           "Create flat image - use sky images",
                           "flats/ flat.zip")
}
