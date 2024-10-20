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
import multiprocessing as mp

import vstarstack.tool.common
import vstarstack.tool.cfg
import vstarstack.tool.usage
import vstarstack.library.common
import vstarstack.library.data
import vstarstack.library.calibration.dark
import vstarstack.library.calibration.flat

# applying flats
def _process_file_flatten(input_fname : str,
                          flat : vstarstack.library.data.DataFrame,
                          output_fname : str):
    print(f"Processing {input_fname}")
    dataframe = vstarstack.library.data.DataFrame.load(input_fname)
    result = vstarstack.library.calibration.flat.flatten(dataframe, flat)
    vstarstack.tool.common.check_dir_exists(output_fname)
    result.store(output_fname)

def _process_dir_flatten(input_path : str,
                         flat : vstarstack.library.data.DataFrame,
                         output_path : str):
    files = vstarstack.tool.common.listfiles(input_path, ".zip")
    files = vstarstack.tool.common.listfiles(input_path, ".zip")
    with mp.Pool(vstarstack.tool.cfg.nthreads) as pool:
        args = [(filename, flat, os.path.join(output_path, name + ".zip")) for name, filename in files]
        pool.starmap(_process_file_flatten, args)

def _process_flatten(_project : vstarstack.tool.cfg.Project,
                     argv : list[str]):
    input_path = argv[0]
    flat_fname = argv[1]
    output_path = argv[2]
    flat = vstarstack.library.data.DataFrame.load(flat_fname)
    if os.path.isdir(input_path):
        _process_dir_flatten(input_path, flat, output_path)
    else:
        _process_file_flatten(input_path, flat, output_path)

# applying darks
def _process_file_remove_dark(input_fname : str,
                              dark : vstarstack.library.data.DataFrame,
                              output_fname : str):
    print(f"Processing {input_fname}")
    dataframe = vstarstack.library.data.DataFrame.load(input_fname)
    result = vstarstack.library.calibration.dark.remove_dark(dataframe, dark)
    if result is None:
        print(f"Can not remove dark from {input_fname}")
    vstarstack.tool.common.check_dir_exists(output_fname)
    result.store(output_fname)

def _process_dir_remove_dark(input_path : str,
                      dark : vstarstack.library.data.DataFrame,
                      output_path : str):
    files = vstarstack.tool.common.listfiles(input_path, ".zip")
    with mp.Pool(vstarstack.tool.cfg.nthreads) as pool:
        args = [(filename, dark, os.path.join(output_path, name + ".zip")) for name, filename in files]
        pool.starmap(_process_file_remove_dark, args)

def _process_remove_dark(_project : vstarstack.tool.cfg.Project,
                         argv : list[str]):
    input_path = argv[0]
    dark_fname = argv[1]
    output_path = argv[2]
    dark = vstarstack.library.data.DataFrame.load(dark_fname)
    if os.path.isdir(input_path):
        _process_dir_remove_dark(input_path, dark, output_path)
    else:
        _process_file_remove_dark(input_path, dark, output_path)

# building darks
def _process_build_dark(project : vstarstack.tool.cfg.Project,
                        argv : list[str]):
    if len(argv) >= 2:
        input_path = argv[0]
        dark_fname = argv[1]
    else:
        input_path = project.config.paths.dark.npy
        dark_fname = project.config.paths.dark.result
    files = vstarstack.tool.common.listfiles(input_path, ".zip")
    darks = [item[1] for item in files]
    src = vstarstack.library.common.FilesImageSource(darks)
    dark = vstarstack.library.calibration.dark.prepare_darks(src)
    dark.store(dark_fname)

# building flats
def _process_build_flat_simple(project : vstarstack.tool.cfg.Project,
                               argv : list[str]):
    if len(argv) >= 2:
        input_path = argv[0]
        flat_fname = argv[1]
    else:
        input_path = project.config.paths.flat.npy
        flat_fname = project.config.paths.flat.result
    smooth = vstarstack.tool.cfg.get_param("smooth", int, 31)
    files = vstarstack.tool.common.listfiles(input_path, ".zip")
    flats = [item[1] for item in files]
    src = vstarstack.library.common.FilesImageSource(flats)
    flat = vstarstack.library.calibration.flat.prepare_flat_simple(src, smooth)
    flat.store(flat_fname)

def _process_build_flat_sky(project : vstarstack.tool.cfg.Project,
                              argv : list[str]):
    if len(argv) >= 2:
        input_path = argv[0]
        flat_fname = argv[1]
    else:
        input_path = project.config.paths.flat.npy
        flat_fname = project.config.paths.flat.result

    smooth = vstarstack.tool.cfg.get_param("smooth", int, 31)
    if smooth % 2 == 0:
        smooth += 1
    files = vstarstack.tool.common.listfiles(input_path, ".zip")
    flats = [item[1] for item in files]
    src = vstarstack.library.common.FilesImageSource(flats)
    flat = vstarstack.library.calibration.flat.prepare_flat_sky(src, smooth)
    vstarstack.tool.common.check_dir_exists(flat_fname)
    flat.store(flat_fname)

def _process_approximate_flat(project : vstarstack.tool.cfg.Project,
                              argv : list[str]):
    if len(argv) >= 2:
        input_fname = argv[0]
        flat_fname = argv[1]
    else:
        input_fname = project.config.paths.flat.result
        flat_fname = project.config.paths.flat.result

    df = vstarstack.library.data.DataFrame.load(input_fname)
    df = vstarstack.library.calibration.flat.approximate_flat_image(df)
    df.store(flat_fname)

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
                           "flats/ flat.zip"),
    "approximate-flat" : (_process_approximate_flat,
                          "Approximate flat with polynomic function",
                          "original_flat.zip result_flat.zip")
}
