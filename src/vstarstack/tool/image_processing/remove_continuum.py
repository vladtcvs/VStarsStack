"""Remove sky from the image"""
#
# Copyright (c) 2025 Vladislav Tsendrovskii
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
import logging

import vstarstack.tool.usage
import vstarstack.tool.cfg
import vstarstack.library.common
import vstarstack.library.data
import vstarstack.library.image_process.remove_continuum
import vstarstack.tool.common

logger = logging.getLogger(__name__)

def remove_continuum(name, infname, outfname, narrow_channel, wide_channel, coeff):
    """Remove continuum from file"""
    logger.info(f"Processing {name}: {narrow_channel} - {wide_channel}")

    img = vstarstack.library.data.DataFrame.load(infname)
    narrow, opts = img.get_channel(narrow_channel)
    wide,_ = img.get_channel(wide_channel)
    no_continuum = vstarstack.library.image_process.remove_continuum.remove_continuum(narrow, wide, coeff)
    img.replace_channel(no_continuum, narrow_channel, **opts)
    vstarstack.tool.common.check_dir_exists(outfname)
    img.store(outfname)

def process_file(argv):
    """Remove continuum from single file"""
    infname = argv[0]
    narrow_channel = argv[1]
    wide_channel = argv[2]
    outfname = argv[3]
    coeff = float(argv[4]) if len(argv) >= 5 else None
    name = os.path.splitext(os.path.basename(infname))[0]
    remove_continuum(name, infname, outfname, narrow_channel, wide_channel, coeff)

def process_dir(argv):
    """Remove continuum from all files in directory"""
    inpath = argv[0]
    narrow_channel = argv[1]
    wide_channel = argv[2]
    outpath = argv[3]
    coeff = float(argv[4]) if len(argv) >= 5 else None
    files = vstarstack.tool.common.listfiles(inpath, ".zip")
    with mp.Pool(vstarstack.tool.cfg.nthreads) as pool:
        pool.starmap(remove_continuum, [(name, fname, os.path.join(
            outpath, name + ".zip"), narrow_channel, wide_channel, coeff) for name, fname in files])

def process(project: vstarstack.tool.cfg.Project, argv: list):
    """Process file(s) in path"""
    if os.path.isdir(argv[0]):
        process_dir(argv)
    else:
        process_file(argv)
