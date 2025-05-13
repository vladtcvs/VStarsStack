#
# Copyright (c) 2024-2025 Vladislav Tsendrovskii
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

import logging

import vstarstack.library.data
import vstarstack.library.photometry.fwhm
import vstarstack.library.stars.detect
import vstarstack.tool.common
import numpy as np
import os

logger = logging.getLogger(__name__)

def _measure_single_fwhm(project, argv):
    fname = argv[0]
    x = int(argv[1])
    y = int(argv[2])
    if len(argv) > 3:
        r = int(argv[3])
    else:
        r = 4
    df = vstarstack.library.data.DataFrame.load(fname)
    channels, mean = vstarstack.library.photometry.fwhm.find_fwhm_df(df, x, y, r)
    print("Mean: %f" % mean)
    for channel in channels:
        print("\t%s: %f" % (channel, channels[channel]))

def _measure_mean_fwhm_file(fname : str):
    df = vstarstack.library.data.DataFrame.load(fname)
    vstarstack.library.stars.detect.configure_detector(thresh_coeff=1.2)
    chvs = {}
    for channel in df.get_channels():
        layer, opts = df.get_channel(channel)
        if not opts["brightness"]:
            continue
        stars = vstarstack.library.stars.detect.detect_stars(layer)
        values = []
        for star in stars:
            x = int(star["x"]+0.5)
            y = int(star["y"]+0.5)
            r = int(star["radius"])
            fwhm = vstarstack.library.photometry.fwhm.find_fwhm(layer, x, y, r)
            if fwhm is not None:
                values.append(fwhm)
        fwhm = np.median(values)
        chvs[channel] = fwhm
    print("Mean: %f" % np.median(list(chvs.values())))
    for channel in chvs:
        print("\t%s: %f" % (channel, chvs[channel]))

def _measure_mean_fwhm_dir(dirname : str):
    files = vstarstack.tool.common.listfiles(dirname, ".zip")
    for name, filename in files:
        print(name)
        _measure_mean_fwhm_file(filename)

def _measure_mean_fwhm(project, argv):
    path = argv[0]
    if os.path.isdir(path):
        _measure_mean_fwhm_dir(path)
    else:
        _measure_mean_fwhm_file(path)

commands = {
    "star" : (_measure_single_fwhm,
              "calculate FWHM of specific star",
              "image.zip x y"),
    "mean" : (_measure_mean_fwhm,
              "calculate mean FWHM of stars",
              "(image.zip | path/)"),
}
