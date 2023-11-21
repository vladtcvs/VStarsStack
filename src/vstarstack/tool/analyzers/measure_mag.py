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

import numpy as np
import cv2
import csv

from vstarstack.library.data import DataFrame
import vstarstack.tool.common

def summ_pixels(image : np.ndarray, x : int, y : int, radius : int) -> float:
    """Find sum of pixels in circle at x,y"""
    area = image[y-radius:y+radius+1, x-radius:x+radius+1]
    mask = np.zeros((2*radius+1, 2*radius+1))
    cv2.circle(mask, (radius, radius), radius, 1, -1)
    area = area * mask
    return np.sum(area)

def summ_pixels_df(image : DataFrame, x : int, y : int, radius : int) -> dict:
    """Find sum of pixels in circle at x,y"""
    vals = {}
    for cn in image.get_channels():
        channel, opts = image.get_channel(cn)
        if opts["brightness"]:
            vals[cn] = summ_pixels(channel, x, y, radius)
    return vals

def _measure_pixels(_project, argv):
    path = argv[0]
    x = int(argv[1])
    y = int(argv[2])
    r = int(argv[3])
    output = argv[4]
    results = {}
    channels = set()
    timestamps = {}
    for name, fname in vstarstack.tool.common.listfiles(path, ".zip"):
        df = DataFrame.load(fname)
        if 'DATE-OBS' in df.tags:
            ts = df.tags['DATE-OBS']
        else:
            ts = '-'
        timestamps[name] = ts
        results[name] = summ_pixels_df(df, x, y, r)
        for cn in results[name]:
            channels.add(cn)
    channels = list(channels)
    with open(output, "w", encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'timestamp'] + channels)
        for name, sums in results.items():
            values = []
            timestamp = timestamps[name]
            for cn in channels:
                if cn not in sums:
                    values.append('-')
                else:
                    values.append(sums[cn])
            writer.writerow([name, timestamp] + values)

commands = {
    "summ": (_measure_pixels, "calculate sum of pixels of star", "path/ x y radius output.csv"),
}
