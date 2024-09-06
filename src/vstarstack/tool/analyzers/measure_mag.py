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
import csv
import numpy as np
import cv2


from vstarstack.tool.cfg import Project
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

def _measure_pixels(project : Project, argv : list[str], method : str):
    options = {}
    manual_params = False

    if method == "summ":
        if len(argv) >= 5:
            manual_params = True

    if manual_params:
        path = argv[0]
        output = argv[1]
        x = int(argv[2])
        y = int(argv[3])
        if method == "summ":
            options["radius"] = int(argv[4])
    else:
        path = project.config.paths.aligned
        output_dir = project.config.paths.photometry
        x = int(argv[0])
        y = int(argv[1])
        if method == "summ":
            options["radius"] = int(argv[2])
        output = os.path.join(output_dir, f"photometry_{x}_{y}.csv")

    vstarstack.tool.common.check_dir_exists(output)
    results = {}
    channels = set()
    timestamps = {}
    if os.path.isdir(path):
        files = vstarstack.tool.common.listfiles(path, ".zip")
    else:
        files = [(os.path.splitext(os.path.basename(path))[0], path)]
    for name, fname in files:
        df = DataFrame.load(fname)
        if 'UTC' in df.params:
            ts = df.params['UTC']
        else:
            ts = '-'
        timestamps[name] = ts
        if method == "summ":
            results[name] = summ_pixels_df(df, x, y, **options)

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
    "summ": (lambda project, argv: _measure_pixels(project, argv, "summ"),
             "calculate sum of pixels of star",
             "path/ output.csv x y radius"),
}
