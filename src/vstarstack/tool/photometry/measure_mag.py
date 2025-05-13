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

from vstarstack.tool.cfg import Project
from vstarstack.library.data import DataFrame
import vstarstack.tool.common
from vstarstack.library.photometry.magnitude import star_magnitude_summ_df

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
            results[name] = star_magnitude_summ_df(df, x, y, options["radius"])

        for cn in results[name]:
            channels.add(cn)
    channels = list(channels)
    with open(output, "w", encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'timestamp', 'x', 'y', 'npixels'] + channels)
        for name, sums in results.items():
            values = []
            timestamp = timestamps[name]
            npixels = None
            for cn in channels:
                if cn not in sums:
                    values.append('-')
                else:
                    values.append(sums[cn][0])
                    npixels = sums[cn][1]
                    _ = sums[cn][2]
            writer.writerow([name, timestamp, x, y, npixels] + values)

commands = {
    "summ": (lambda project, argv: _measure_pixels(project, argv, "summ"),
             "calculate sum of pixels of star",
             "(path/ | image.zip) output.csv x y radius"),
}
