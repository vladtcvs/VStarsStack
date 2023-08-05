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

import numpy as np

from vstarstack.library.fine_shift.image_wave import ImageWave
import vstarstack.library.data
import vstarstack.library.common

def _cluster_average(cluster):
    xs = [cluster[name]["x"] for name in cluster]
    ys = [cluster[name]["y"] for name in cluster]
    av = {
        "x" : sum(xs) / len(xs),
        "y" : sum(ys) / len(ys),
    }
    return av

class Aligner:
    """Alignment calculator"""
    def __init__(self, W, H, gridW, gridH, spk, num_steps, min_points, dh):
        self.W = W
        self.H = H
        self.gridW = gridW
        self.gridH = gridH
        self.spk = spk
        self.num_steps = num_steps
        self.min_points = min_points
        self.dh = dh

    def process_alignment(self,
                          name : str,
                          clusters : list):
        """Find alignment of image `name`"""
        points = []
        targets = []
        for cluster in clusters:
            if name not in cluster:
                continue
            average = _cluster_average(cluster)

            # we need reverse transformation
            x = average["x"]
            y = average["y"]
            points.append((x, y))
            x = cluster[name]["x"]
            y = cluster[name]["y"]
            targets.append((x, y))

        print(f"\tusing {len(points)} points")
        if len(points) < self.min_points:
            print("\tskip - too low points")
            return None
        wave = ImageWave(self.W, self.H, self.gridW, self.gridH, self.spk)
        wave.approximate(targets, points, self.num_steps, self.dh)
        descriptor = wave.data()
        return descriptor

    def find_all_alignments(self,
                            clusters : list):
        """Build all alignment descriptors"""
        names = []
        for cluster in clusters:
            names += cluster.keys()
        names = set(names)
        descs = {}
        for name in names:
            desc = self.process_alignment(name, clusters)
            if desc is not None:
                descs[name] = desc
        return descs

    def apply_alignment(self,
                        dataframe : vstarstack.library.data.DataFrame,
                        descriptor):
        """Apply alignment descriptor to file"""
        wave = ImageWave.from_data(descriptor)
        for channel in dataframe.get_channels():
            image, opts = dataframe.get_channel(channel)
            if opts["encoded"]:
                continue
            fixed = np.zeros(image.shape)
            for y in range(fixed.shape[0]):
                for x in range(fixed.shape[1]):
                    ox, oy = wave.interpolate(x, y)
                    fixed[y, x] = vstarstack.library.common.getpixel(image, oy, ox)[1]

            dataframe.replace_channel(fixed, channel)
        return dataframe
