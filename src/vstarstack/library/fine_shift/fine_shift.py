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
import scipy

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

    def apply_alignment(self,
                        dataframe : vstarstack.library.data.DataFrame,
                        align : dict,
                        subpixels : int):
        """Apply alignment descriptor to file"""
        wave = ImageWave.from_data(align)
        for channel in dataframe.get_channels():
            image, opts = dataframe.get_channel(channel)
            if opts["encoded"]:
                continue
            image = image.astype('double')
            fixed = wave.apply_shift(image, subpixels)
            fixed[np.where(np.isnan(fixed))] = 0
            dataframe.replace_channel(fixed, channel)
        return dataframe

class ClusterAlignerBuilder:

    def __init__(self, W, H, gridW, gridH, spk, num_steps, min_points, dh):
        self.W = W
        self.H = H
        self.gridW = gridW
        self.gridH = gridH
        self.spk = spk
        self.num_steps = num_steps
        self.min_points = min_points
        self.dh = dh

    def find_alignment(self, name : str, clusters : list) -> dict:
        """Find alignment of image `name` using clusters"""
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
        wave.approximate_by_targets(targets, points, self.num_steps, self.dh)
        descriptor = wave.data()
        return descriptor

    def find_all_alignments(self, clusters : list) -> dict:
        """Build alignment descriptor using clusters"""
        names = []
        for cluster in clusters:
            names += cluster.keys()
        names = set(names)
        descs = {}
        for name in names:
            desc = self.process_alignment_by_clusters(name, clusters)
            if desc is not None:
                descs[name] = desc
        return descs

class CorrelationAlignedBuilder:

    def __init__(self, radius : int, maximal_shift : float, subpixels : int):
        self.r = radius
        self.shift = maximal_shift
        self.subp = subpixels

    def find_alignment(self,
                       image : np.ndarray,
                       pre_align : dict | None,
                       image_ref : np.ndarray,
                       pre_align_ref : dict | None,
                       smooth : int | None):
        """Build alignment descriptor of image using correlations"""
        if image.shape != image_ref.shape:
            return None
        if pre_align is not None:
            pre_wave = ImageWave.from_data(pre_align)
        else:
            pre_wave = None

        if pre_align_ref is not None:
            pre_wave_ref = ImageWave.from_data(pre_align_ref)
        else:
            pre_wave_ref = None
        wave = ImageWave.find_shift_array(image, pre_wave,
                                          image_ref, pre_wave_ref,
                                          self.r, self.shift, self.subp)
        align = wave.data()
        print(f"smooth = {smooth}")
        if smooth is not None:
            data = align["data"]
            Nw = align["Nw"]
            Nh = align["Nh"]
            data = np.array(data, dtype='double')
            data = data.reshape((Nh, Nw, 2))
            data = scipy.ndimage.gaussian_filter(data, sigma=smooth, axes=(0,1))
            data = list(data.reshape((Nh*Nw*2,)))
            align["data"] = data
        return align
