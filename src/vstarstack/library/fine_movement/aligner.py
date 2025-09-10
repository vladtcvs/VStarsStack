#
# Copyright (c) 2022-2024 Vladislav Tsendrovskii
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
import numpy as np
import scipy

from vstarstack.library.fine_movement.module import ImageGrid
from vstarstack.library.fine_movement.module import ImageDeform
from vstarstack.library.fine_movement.module import ImageDeformLC
from vstarstack.library.fine_movement.module import ImageDeformGC

from vstarstack.library.data import DataFrame

logger = logging.getLogger(__name__)

def _cluster_average(cluster):
    xs = [cluster[name]["x"] for name in cluster]
    ys = [cluster[name]["y"] for name in cluster]
    av = {
        "x" : sum(xs) / len(xs),
        "y" : sum(ys) / len(ys),
    }
    return av

class Aligner:
    """Alignment applier"""

    def __init__(self, image_w : int, image_h : int, shift_array : np.ndarray):
        self.image_w = image_w
        self.image_h = image_h
        self.grid_h = shift_array.shape[0]
        self.grid_w = shift_array.shape[1]
        self.deform = ImageDeform(self.image_w, self.image_h, self.grid_w, self.grid_h)
        self.deform.fill(shift_array=shift_array)

    def divergence(self, subpixels : int) -> np.ndarray:
        """Find divergence"""
        if type(subpixels) != int:
            raise Exception("Invalid argument type")
        return self.deform.divergence(subpixels)

    def apply_alignment(self, dataframe : DataFrame, subpixels : int) -> DataFrame:
        """Apply alignment descriptor to file"""
        for channel in dataframe.get_channels():
            image, opts = dataframe.get_channel(channel)
            if opts["encoded"]:
                continue
            w = image.shape[1]
            h = image.shape[0]
            grid = ImageGrid(w, h)
            grid.fill(image.astype(np.float32))
            fixed_grid = self.deform.apply_image(image=grid, subpixels=subpixels)
            fixed = fixed_grid.content()
            fixed[np.where(np.isnan(fixed))] = 0
            dataframe.replace_channel(fixed, channel)
        return dataframe

    def serialize(self) -> dict:
        """Serialize image deform"""
        nitems = self.grid_h * self.grid_w * 2
        shift_array = np.reshape(self.deform.content(), (nitems,)).astype(np.double)
        return {
            "grid_w" : self.grid_w,
            "grid_h" : self.grid_h,
            "image_w" : self.image_w,
            "image_h" : self.image_h,
            "array" : list(shift_array),
        }

    @staticmethod
    def deserialize(description : dict):
        """Deserialize image deform"""
        grid_h = description["grid_h"]
        grid_w = description["grid_w"]
        shift_array = np.reshape(description["array"], (grid_h, grid_w, 2))
        return Aligner(description["image_w"], description["image_h"], shift_array)

class ClusterAlignerBuilder:
    def __init__(self, image_w, image_h, grid_w, grid_h, spk, num_steps, min_points, dh):
        self.image_w = image_w
        self.image_h = image_h
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.spk = spk
        self.num_steps = num_steps
        self.min_points = min_points
        self.dh = dh
        self.correlator = ImageDeformGC(self.image_w, self.image_h,
                                        self.grid_w, self.grid_h,
                                        self.spk)

    def find_alignment(self, name : str, clusters : list) -> Aligner:
        """Find alignment of image `name` using clusters"""
        image_points_coordinates = []
        mean_points_coordinates = []

        for cluster in clusters:
            if name not in cluster:
                continue
            average = _cluster_average(cluster)

            # we need reverse transformation
            x = average["x"]
            y = average["y"]
            mean_points_coordinates.append((y, x))
            x = cluster[name]["x"]
            y = cluster[name]["y"]
            image_points_coordinates.append((y, x))

        mean_points_coordinates = np.array(mean_points_coordinates).astype("double")
        image_points_coordinates = np.array(image_points_coordinates).astype("double")

        logger.info(f"using {len(mean_points_coordinates)} points")
        if len(mean_points_coordinates) < self.min_points:
            logger.warning(f"skip - too low points: {len(mean_points_coordinates)}, need {self.min_points}")
            return None

        deform = self.correlator.find(points=mean_points_coordinates,
                                      expected_points=image_points_coordinates,
                                      dh=self.dh,
                                      Nsteps=self.num_steps)
        shift_array = deform.content()
        return Aligner(self.image_w, self.image_h, shift_array)

    def find_all_alignments(self, clusters : list) -> dict:
        """Build alignment descriptor using clusters"""
        names = []
        for cluster in clusters:
            names += cluster.keys()
        names = set(names)
        deforms = {}
        for name in names:
            deform = self.find_alignment(name, clusters)
            if deform is not None:
                deforms[name] = deform
        return deforms

class CorrelationAlignedBuilder:

    def __init__(self,
                 image_w : int,
                 image_h : int,
                 pixels : int,
                 radius : int,
                 maximal_shift : float,
                 subpixels : int):
        self.image_w = image_w
        self.image_h = image_h
        self.pixels = pixels
        self.correlator = ImageDeformLC(self.image_w, self.image_h, self.pixels)

        self.radius = radius
        self.max_shift = maximal_shift
        self.subpixels = subpixels

    def find_alignment(self, image : np.ndarray,
                       mask : np.ndarray,
                       image_ref : np.ndarray,
                       mask_ref : np.ndarray,
                       pre_align : Aligner | None,
                       pre_align_ref : Aligner | None,
                       smooth : int | None) -> Aligner:
        """Build alignment descriptor of image using correlations"""
        if image.shape != image_ref.shape:
            return None
        if mask.shape != image_ref.shape:
            return None
        if mask_ref.shape != image_ref.shape:
            return None
        w = image.shape[1]
        h = image.shape[0]

        grid = ImageGrid(w, h)
        grid.fill(image)
        #mask_grid = ImageGrid(w, h)
        #mask_grid.fill(mask)

        grid_ref = ImageGrid(w, h)
        grid_ref.fill(image_ref)
        #mask_grid_ref = ImageGrid(w, h)
        #mask_grid_ref.fill(mask_ref)

        deform_img = pre_align.deform if pre_align is not None else None
        deform_ref = pre_align_ref.deform if pre_align_ref is not None else None
        deform = self.correlator.find(grid, deform_img,
                                      grid_ref, deform_ref,
                                      self.radius,
                                      self.max_shift,
                                      self.subpixels)
        logger.info(f"smooth = {smooth}")
        if smooth is not None:
            data = deform.content()
            data = scipy.ndimage.gaussian_filter(data, sigma=smooth, axes=(0,1))
            deform.fill(data)
        shift_array = deform.content()
        return Aligner(self.image_w, self.image_h, shift_array)
