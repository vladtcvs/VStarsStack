"""Image shifting"""
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
import scipy.ndimage

import vstarstack.library.common
import vstarstack.library.data
from vstarstack.library.data import DataFrame

import vstarstack.library.projection.tools
import vstarstack.library.movement.basic_movement as basic_movement

from vstarstack.library.common import getpixel

def _generate_points(height, width):
    """Generate grid of pixel coordinates"""
    points = np.zeros((height*width, 2), dtype='int')
    for y in range(height):
        for x in range(width):
            points[y * width + x, 0] = x
            points[y * width + x, 1] = y
    return points


def move_image(image: np.ndarray,
               transformation: basic_movement.Movement,
               input_proj, output_proj,
               image_weight: float,
               image_weight_layer=None):
    """Apply movement to image"""
    shape = image.shape
    h = shape[0]
    w = shape[1]

    shifted = np.zeros(shape)
    shifted_weight_layer = np.zeros(shape)

    if image_weight_layer is None:
        image_weight_layer = np.ones(shape)*image_weight

    positions = _generate_points(h, w)
    original_positions = transformation.reverse(positions.astype('double'), input_proj, output_proj)
    num = positions.shape[0]
    transform_array = np.zeros([h, w, 2], dtype='double')
    for index in range(num):
        position = positions[index]
        original_position = original_positions[index]
        x, y = position[0], position[1]
        orig_x, orig_y = original_position[0], original_position[1]
        transform_array[y, x, 0] = orig_y
        transform_array[y, x, 1] = orig_x

    crdtf = lambda pos : tuple(transform_array[pos[0], pos[1], :])
    shifted = scipy.ndimage.geometric_transform(image, crdtf, order=3)
    shifted_weight_layer = scipy.ndimage.geometric_transform(image_weight_layer, crdtf, order=3)
    return shifted, shifted_weight_layer

def move_dataframe(dataframe: DataFrame,
                   transformation: basic_movement.Movement,
                   proj=None):
    """Apply movement to dataframe"""
    if proj is None:
        proj = vstarstack.library.projection.tools.get_projection(dataframe)

    for channel in dataframe.get_channels():
        image, opts = dataframe.get_channel(channel)
        if opts["weight"]:
            continue
        if opts["encoded"]:
            continue

        weight_channel = None
        if channel in dataframe.links["weight"]:
            weight_channel = dataframe.links["weight"][channel]

        if weight_channel:
            weight, _ = dataframe.get_channel(weight_channel)
        else:
            weight = np.ones(image.shape)*1

        shifted, shifted_weight = move_image(image, transformation, proj, proj, weight)
        dataframe.add_channel(shifted, channel, **opts)
        dataframe.add_channel(shifted_weight, weight_channel, weight=True)
        dataframe.add_channel_link(channel, weight_channel, "weight")

    return dataframe
