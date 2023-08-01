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
import vstarstack.library.common
import vstarstack.library.data
from vstarstack.library.data import DataFrame

import vstarstack.library.projection.tools
import vstarstack.library.movement.basic_movement as basic_movement

def _generate_points(height, width, len0):
    """Generate grid of pixel coordinates"""
    points = []
    for y in range(height):
        for x in range(width):
            points.append((y, x))
            if len(points) >= len0:
                yield points
                points = []
    if len(points) > 0:
        yield points

def move_image(image : np.ndarray,
               transformation : basic_movement.Movement,
               proj,
               image_weight : float,
               image_weight_layer=None):
    """Apply movement to image"""
    shape = image.shape
    h = shape[0]
    w = shape[1]

    shifted = np.zeros(shape)
    shifted_weight_layer = np.zeros(shape)

    if image_weight_layer is None:
        image_weight_layer = np.ones(shape)*image_weight

    for positions in _generate_points(h, w, w*4):
        original_positions = transformation.reverse(positions, proj)
        for position, original_position in zip(positions, original_positions):
            y, x = position
            orig_y, orig_x = original_position
            _, pixel = vstarstack.library.common.getpixel(image, orig_y, orig_x, False)
            _, pixel_weight = vstarstack.library.common.getpixel(
                image_weight_layer, orig_y, orig_x, False)

            shifted[y][x] = pixel
            shifted_weight_layer[y][x] = pixel_weight

    return shifted, shifted_weight_layer

def move_dataframe(dataframe : DataFrame,
                   transformation : basic_movement.Movement,
                   proj = None):
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

        shifted, shifted_weight = move_image(image, transformation, proj, weight)
        dataframe.add_channel(shifted, channel, **opts)
        dataframe.add_channel(shifted_weight, weight_channel, weight=True)
        dataframe.add_channel_link(channel, weight_channel, "weight")

    return dataframe
