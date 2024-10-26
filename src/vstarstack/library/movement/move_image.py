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
               input_proj, output_proj,*,
               image_weight: float = 1,
               image_weight_layer: np.ndarray | None = None,
               output_shape: tuple | None = None,
               interpolate : bool = True):
    """
    Apply movement to image
    
    Parameters:
        image (np.ndarray) - input image
        transformation (Movement) - movement which should be applied
        input_proj (Projection) - input image projection
        output_proj (Projection) - output image projection
        image_weight (float) - weight of input image, if weight layer is not provided
        image_weight_layer (np.ndarray) - weight layer of input image
        output_shape (tuple(h,w)) - dimensions of output image
        interpolate (bool) - whether interpolation should be applied during movement
    Returns:
        shifted image, shifted layer
    """
    if output_shape is None:
        shape = image.shape
    else:
        shape = output_shape

    h = shape[0]
    w = shape[1]

    shifted = np.zeros(shape)
    shifted_weight_layer = np.zeros(shape)

    if image_weight_layer is None:
        image_weight_layer = np.ones(image.shape)*image_weight

    positions = _generate_points(h, w)
    original_positions = transformation.reverse(positions.astype('double'),
                                                input_proj,
                                                output_proj)
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
    if interpolate:
        shifted = scipy.ndimage.geometric_transform(image, crdtf, output_shape=shape, order=3)
    else:
        shifted = scipy.ndimage.geometric_transform(image, crdtf, output_shape=shape, order=0)
    shifted_weight_layer = scipy.ndimage.geometric_transform(image_weight_layer, crdtf, output_shape=shape, order=3)
    return shifted, shifted_weight_layer

def move_dataframe(dataframe: DataFrame,
                   transformation: basic_movement.Movement,*,
                   input_proj = None,
                   output_proj = None,
                   output_shape : tuple | None = None,
                   interpolate: bool | None = None):
    """Apply movement to dataframe
    Parameters:
        dataframe (DataFrame) - input dataframe
        transformation (Movement) - movement which should be applied
        input_proj (Projection) - input image projection
        output_proj (Projection) - output image projection
        output_shape (tuple(h,w)) - dimensions of output image
        interpolate (bool|None) - apply spline interpolation. If None, then interpolation False when CFA layer
    Returns:
        shifted image, shifted layer"""

    if input_proj is None:
        input_proj = vstarstack.library.projection.tools.get_projection(dataframe)
    if output_proj is None:
        output_proj = input_proj

    if output_shape is None:
        w = dataframe.get_parameter("w")
        h = dataframe.get_parameter("h")
        output_shape = (h, w)

    output_dataframe = DataFrame(params=dataframe.params)
    output_dataframe.add_parameter(output_shape[0], "h")
    output_dataframe.add_parameter(output_shape[1], "w")

    for channel in dataframe.get_channels():
        image, opts = dataframe.get_channel(channel)
        if not dataframe.get_channel_option(channel, "signal"):
            continue

        if channel in dataframe.links["weight"]:
            weight_channel = dataframe.links["weight"][channel]
            weight, _ = dataframe.get_channel(weight_channel)
        else:
            weight_channel = f"weight-{channel}"
            if (w := dataframe.get_parameter("weight")) is not None:
                weight = np.ones(image.shape)*w
            else:
                weight = np.ones(image.shape)

        if interpolate is not None:
            apply_interpolate = interpolate
        else:
            apply_interpolate = not dataframe.get_channel_option(channel, "cfa")

        shifted, shifted_weight = move_image(image,
                                             transformation,
                                             input_proj,
                                             output_proj,
                                             image_weight_layer=weight,
                                             output_shape=output_shape,
                                             interpolate=apply_interpolate)

        output_dataframe.add_channel(shifted, channel, **opts)
        output_dataframe.add_channel(shifted_weight, weight_channel, weight=True)
        output_dataframe.add_channel_link(channel, weight_channel, "weight")

    return output_dataframe 
