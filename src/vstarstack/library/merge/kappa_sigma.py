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

import vstarstack.library.data
from vstarstack.library.data import DataFrame


def hard_clip(delta, sigma, kappa):
    """Hard clip where |delta| > sigma * kappa"""
    return (abs(delta) - sigma * kappa <= 0).astype("float")

def calculate_clip(image, mean, sigma, kappa):
    """Calculate clip array"""
    delta = image - mean
    return hard_clip(delta, sigma, kappa)

def _read_and_prepare(dataframe, channel):
    """Remove invalid points from image"""
    image, opts = dataframe.get_channel(channel)
    if not opts["signal"]:
        return None, None, None

    if channel in dataframe.links["weight"]:
        weight_channel = dataframe.links["weight"][channel]
        weight, _ = dataframe.get_channel(weight_channel)
    else:
        weight = np.ones(image.shape, dtype=np.float64)

    image[np.where(weight == 0)] = 0
    return image, weight, opts


def _calculate_mean(images, means: dict, sigmas: dict, kappa: float):
    """Calculate mean value of images"""
    mean_image = {}
    total_weight = {}
    channel_opts = {}

    for img in images.items():
        for channel in img.get_channels():
            image, weight, opts = _read_and_prepare(img, channel)
            if image is None:
                continue
            if channel not in channel_opts:
                channel_opts[channel] = opts

            if channel in means:
                clip = calculate_clip(image, means[channel], sigmas[channel], kappa)
            else:
                clip = np.ones(image.shape)

            if channel not in mean_image:
                mean_image[channel] = image * clip
                total_weight[channel] = weight * clip
            else:
                mean_image[channel] += image * clip
                total_weight[channel] += weight * clip

    for channel in mean_image:
        mean_image[channel] = mean_image[channel] / total_weight[channel]
        mean_image[channel][np.where(total_weight[channel] == 0)] = 0

    return mean_image, total_weight, channel_opts


def _calculate_sigma(images, means, sigmas, kappa):
    """Calculate sigma in each pixel"""
    sigma = {}
    clips = {}

    for img in images.items():
        for channel in img.get_channels():
            image, _, _ = _read_and_prepare(img, channel)
            if image is None:
                continue

            if channel in means:
                if channel in sigmas:
                    clip = calculate_clip(image,
                                          means[channel],
                                          sigmas[channel],
                                          kappa)
                else:
                    clip = np.ones(image.shape)
            else:
                clip = np.ones(image.shape)

            delta2 = (image - means[channel])**2
            if channel not in sigma:
                sigma[channel] = delta2
                clips[channel] = clip
            else:
                sigma[channel] += delta2
                clips[channel] += clip

    for channel in sigma:
        sigma[channel] = np.sqrt(sigma[channel] / clips[channel])
        sigma[channel][np.where(clips[channel] == 0)] = 0

    return sigma

def kappa_sigma(images: vstarstack.library.common.IImageSource,
                kappa1: float,
                kappa2: float,
                steps: int) -> DataFrame:
    """Sigma clipped summary of images"""
    means = {}
    sigmas = {}
    params = {}

    first = next(images.items())
    if first is not None:
        params = first.params

    for step in range(steps):
        if steps > 1:
            kappa = (kappa1 * (steps-1-step) + kappa2 * step) / (steps-1)
        else:
            kappa = (kappa1 + kappa2) / 2
        means, _, _ = _calculate_mean(images, means, sigmas, kappa)
        sigmas = _calculate_sigma(images, means, sigmas, kappa)

    lights, weights, channel_opts = _calculate_mean(images,
                                                    means,
                                                    sigmas,
                                                    kappa2)

    result = DataFrame(params=params)
    for channel_name, light in lights.items():
        weight = weights[channel_name]
        result.add_channel(light, channel_name, **channel_opts[channel_name])
        result.add_channel(weight,  "weight-"+channel_name, weight=True)
        result.add_channel_link(channel_name, "weight-"+channel_name, "weight")

    return result
