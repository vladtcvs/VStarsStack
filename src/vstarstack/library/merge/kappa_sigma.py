#
# Copyright (c) 2023-2024 Vladislav Tsendrovskii
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

from vstarstack.library.image_process.normalize import normalize, denormalize
from vstarstack.library.data import DataFrame
from vstarstack.library.common import IImageSource

def _hard_clip(delta : np.ndarray, sigma : np.ndarray, kappa : float) -> np.ndarray:
    """Hard clip where |delta| > sigma * kappa"""
    return (abs(delta) - sigma * kappa <= 0).astype(np.int32)

def _calculate_clip(image : np.ndarray, mean : np.ndarray, sigma : np.ndarray, kappa : float) -> np.ndarray:
    """Calculate clip array"""
    delta = image - mean
    return _hard_clip(delta, sigma, kappa)

def _read_and_prepare(dataframe : DataFrame, channel : str):
    """Remove invalid points from image"""
    image, opts = dataframe.get_channel(channel)
    weight, _, _ = dataframe.get_linked_channel(channel, "weight")
    if weight is None:
        if (weight_k := dataframe.get_parameter("weight")) is None:
            weight_k = 1
        weight = np.ones(image.shape, dtype=np.float64) * weight_k

    image[np.where(weight < 1e-12)] = 0
    return image, weight, opts

def _calculate_mean(images : IImageSource, means: dict, sigmas: dict, kappa: float):
    """Calculate mean value of images"""
    signal_sum = {}
    weight_sum = {}

    for img in images.items():
        img = normalize(img, deepcopy=False)
        for channel in img.get_channels():
            if not img.get_channel_option(channel, "signal"):
                continue

            signal, weight, _ = _read_and_prepare(img, channel)

            if channel in means and channel in sigmas:
                clip = _calculate_clip(signal, means[channel], sigmas[channel], kappa)
            else:
                clip = np.ones(signal.shape, dtype=np.int32)

            if channel not in signal_sum:
                signal_sum[channel] = signal * weight * clip
                weight_sum[channel] = weight * clip
            else:
                signal_sum[channel] += signal * weight * clip
                weight_sum[channel] += weight * clip

    new_means = {}
    for channel in signal_sum:
        new_means[channel] = signal_sum[channel] / weight_sum[channel]
        new_means[channel][np.where(weight_sum[channel] < 1e-12)] = 0

    return new_means

def _calculate_sigma(images : IImageSource, means : dict, sigmas : dict, kappa : float):
    """Calculate sigma in each pixel"""
    sigma = {}
    clips = {}

    for img in images.items():
        img = normalize(img, deepcopy=False)
        for channel in img.get_channels():
            if not img.get_channel_option(channel, "signal"):
                continue

            signal, _, _ = _read_and_prepare(img, channel)
            if channel in means and channel in sigmas:
                clip = _calculate_clip(signal, means[channel], sigmas[channel], kappa)
            else:
                clip = np.ones(signal.shape, dtype=np.int32)

            delta2 = ((signal - means[channel])**2) * clip
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

def _calculate_sum(images : IImageSource, means: dict, sigmas: dict, kappa: float):
    """Calculate mean value of images"""
    signal_sum = {}
    weight_sum = {}
    channel_opts = {}

    for img in images.items():
        img = normalize(img, deepcopy=False)
        for channel in img.get_channels():
            if not img.get_channel_option(channel, "signal"):
                continue

            signal, weight, opts = _read_and_prepare(img, channel)
            if channel not in channel_opts:
                channel_opts[channel] = opts

            if channel in means and channel in sigmas:
                clip = _calculate_clip(signal, means[channel], sigmas[channel], kappa)
            else:
                clip = np.ones(signal.shape, dtype=np.int32)

            if channel not in signal_sum:
                signal_sum[channel] = signal * weight * clip
                weight_sum[channel] = weight * clip
            else:
                signal_sum[channel] += signal * weight * clip
                weight_sum[channel] += weight * clip

    for channel in signal_sum:
        signal_sum[channel][np.where(weight_sum[channel] < 1e-12)] = 0
        weight_sum[channel][np.where(weight_sum[channel] < 1e-12)] = 0

    channel_opts["normed"] = False
    return signal_sum, weight_sum, channel_opts

def kappa_sigma(images: IImageSource,
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
        means = _calculate_mean(images, means, sigmas, kappa)
        sigmas = _calculate_sigma(images, means, sigmas, kappa)

    signals, weights, channel_opts = _calculate_sum(images, means, sigmas, kappa2)

    result = DataFrame(params=params)
    for channel_name, light in signals.items():
        weight = weights[channel_name]
        result.add_channel(light, channel_name, **channel_opts[channel_name])
        result.add_channel(weight,  "weight-"+channel_name, weight=True)
        result.add_channel_link(channel_name, "weight-"+channel_name, "weight")

    return result
