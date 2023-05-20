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
import vstarstack.library.data
from vstarstack.library.data import DataFrame

def simple_add(input_filenames : list) -> DataFrame:
    """Just add images"""

    summary = {}
    summary_weight = {}
    sum_opts = {}

    for filename in input_filenames:
        img = DataFrame.load(filename)
        params = img.params

        for channel_name in img.get_channels():
            channel, opts = img.get_channel(channel_name)
            if opts["encoded"]:
                continue
            if opts["weight"]:
                continue

            if channel_name in img.links["weight"]:
                weight_channel = img.links["weight"][channel_name]
                weight, _ = img.get_channel(weight_channel)
            else:
                weight = np.ones(channel.shape, dtype=np.float64)

            if channel_name not in summary:
                summary[channel_name] = channel.astype(np.float64)
                summary_weight[channel_name] = weight
            else:
                try:
                    summary[channel_name] += channel
                    summary_weight[channel_name] += weight
                except Exception:
                    print(f"Can not add image {filename}. Skipping")

            sum_opts[channel_name] = opts

    result = vstarstack.library.data.DataFrame()
    for channel_name, channel in summary.items():
        print(channel_name)
        result.add_channel(channel, channel_name, **(sum_opts[channel_name]))
        result.add_channel(
            summary_weight[channel_name], "weight-"+channel_name, weight=True)
        result.add_channel_link(channel_name, "weight-"+channel_name, "weight")

    for param in params:
        result.add_parameter(params[param], param)

    return result

def mean(input_filenames : list) -> DataFrame:
    """Just add images"""

    counts = {}
    summary = {}
    summary_weight = {}
    sum_opts = {}

    for filename in input_filenames:
        img = DataFrame.load(filename)
        params = img.params

        for channel_name in img.get_channels():
            channel, opts = img.get_channel(channel_name)
            if opts["encoded"]:
                continue
            if opts["weight"]:
                continue

            if channel_name in img.links["weight"]:
                weight_channel = img.links["weight"][channel_name]
                weight, _ = img.get_channel(weight_channel)
            else:
                weight = np.ones(channel.shape, dtype=np.float64)

            if channel_name not in summary:
                counts[channel_name] = 0
                summary[channel_name] = channel.astype(np.float64)
                summary_weight[channel_name] = weight
            else:
                try:
                    summary[channel_name] += channel
                    summary_weight[channel_name] += weight
                    counts[channel_name] += 1
                except Exception:
                    print(f"Can not add image {filename}. Skipping")

            sum_opts[channel_name] = opts

    result = vstarstack.library.data.DataFrame()
    for channel_name, channel in summary.items():
        print(channel_name)
        count = counts[channel_name]
        result.add_channel(channel/count,
                           channel_name,
                           **(sum_opts[channel_name]))

        result.add_channel(summary_weight[channel_name]/count,
                           "weight-"+channel_name,
                           weight=True)

        result.add_channel_link(channel_name, "weight-"+channel_name, "weight")

    for param in params:
        result.add_parameter(params[param], param)

    return result

def _read_and_prepare(dataframe, channel, lows, highs):
    """Remove invalid points from image"""
    image, opts = dataframe.get_channel(channel)
    if opts["encoded"]:
        return None, None, None
    if opts["weight"]:
        return None, None, None

    if channel in dataframe.links["weight"]:
        weight_channel = dataframe.links["weight"][channel]
        weight, _ = dataframe.get_channel(weight_channel)
    else:
        weight = np.ones(image.shape, dtype=np.float64)

    image /= weight
    image[np.where(weight == 0)] = 0
    if channel in lows:
        too_low_idx = np.where(image < lows[channel])
        image[too_low_idx] = 0
        weight[too_low_idx] = 0

    if channel in highs:
        too_high_idx = np.where(image > highs[channel])
        image[too_high_idx] = 0
        weight[too_high_idx] = 0

    return image, weight, opts

def _calculate_mean(input_filenames, lows, highs):
    """Calculate mean value of images, where they are fitted between low and high"""
    mean = {}
    mean_weight = {}

    for filename in input_filenames:
        img = DataFrame.load(filename)
        for channel in img.get_channels():
            image, weight, _ = _read_and_prepare(img, channel, lows, highs)
            if image is None:
                continue

            if channel not in mean:
                mean[channel] = image * weight
                mean_weight[channel] = weight
            else:
                mean[channel] += image*weight
                mean_weight[channel] += weight

    for channel in mean:
        mean[channel] /= mean_weight[channel]
        mean[channel][np.where(mean_weight[channel] == 0)] = 0
    return mean, mean_weight

def _calculate_sum(input_filenames, lows, highs):
    """Calculate sum value of images, where they are fitted between low and high"""
    summary = {}
    summary_weight = {}
    summary_opts = {}
    params = {}

    for filename in input_filenames:
        img = DataFrame.load(filename)
        params = img.params
        for channel in img.get_channels():
            image, opts = img.get_channel(channel)
            if opts["encoded"]:
                continue
            if opts["weight"]:
                continue

            if channel in img.links["weight"]:
                weight_channel = img.links["weight"][channel]
                weight, _ = img.get_channel(weight_channel)
            else:
                weight = np.ones(image.shape, dtype=np.float64)

            image[np.where(weight == 0)] = 0
            if channel in lows:
                too_low_idx = np.where(image < lows[channel]*weight)
                image[too_low_idx] = 0
                weight[too_low_idx] = 0

            if channel in highs:
                too_high_idx = np.where(image > highs[channel]*weight)
                image[too_high_idx] = 0
                weight[too_high_idx] = 0

            if channel not in summary:
                summary[channel] = image
                summary_weight[channel] = weight
                summary_opts[channel] = opts
            else:
                summary[channel] += image
                summary_weight[channel] += weight

    return summary, summary_weight, summary_opts, params

def _calculate_sigma(input_filenames, summary, summary_weight, lows, highs):
    """Calculate sigma in each pixel"""
    sigma = {}
    nums = {}
    for filename in input_filenames:
        img = DataFrame.load(filename)

        for channel in img.get_channels():
            image, weight, _ = _read_and_prepare(img, channel, lows, highs)
            if image is None:
                continue

            delta2 = (image - summary[channel])**2
            if channel not in sigma:
                sigma[channel] = delta2 * (weight != 0)
                nums[channel] = (weight != 0).astype(np.int)
            else:
                sigma[channel] += delta2 * (weight != 0)
                nums[channel] += (weight != 0).astype(np.int)

    for channel in sigma:
        sigma[channel] = np.sqrt(sigma[channel] / nums[channel])
        sigma[channel][np.where(nums[channel] == 0)] = 0
        sigma[channel][np.where(summary_weight[channel] == 0)] = 0

    return sigma

def _sigma_clip_step(input_filenames, lows, highs, sigma_k):
    """Single step of sigma clipping"""
    mean, mean_weight = _calculate_mean(input_filenames, lows, highs)
    sigma = _calculate_sigma(input_filenames, mean, mean_weight, lows, highs)

    lows = {}
    highs = {}
    for channel, mean_value in mean.items():
        lows[channel] = mean_value - sigma[channel] * sigma_k
        highs[channel] = mean_value + sigma[channel] * sigma_k

    return lows, highs

def sigma_clip(input_filenames : list,
               sigma_k : float,
               steps : int) -> DataFrame:
    """Sigma clipped summary of images"""
    lows = {}
    highs = {}

    for _ in range(steps):
        lows, highs = _sigma_clip_step(input_filenames, lows, highs, sigma_k)

    summary, weight, opts, params = _calculate_sum(input_filenames, lows, highs)

    result = DataFrame()
    for channel_name, channel in summary.items():
        result.add_channel(channel, channel_name, **(opts[channel_name]))
        result.add_channel(weight[channel_name],  "weight-"+channel_name, weight=True)
        result.add_channel_link(channel_name, "weight-"+channel_name, "weight")

    for param in params:
        result.add_parameter(params[param], param)

    return result
