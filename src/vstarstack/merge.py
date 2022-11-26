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

from calendar import c
import numpy as np
import sys
import matplotlib.pyplot as plt

import vstarstack.cfg
import vstarstack.common
import vstarstack.data
import vstarstack.usage

def simple_add(argv):
	if len(argv) > 0:
		path_images = argv[0]
		out = argv[1]
	else:
		path_images = vstarstack.cfg.config["paths"]["aligned"]
		out = vstarstack.cfg.config["paths"]["output"]

	imgs = vstarstack.common.listfiles(path_images, ".zip")

	shape = None
	params = {}

	summary = {}
	summary_weight = {}
	sum_opts = {}

	for name, filename in imgs:
		print(name, filename)
		img = vstarstack.data.DataFrame.load(filename)

		params = img.params

		for channel in img.get_channels():
			image, opts = img.get_channel(channel)
			if opts["encoded"]:
				continue
			if opts["weight"]:
				continue

			if channel in img.links["weight"]:
				weight_channel = img.links["weight"][channel]
				weight,_ = img.get_channel(weight_channel)
			else:
				weight = np.ones(image.shape, dtype=np.float64)

			if channel not in summary:
				summary[channel] = image.astype(np.float64)
				summary_weight[channel] = weight
			else:
				try:
					summary[channel] += image
					summary_weight[channel] += weight
				except Exception:
					print("Can not add image %s. Skipping" % name)
					pass
			sum_opts[channel] = opts

	result = vstarstack.data.DataFrame()
	for channel in summary:
		print(channel)
		result.add_channel(summary[channel], channel, **(sum_opts[channel]))
		result.add_channel(summary_weight[channel], "weight-"+channel, weight=True)
		result.add_channel_link(channel, "weight-"+channel, "weight")

	for param in params:
		result.add_parameter(params[param], param)

	result.store(out)

def read_and_prepare(img, channel, lows, highs):
	image, opts = img.get_channel(channel)
	if opts["encoded"]:
		return None, None, None
	if opts["weight"]:
		return None, None, None

	if channel in img.links["weight"]:
		weight_channel = img.links["weight"][channel]
		weight,_ = img.get_channel(weight_channel)
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

def calculate_mean(imgs, lows, highs):
	summary = {}
	summary_weight = {}
	summary_opts = {}

	print("Calculate mean value where value between low and high")
	for name, filename in imgs:
#		print(name, filename)
		img = vstarstack.data.DataFrame.load(filename)
		for channel in img.get_channels():
			image, weight, opts = read_and_prepare(img, channel, lows, highs)
			if image is None:
				continue

			if channel not in summary:
				summary[channel] = image * weight
				summary_weight[channel] = weight
				summary_opts[channel] = opts
			else:
				summary[channel] += image*weight
				summary_weight[channel] += weight

	for channel in summary:
		summary[channel] /= summary_weight[channel]
		summary[channel][np.where(summary_weight[channel] == 0)] = 0
	return summary, summary_weight, summary_opts

def calculate_sum(imgs, lows, highs):
	summary = {}
	summary_weight = {}
	summary_opts = {}
	params = {}

	print("Calculate mean value where value between low and high")
	for name, filename in imgs:
		print(name, filename)
		img = vstarstack.data.DataFrame.load(filename)
		params = img.params
		for channel in img.get_channels():
			image, opts = img.get_channel(channel)
			if opts["encoded"]:
				continue
			if opts["weight"]:
				continue

			if channel in img.links["weight"]:
				weight_channel = img.links["weight"][channel]
				weight,_ = img.get_channel(weight_channel)
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

def calculate_sigma(imgs, summary, summary_weight, lows, highs):
	print("Calculate sigma")
	sigma = {}
	nums = {}
	for name, filename in imgs:
#		print(name, filename)
		img = vstarstack.data.DataFrame.load(filename)

		for channel in img.get_channels():
			image, weight, _ = read_and_prepare(img, channel, lows, highs)
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

def sigma_clip_step(imgs, lows, highs, sigma_k):
	summary, summary_weight, summary_opts = calculate_mean(imgs, lows, highs)
	sigma = calculate_sigma(imgs, summary, summary_weight, lows, highs)

	print("Calculate new low and high")
	lows = {}
	highs = {}
	for channel in summary:
		lows[channel]  = summary[channel] - sigma[channel] * sigma_k
		highs[channel] = summary[channel] + sigma[channel] * sigma_k

	return summary, summary_weight, lows, highs, summary_opts

def sigma_clip(argv):
	if len(argv) > 0:
		path_images = argv[0]
		out = argv[1]
		sigma_k = float(argv[2])
	else:
		path_images = vstarstack.cfg.config["paths"]["aligned"]
		out = vstarstack.cfg.config["paths"]["output"]
		sigma_k = vstarstack.cfg.config["sigma_clip_coefficient"]

	imgs = vstarstack.common.listfiles(path_images, ".zip")

	lows = {}
	highs = {}
	_, _, lows, highs, _ = sigma_clip_step(imgs, lows, highs, sigma_k)
	_, _, lows, highs, _ = sigma_clip_step(imgs, lows, highs, sigma_k)
	summary, weight, opts, params = calculate_sum(imgs, lows, highs)

	result = vstarstack.data.DataFrame()
	for channel in summary:
		result.add_channel(summary[channel], channel, **(opts[channel]))
		result.add_channel(weight[channel],  "weight-"+channel, weight=True)
		result.add_channel_link(channel, "weight-"+channel, "weight")

	for param in params:
		result.add_parameter(params[param], param)

	result.store(out)

commands = {
	"simple"     : (simple_add, "simple add images"),
	"sigma-clip" : (sigma_clip, "add images with sigma clipping"),
}

def run(argv):
	vstarstack.usage.run(argv, "merge", commands, autohelp=True)
