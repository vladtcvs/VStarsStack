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

import vstarstack.common
import vstarstack.data
import numpy as np
import sys
import os
import json
import vstarstack.cfg

def run(argv):
	if len(argv) < 3:
		npypath = vstarstack.cfg.config["paths"]["npy-fixed"]
		jsonpath = vstarstack.cfg.config["paths"]["descs"]
		cutpath = vstarstack.cfg.config["paths"]["npy-fixed"]
	else:
		npypath = argv[0]
		jsonpath = argv[1]
		cutpath = argv[2]

	if len(argv) > 3:
		margin = int(argv[3])
	else:
		margin = vstarstack.cfg.config["compact_objects"]["margin"]

	require_size = vstarstack.cfg.config["compact_objects"]["require_size"]

	files = vstarstack.common.listfiles(jsonpath, ".json")
	for name, filename in files:
		print("Loading info: %s" % name)
		with open(filename) as f:
			detection = json.load(f)

	maxr = 0
	for name, filename in files:
		with open(filename) as f:
			detection = json.load(f)

		r = int(detection["compact_object"]["r"])
		if r > maxr:
			maxr = r
	maxr = int(maxr+0.5)+margin

	for name, filename in files:
		print(name)
		with open(filename) as f:
			detection = json.load(f)

		x = int(detection["compact_object"]["x"])
		y = int(detection["compact_object"]["y"])
		r = int(detection["compact_object"]["r"])
		left   = int(x - maxr)
		right  = int(x + maxr)
		top    = int(y - maxr)
		bottom = int(y + maxr)

		if left < 0:
			left = 0
		if top < 0:
			top = 0

		imagename = os.path.join(npypath, name + ".zip")
		try:
			image = vstarstack.data.DataFrame.load(imagename)
		except:
			print("Can not load ", name)
			continue

		weight_links = dict(image.links["weight"])

		for channel in image.get_channels():
			img, opts = image.get_channel(channel)
			if opts["encoded"]:
				image.remove_channel(channel)
				continue

			img = img[top:bottom+1, left:right+1]

			if require_size:
				if img.shape[0] != 2*maxr + 1:
					print("\tSkip %s" % channel)
					image.remove_channel(channel)
					continue
				if img.shape[1] != 2*maxr + 1:
					print("\tSkip %s" % channel)
					image.remove_channel(channel)
					continue

			detection["roi"] = {
				"x1" : left,
				"y1" : top,
				"x2" : right,
				"y2" : bottom
			}
			image.add_channel(img, channel, **opts)
			with open(filename, "w") as f:
				json.dump(detection, f, indent=4, ensure_ascii=False)

		for ch in weight_links:
			image.add_channel_link(ch, weight_links[ch], "weight")

		outname = os.path.join(cutpath, name + ".zip")
		image.store(outname)
