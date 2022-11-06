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

import cfg
import sys
import common
import projection.perspective
import math
import os
import numpy as np
import multiprocessing as mp
import usage
import data

ncpu = max(1, mp.cpu_count()-1)

def fix(image, image_weight_layer, proj, image_weight=1):
	if image_weight_layer is None:
		image_weight_layer = np.ones(image.shape)*image_weight

	if cfg.distorsion is None:
		return image, image_weight_layer

	h = image.shape[0]
	w = image.shape[1]

	fixed  = np.zeros((h, w))
	fixed_weight = np.zeros((h,w))

	a = cfg.distorsion["a"]
	b = cfg.distorsion["b"]
	c = cfg.distorsion["c"]

	v0 = np.zeros((3,))
	v0[0] = 1
	v0[1] = 0
	v0[2] = 0

	for y in range(h):
		for x in range(w):
			fixedlat, fixedlon = proj.project(y, x)
			fixedangle = math.acos(math.cos(fixedlat)*math.cos(fixedlon))
			k = a * fixedangle**2 + b * fixedangle + c
			angle = fixedangle * k
			if angle < 1e-12:
				lat = fixedlat
				lon = fixedlon
			else:
				p = np.empty((3,))
				p[0] = 0
				p[1] = math.cos(fixedlat) * math.sin(fixedlon)
				p[2] = math.sin(fixedlat)
				p /= (p[1]**2+p[2]**2)**0.5
				v = v0 * math.cos(angle) + p * math.sin(angle)
				lat = math.asin(v[2])
				lon = math.atan2(v[1], v[0])

			fy, fx = proj.reverse(lat, lon)

			_, pixel        = common.getpixel(image, fy, fx, False)
			_, pixel_weight = common.getpixel(image_weight_layer, fy, fx, False)
			
			fixed[y][x] = pixel
			fixed_weight[y][x] = pixel_weight

	return fixed, fixed_weight

def dedistorsion(name, fname, outfname, proj):
	print(name)
	dataframe = data.DataFrame.load(fname)
	for channel in dataframe.get_channels():
		image, opts = dataframe.get_channel(channel)
		if opts["encoded"]:
			continue
		if opts["weight"]:
			continue
		weight_channel = dataframe.links["weight"][channel]
		image_weight,_ = dataframe.get_channel(weight_channel)
		

		fixed, fixed_weight = fix(image, image_weight, proj, 1)
		dataframe.add_channel(fixed, channel, **opts)
		dataframe.add_channel(fixed_weight, weight_channel, weight=True)
		dataframe.add_channel_link(channel, weight_channel, "weight")

	dataframe.store(outfname)

def process_file(argv):
	proj = projection.Projection(cfg.camerad["W"], cfg.camerad["H"], cfg.camerad["F"], cfg.camerad["w"], cfg.camerad["h"])
	infname = argv[0]
	outfname = argv[1]
	name = os.path.splitext(os.path.basename(infname))[0]
	dedistorsion(name, infname, outfname, proj)

def process_dir(argv):
	proj = projection.perspective.Projection(cfg.camerad["W"], cfg.camerad["H"], cfg.camerad["F"], cfg.camerad["w"], cfg.camerad["h"])
	inpath = argv[0]
	outpath = argv[1]
	files = common.listfiles(inpath, ".zip")
	pool = mp.Pool(ncpu)
	pool.starmap(dedistorsion, [(name, fname, os.path.join(outpath, name + ".zip"), proj) for name, fname in files])
	pool.close()

def process(argv):
	if len(argv) > 0:
		if os.path.isdir(argv[0]):
			process_dir(argv)
		else:
			process_file(argv)
	else:
		process_dir([cfg.config["paths"]["npy-fixed"], cfg.config["paths"]["npy-fixed"]])

commands = {
	"*" :  (process, "Remove distrosion", "(input.file output.file | [input/ output/])"),
}

def run(argv):
	usage.run(argv, "image-fix distorsion", commands)

