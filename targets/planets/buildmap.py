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

import targets.planets.projection
import cfg
import usage
import common
import os
import math
import numpy as np

def build_surface_map(image, a, b, angle, rot, maph, exposure):
	W = image.shape[1]
	H = image.shape[0]
	proj = targets.planets.projection.PlanetProjection(W, H, a, b, angle, rot)
	surface = np.zeros((maph, 2*maph))
	mask = np.zeros((maph, 2*maph))
	for y in range(maph):
		lat = (maph/2 - y) / (maph) * math.pi
		#print()
		for x in range(2*maph):
			lon = x / maph * math.pi
			if lon > math.pi/2 and lon < 3*math.pi/2:
				continue
			X, Y = proj.from_planet_coordinates(lon, lat)
		#	print(X, Y)
			res, pix = common.getpixel(image, Y, X)
			if res:
				surface[y,x] = pix
				mask[y,x] = exposure
	return surface, mask

def process_file(filename, mapname):
	image = common.data_load(filename)

	exposure = image["meta"]["params"]["exposure"]
	a = 63
	b = 58
	angle = 0.29736
	rot = 0
	maph = cfg.config["planets"]["map_resolution"]

	mapimage = common.data_create(image["meta"]["tags"], image["meta"]["params"])

	for channel in image["meta"]["channels"]:
		if channel in image["meta"]["encoded_channels"]:
			continue

		print(channel)
		layer = image["channels"][channel]
		sm, mask = build_surface_map(layer, a, b, angle, rot, maph, exposure)
		common.data_add_channel(mapimage, sm, channel)
		common.data_add_channel(mapimage, mask, "mask")
	
	common.data_store(mapimage, mapname)

def process_path(npys, maps):
	files = common.listfiles(npys, ".zip")
	for name, filename  in files:
		print(name)
		out = os.path.join(maps, name + ".zip")
		process_file(filename, out)

def process(argv):
	if len(argv) > 0:
		input = argv[0]
		output = argv[1]
		if os.path.isdir(input):
			process_path(input, output)
		else:
			process_file(input, output)
	else:
		process_path(cfg.config["planets"]["paths"]["cutted"], cfg.config["planets"]["paths"]["maps"])


commands = {
	"*" : (process, "build surface map from image", "cutted/ maps/"),
}

def run(argv):
	usage.run(argv, "planets buildmap", commands)
