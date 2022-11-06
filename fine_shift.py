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
import sys
import os
import json
from scipy.spatial import Delaunay
import cfg

def isinside(triangle, point):
	p1 = triangle["p1"][0]
	p2 = triangle["p2"][0]
	p3 = triangle["p3"][0]
	d2 = p2 - p1
	d3 = p3 - p1
	d = point - p1
	
	D = d2[0]*d3[1] - d2[1]*d3[0]
	D2 = d[0]*d3[1] - d[1]*d3[0]
	D3 = d[1]*d2[0] - d[0]*d2[1]

	k2 = D2 / D
	k3 = D3 / D

	if k2 < 0 or k3 < 0:
		return False, None, None, None
	if k2 + k3 > 1:
		return False, None, None, None
	return True, 1-k2-k3, k2, k3

def shift(p, net):
	for t in net:
		inside, k1, k2, k3 = isinside(t, p)
		if inside:
			s1 = t["p1"][1]
			s2 = t["p2"][1]
			s3 = t["p3"][1]
			dp = s1 * k1 + s2 * k2 + s3 * k3
			return p + dp
	return None

def shift_image(clusters, img, imgname, baseimgname):
	simg = np.zeros(img.shape)
	shifts = []
	for cluster in clusters:
		if imgname not in cluster or baseimgname not in cluster:
			continue
		p = cluster[imgname]
		bp = cluster[baseimgname]
		x = p["x"]
		y = p["y"]
		x0 = bp["x"]
		y0 = bp["y"]
		dx = x - x0
		dy = y - y0
		shifts.append(np.array([[y0, x0], [dy, dx]]))
	shifts = np.array(shifts)
	tris = Delaunay(shifts[:,0,:])
	net = []
	for t in tris.simplices:
		print(t)
		p1 = shifts[t[0],0,:]
		s1 = shifts[t[0],1,:]
		p2 = shifts[t[1],0,:]
		s2 = shifts[t[1],1,:]
		p3 = shifts[t[2],0,:]
		s3 = shifts[t[2],1,:]
		print(p1, p2, p3)
		net.append({"p1" : (p1,s1), "p2":(p2,s2), "p3":(p3,s3)})
	
	for y in range(cfg.camerad["h"]):
		for x in range(cfg.camerad["w"]):
			p = np.array([y, x])
			sp = shift(p, net)
			if sp is None:
				continue
			sy = int(round(sp[0]))
			sx = int(round(sp[1]))
#			print(sy, sx)
			if sy < 0 or sx < 0 or sy >= img.shape[0] or sx >= img.shape[1]:
				continue
			simg[y][x] = img[sy][sx]
	return simg

def run(argv):
	imgname = argv[0]
	baseimgname = argv[1]
	npydir = argv[2]
	clustersname = argv[3]
	
	img = np.load(os.path.join(npydir, imgname + ".npy"))
	simg = np.zeros(img.shape)

	with open(clustersname) as f:
		clusters = json.load(f)

	simg = shift_image(clusters, img, imgname, baseimgname)
	np.save(argv[4], simg)

if __name__ == "__main__":
	run(sys.argv[1:])

