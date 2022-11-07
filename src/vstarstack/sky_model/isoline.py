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

# Build isocurves of sky brightness in each channel

import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import fmin_cobyla
from functools import partial

def error(xs, ys, z):
	K = 10
	errors = []
	vals = np.polyval(z, xs)
	n = len(xs)
	for i in range(n):
		x = xs[i]
		y = ys[i]
		v = vals[i]
		errors.append(v-y)

	miderr = 0
	for i in range(n):
		miderr += errors[i]**2
	miderr = (miderr / n) ** 0.5

	goodxs = []
	err = 0
	cnt = 0
	for i in range(n):
		if abs(errors[i]) < miderr * K:
			err += errors[i]**2
			cnt += 1
			goodxs.append(xs[i])
	err = (err/cnt)**0.5
	return err, goodxs
	

def isolines(img):
	minpoints = 100
	N = 20
	shape = img.shape

	min_v = np.amin(img)
	max_v = np.amax(img)

	vals = []
	for i in range(N):
		v = ((1 + i) / (N+1))**2
		val = max_v * v + min_v * (1-v)
		vals.append(val)

	curves = []

	for val in vals:
		iso = img.round(3) == round(val,3)
		ys,xs = np.where(iso)
		if len(xs) >= minpoints:
			rmax = 3
			for r in range(1,rmax+1):
				z = np.polyfit(xs, ys, r)
				err, goodxs = error(xs, ys, z)
				if err < shape[0] / N / 15 or r == rmax:
					p = np.poly1d(z)
					break
			
			xmin = min(goodxs)
			xmax = max(goodxs)
			curves.append((val, xmin, xmax, z))
	return curves

def is_right(tan, dir):
	return dir[0]*tan[1] - dir[1]*tan[0] > 0

def c1(X, z):
	x, y = X
	f = np.polyval(z, [x])[0]
	return -abs(f - y)

def objective(X, P):
	return np.sqrt((X[0] - P[0])**2 + (X[1] - P[1])**2)

def approximate(iso, x, y, w, h):
	ni = len(iso)
	ds = []
	vs = []
	for i in range(ni):
		val, xmin, xmax, z = iso[i]
		P = (x,y)
		X = fmin_cobyla(objective, args=(P,), x0=[x,y], cons=[c1], consargs=[z], maxfun=1e6)
		D = objective(X, P)
		der = np.polyder(z)
		tan = np.polyval(der, [X[0]])[0]
		dir = P - X
		if not is_right((1, tan), dir):
			D = -D
		ds.append(D)
		vs.append(val)
	z = np.polyfit(ds, vs, 3)
	val = np.polyval(z, [0])[0]
	return val

def curve(x, z, xmin, xmax):
	if x >= xmin and x <= xmax:
		return np.polyval(z, [x])[0]
	if x < xmin:
		x0 = xmin
	else:
		x0 = xmax
	der = np.polyder(z)
	tan = np.polyval(der, [x0])[0]
	return np.polyval(z, [x0])[0] + tan * (x-x0)
		

def approximate_image(image):
	blur = 51
	shape = image.shape
	iso = isolines(image)
	ni = len(iso)

	h = int(shape[0])
	w = int(shape[1])
	app = np.zeros((h,w))
	for x in range(w):
		ys   = [curve(x, item[3], item[1], item[2]) for item in iso]
		vals = [item[0] for item in iso]
		z = np.polyfit(ys, vals, 3)
		for y in range(h):
			app[y,x] = np.polyval(z, [y])[0]
	
	app = cv2.GaussianBlur(app, (blur, blur), 0)
	return app

def model(img):
	blur = 31
	sky = cv2.GaussianBlur(img, (blur, blur), 0)

	maxval = np.amax(sky)
	sky    = sky / maxval
	sky    = approximate_image(img) * maxval

	return sky
