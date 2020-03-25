import movement
import json
import sys
import numpy as np
import os

import common

def shift_image(image, t):
	shape = image.shape
	h = shape[0]
	w = shape[1]
	shifted = np.zeros(shape)
	for y in range(h):
		for x in range(w):
			oy, ox = t.reverse(y, x)
			oy = round(oy)
			ox = round(ox)
			if oy < 0 or ox < 0 or oy >= h or ox >= w:
				continue
			shifted[y][x] = image[oy][ox]
	return shifted

images = common.listfiles(sys.argv[1], ".npy")
with open(sys.argv[2]) as f:
	shiftsf = json.load(f)

shifts = {}
names = []

for name1 in shiftsf:
	shifts[name1] = {}
	names.append(name1)
	for name2 in shiftsf:
		if shiftsf[name1][name2] is None:
			continue
		shifts[name1][name2] = movement.Movement.deserialize(shiftsf[name1][name2])

# find image with minimal shift to other
name0 = None
mind2 = None
maxc = None
for name in names:
	d2 = 0
	if maxc is not None and len(shifts[name]) < maxc:
		continue
	maxc = len(shifts[name])
	for name2 in shifts[name]:
		shift = shifts[name][name2]
		d2 += shift.dx**2 + shift.dy**2
	if mind2 is None or d2 < mind2:
		mind2 = d2
		name0 = name

for name, filename in images:
	if name0 not in shifts[name]:
		continue
	image = np.load(filename)
	t = shifts[name][name0]
	shifted = shift_image(image, t)
	np.save(os.path.join(sys.argv[3], name + ".npy"), shifted)

