import numpy as np
import sys
import cv2
import json
import normalize

import matplotlib.pyplot as plt

power = 1

power = 0.3
#mul = 2

path=sys.argv[1]

img = np.load(path)
try:
	img = img["arr_0"] # case of npz
except:
	pass

markstars = False

if len(sys.argv) > 2:
	power = float(sys.argv[2])

if len(sys.argv) > 3:
	markstars = True
	with open(sys.argv[3]) as f:
		stars = json.load(f)

print(img.shape)

if len(img.shape) == 3:
	img = normalize.normalize(img)
	img = img[:,:,0:3]

#img = img - np.average(img)
img = np.clip(img, 0, 1e6)
img = np.power(img, power)
amax = np.amax(img)
img /= amax

if markstars:
	for star in stars["stars"]:
		print(star)
		if len(img.shape) == 3:
			color = (1,0,0)
		else:
			color = 0
		r = round(star["size"]*5+1)
		r = round(star["size"])
		cv2.circle(img, (round(star["x"]), round(star["y"])), r, color, 2)

fig = plt.figure()
fig.patch.set_facecolor('#222222')

if len(img.shape) == 2:
	plt.imshow(img, cmap="gray")
else:
	plt.imshow(img)

plt.show()

