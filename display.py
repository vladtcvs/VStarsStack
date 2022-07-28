import numpy as np
import sys
import cv2
import json
import normalize

import matplotlib.pyplot as plt

import common

power = 1

power = 1
#mul = 2

path=sys.argv[1]

data = common.data_load(path)


if len(sys.argv) > 2 and sys.argv[2] == "RGB":
	r = data["channels"]["R"]
	g = data["channels"]["G"]
	b = data["channels"]["B"]

	rgb = np.zeros((r.shape[0], r.shape[1], 3))
	rgb[:,:,0] = r
	rgb[:,:,1] = g
	rgb[:,:,2] = b

	plt.imshow(rgb/np.amax(rgb))
	plt.show()
	sys.exit()

if len(sys.argv) > 2:
	channels = [sys.argv[2]]
else:
	channels = data["meta"]["channels"]

nch = len(channels)
fig, axs = plt.subplots(1, nch)
fig.patch.set_facecolor('#222222')

id = 0
for channel in channels:
	print("Channel = ", channel)
	img = data["channels"][channel]
	print("Shape = ", img.shape)
	img = np.clip(img, -1e6, 1e6)
	amax = np.amax(img)
	img /= amax
	img = np.power(img, power)

	if nch > 1:
		axs[id].imshow(img, cmap="gray")
		axs[id].set_title(channel)
		id += 1
	else:
		axs.imshow(img, cmap="gray")
		axs.set_title(channel)

plt.show()
