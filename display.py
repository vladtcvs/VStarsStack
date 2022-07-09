import numpy as np
import sys
import cv2
import json
import normalize

import matplotlib.pyplot as plt

import common

power = 1

power = 0.3
#mul = 2

path=sys.argv[1]

data = common.data_load(path)



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
	img = np.clip(img, 0, 1e6)
	img = np.power(img, power)
	amax = np.amax(img)
	img /= amax

	if nch > 1:
		axs[id].imshow(img, cmap="gray")
		axs[id].set_title(channel)
		id += 1
	else:
		axs.imshow(img, cmap="gray")
		axs.set_title(channel)

plt.show()
