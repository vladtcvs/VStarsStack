import numpy as np
import sys

import matplotlib.pyplot as plt

power = 1
mul = 1

#power = 0.4
#mul = 2

path=sys.argv[1]
img = np.load(path)

print(img.shape)

img = img[:,:,0:3]

#img = img - np.average(img)
img = np.clip(img, 0, 1e6)
#img = np.power(img, power)
amax = np.amax(img)
if len(img.shape) == 2:
	plt.imshow(img/amax, cmap="gray")
else:
	img = img/amax * mul
#	img = img[:,:,3]
	plt.imshow(img)
	
plt.show()

