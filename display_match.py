import numpy as np
import sys

import matplotlib.pyplot as plt

path=sys.argv[1]
img = np.load(path)
amax = np.amax(img)
if len(img.shape) == 2:
	plt.imshow(img/amax, cmap="gray")
else:
	plt.imshow(img/amax)
	
plt.show()

