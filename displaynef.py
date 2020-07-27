import os
import rawpy
import imageio
from os import listdir
from os.path import isfile, join
import numpy as np
import sys

import matplotlib.pyplot as plt


if __name__ == "__main__":
	img = rawpy.imread(sys.argv[1]).raw_image_visible

	amax = np.amax(img)
	if len(img.shape) == 2:
		plt.imshow(img/amax, cmap="gray")
	
	plt.show()

