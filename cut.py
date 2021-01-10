import numpy as np
import sys

image = np.load(sys.argv[1])

try:
	image = image["arr_0"]
except:
	pass

x1 = int(sys.argv[2])
y1 = int(sys.argv[3])
x2 = int(sys.argv[4])
y2 = int(sys.argv[5])

image[y1:y2, x1:x2] = 0
np.savez_compressed(sys.argv[6], image)

