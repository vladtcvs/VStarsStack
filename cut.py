import numpy as np
import sys

image = np.load(sys.argv[1])

x1 = int(sys.argv[2])
y1 = int(sys.argv[3])
x2 = int(sys.argv[4])
y2 = int(sys.argv[5])

image[y1:y2, x1:x2] = np.array([0, 0, 0, 0])
np.save(sys.argv[6], image)

