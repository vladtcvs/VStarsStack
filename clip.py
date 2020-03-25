import numpy as np
import sys

image = np.load(sys.argv[1])

x1 = int(sys.argv[2])
y1 = int(sys.argv[3])
x2 = image.shape[1] - int(sys.argv[4])
y2 = image.shape[0] - int(sys.argv[5])

image = image[y1:y2, x1:x2]
np.save(sys.argv[6], image)

