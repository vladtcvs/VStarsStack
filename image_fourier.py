from scipy.fft import fft2, ifft2

import sys
import numpy as np

import matplotlib.pyplot as plt

def blur(image, rad):
	ft = fft2(image)
	ft[3:,:] = 0
	ft[:, 3:] = 0
	ft[2,2] = 0
#
#	for i in range(ft.shape[0]):
#		for j in range(ft.shape[1]):
#			if i**2+j**2 > rad**2:
#				ft[i,j] = 0
	return np.real(ifft2(ft))

	

if __name__ == "__main__":
	src = sys.argv[1]
	dst = sys.argv[2]
	rad = 2

	image = np.load(src)
	try:
		image = image["arr_0"]
	except:
		pass

	image = image[:,:,0:3]
	image = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
#	image = image**0.3

	imgr = image[:,:,0]
	imgg = image[:,:,1]
	imgb = image[:,:,2]

	br = blur(imgr, rad)
	bg = blur(imgg, rad)
	bb = blur(imgb, rad)

	res = np.empty(image.shape)
	res[:,:,0] = br
	res[:,:,1] = bg
	res[:,:,2] = bb

	np.savez_compressed(dst, image - res)

#	plt.imshow(image-res)
#	plt.show()
