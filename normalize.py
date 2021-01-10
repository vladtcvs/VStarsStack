import sys
import numpy as np

def normalize(image):
	if image.shape[2] == 3:
		return image
	elif image.shape[2] == 4:
		nums = image[:,:,3]
		nums = np.where(nums == 0, 1, nums)
		image[:,:,0] /= nums
		image[:,:,1] /= nums
		image[:,:,2] /= nums
		image[:,:,3] /= nums
		return image
	else:
		raise Exception("unknown shape")

if __name__ == "__main__":
	img = np.load(sys.argv[1])
	try:
		img = img["arr_0"]
	except:
		pass
	img = normalize(img)
	np.savez_compressed(sys.argv[2], img)

