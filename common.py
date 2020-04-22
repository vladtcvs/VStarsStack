import os
import imageio
from skimage.color import rgb2gray
import cv2
from skimage import data, img_as_float
from skimage import exposure
import numpy as np

def listfiles(path, ext):
	images = []
	for f in os.listdir(path):
		filename = os.path.abspath(os.path.join(path, f))
		if not os.path.isfile(filename) or f[-len(ext):].lower() != ext:
			continue
		name = os.path.splitext(f)[0]
		images.append((name, filename))
	images.sort(key=lambda item : item[0])
	return images

def length(vec):
	return (vec[0]**2+vec[1]**2)**0.5

def norm(vec):
	l = (vec[0]**2+vec[1]**2)**0.5
	return (vec[0] / l, vec[1] / l)	

def prepare_image_for_model(image):
	image = image[:,:,0:3]
	image = rgb2gray(image)
	image = cv2.GaussianBlur(image, (3, 3), 0)
	am = np.amax(image)
	if am >= 1:
		image /= am
	image = exposure.equalize_hist(image)
	return image

