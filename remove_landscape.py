import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import sys
import common
import cv2
import os

def hasnbr(y0, x0, bs):
	h = bs.shape[0]
	w = bs.shape[1]
	has = False
	for y in range(y0-1, y0+2):
		for x in range(x0-1, x0+2):
			if y >= 0 and x >=0 and y < h and x < w and (x != x0 or y != y0):
				if bs[y][x] == 0:
					return True
	return False

size=28

ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "landscape_model") 

files = common.listfiles(sys.argv[1], ".npy")


with open(ROOT_PATH + "/model.json") as f:
	s = f.read()
	model = tf.keras.models.model_from_json(s)
model.load_weights(ROOT_PATH + "/model.weights")


for name, filename in files:
	print(name)
	image = np.load(filename)
	h = image.shape[0]
	w = image.shape[1]

	if image.shape[2] == 3:
		cimage = np.zeros((h, w, 4))
		cimage[:,:,0:3] = image
		cimage[:,:,3] = 1
	else:
		cimage = image

	wc = int((w / size + 2) / 2)
	hc = int((h / size + 2) / 2)

	lts = np.zeros((hc, wc))
	rts = np.zeros((hc, wc))
	lbs = np.zeros((hc, wc))
	rbs = np.zeros((hc, wc))

	for y in range(hc):
		posy = y * size
		for x in range(wc):
			posx = x * size
			lt = common.prepare_image_for_model(image[posy:posy+size, posx:posx+size])
			rt = common.prepare_image_for_model(image[posy:posy+size, w-posx-size:w-posx])
			lb = common.prepare_image_for_model(image[h-posy-size:h-posy, posx:posx+size])
			rb = common.prepare_image_for_model(image[h-posy-size:h-posy, w-posx-size:w-posx])
			bls = np.array([lt, rt, lb, rb])
			bls = np.reshape(bls, (len(bls), bls.shape[1], bls.shape[2], 1))
			predict = model.predict(bls)
			predict = [np.argmax(predict[i]) for i in range(len(predict))]
	#		print(predict)
			lts[y][x] = predict[0]
			rts[y][x] = predict[1]
			lbs[y][x] = predict[2]
			rbs[y][x] = predict[3]

	for y in range(hc):
		posy = y * size
		for x in range(wc):
			posx = x * size

			if lts[y][x] == 0 and hasnbr(y, x, lts):
				cimage[posy:posy+size, posx:posx+size] = 0
			if rts[y][x] == 0 and hasnbr(y, x, rts):
				cimage[posy:posy+size, w-posx-size:w-posx] = 0
			if lbs[y][x] == 0 and hasnbr(y, x, lbs):
				cimage[h-posy-size:h-posy, posx:posx+size] = 0
			if rbs[y][x] == 0 and hasnbr(y, x, rbs):
				cimage[h-posy-size:h-posy, w-posx-size:w-posx] = 0
	np.save(os.path.join(sys.argv[2], name + ".npy"), cimage)

