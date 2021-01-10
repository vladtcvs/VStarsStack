import cfg
import sys
import common
import projection
import math
import os
import numpy as np
import cv2

def run(argv):
	proj = projection.Projection(cfg.camerad["W"], cfg.camerad["H"], cfg.camerad["F"], cfg.camerad["w"], cfg.camerad["h"])
	path = argv[0]
	outpath = argv[1]
	img = np.load(path).astype(np.float32)
	img = cv2.GaussianBlur(img, (11, 11), 0)
	img[:,:,0] /= np.amax(img[:,:,0])
	img[:,:,1] /= np.amax(img[:,:,1])
	img[:,:,2] /= np.amax(img[:,:,2])

	h = img.shape[0]
	w = img.shape[1]
	N = 100

	f = open(os.path.join(outpath, "measure1"), "w")
	y = int(h/2)
	for i in range(N):
		x = int(i / N * (w-1))
		lat, lon = proj.project(y, x)
		cosa = math.cos(lon)*math.cos(lat)
		b = (img[y][x][0] + img[y][x][1] + img[y][x][2])/3
		f.write("%i %i %i %f %f\n" % (i, y, x, cosa, b))
	f.close()
	
	f = open(os.path.join(outpath, "measure2"), "w")
	x = int(w/2)
	for i in range(N):
		y = int(i / N * (h-1))
		lat, lon = proj.project(y, x)
		cosa = math.cos(lon)*math.cos(lat)
		b = (img[y][x][0] + img[y][x][1] + img[y][x][2])/3
		f.write("%i %i %i %f %f\n" % (i, y, x, cosa, b))
	f.close()
	
	f = open(os.path.join(outpath, "measure3"), "w")
	for i in range(N):
		x = int(i / N * (w-1))
		y = int(i / N * (h-1))
		lat, lon = proj.project(y, x)
		cosa = math.cos(lon)*math.cos(lat)
		b = (img[y][x][0] + img[y][x][1] + img[y][x][2])/3
		f.write("%i %i %i %f %f\n" % (i, y, x, cosa, b))
	f.close()

	np.save(os.path.join(outpath, ""), img)

if __name__ == "__main__":
	run(sys.argv[1:])

