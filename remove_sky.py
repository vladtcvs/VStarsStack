import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import sys
import cv2


def detect(image):
	if len(image.shape) == 3:
		gray = np.sum(image, axis=2)
	else:
		gray = image

	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	mb = np.amax(blurred)
	blurred = blurred / mb * 255

	percent = 1

	hist = np.histogram(blurred, bins=1024)
	nums = list(hist[0])
	bins = list(hist[1])
	nums.reverse()
	bins.reverse()

	total = sum(nums)
	maxp = total * percent / 100
	summ = 0
	for i in range(1024):
		thr = bins[i]
		c = nums[i]
		summ += c
		if summ >= maxp:
			break
		
	print("Threshold = %f" % thr)

	mask = cv2.threshold(blurred, thr, 1.0, cv2.THRESH_BINARY)[1]
	return mask

def fun(l):
	return 2/(l**2+1) - 1

def interp(y, x, h, w):
	L = ((w/2)**2+(h/2)**2)**0.5
	x -= w/2
	y -= h/2
	l = (x**2+y**2)**0.5
	return fun(l/L)


image = np.load(sys.argv[1])
shape = image.shape
w = shape[1]
h = shape[0]
x1 = int(w/2-w/10)
x2 = int(w/2+w/10)
y1 = int(h/2-h/10)
y2 = int(h/2+h/10)

print("Remove circlular gradient")

center = image[y1:y2, x1:x2]
centeravg = np.mean(np.mean(center,axis=0), axis=0)

x2 = int(w/5)
y2 = int(h/5)
lt = image[0:y2, 0:x2]
ltavg = np.mean(np.mean(lt, axis=0), axis=0)

x1 = int(w-w/5)
y1 = 0
x2 = w
y2 = int(h/5)
rt = image[y1:y2, x1:x2]
rtavg = np.mean(np.mean(rt, axis=0), axis=0)

corneravg = (ltavg + rtavg)/2

print(centeravg, corneravg)

sky = np.zeros(shape)
for y in range(h):
	for x in range(w):
		f = interp(y, x, h, w)
		c = corneravg*(1-f) + centeravg*f
		sky[y][x] = c


image = image - sky

print("Remove vertical gradient")

x1 = 0
y1 = 0
x2 = int(w/5)
y2 = int(h/5)
lt = image[0:y2, 0:x2]
ltavg = np.mean(np.mean(lt, axis=0), axis=0)

x1 = 0
y1 = int(h-h/5)
x2 = int(w/5)
y2 = h
lb = image[y1:y2, x1:x2]
lbavg = np.mean(np.mean(lb, axis=0), axis=0)

sky = np.zeros(shape)
for y in range(h):
	for x in range(w):
		f = y / h
		c = ltavg*(1-f) + lbavg*f
		sky[y][x] = c

image = image - sky

print("Remove horizontal gradient")

x1 = 0
y1 = 0
x2 = int(w/5)
y2 = int(h/5)
lt = image[y1:y2, x1:x2]
ltavg = np.mean(np.mean(lt, axis=0), axis=0)

x1 = w-int(w/5)
y1 = 0
x2 = w
y2 = int(h/5)
rt = image[y1:y2, x1:x2]
rtavg = np.mean(np.mean(rt, axis=0), axis=0)

sky = np.zeros(shape)
for y in range(h):
	for x in range(w):
		f = x / w
		c = ltavg*(1-f) + rtavg*f
		sky[y][x] = c

image = image - sky

print("Remove sky residuals")

mask = detect(image)
sky = np.array(image)
for y in range(image.shape[0]):
	for x in range(image.shape[1]):
		if mask[y][x] == 1:
			sky[y][x] = np.array([0, 0, 0])

sky = cv2.GaussianBlur(sky, (2*int(h/30)+1, 2*int(w/30)+1), 0)
image = image - sky

np.save(sys.argv[2], image)

