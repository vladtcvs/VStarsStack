import os
import sys
import json
import math

import common
import multiprocessing as mp
ncpu = mp.cpu_count()

thr_num = 0.4
thr_val = 6

def difference(val1, val2):
	dist1 = abs(val1[0] - val2[0])**2
	size1 = abs(val1[1] - val2[1])**2
	dist2 = abs(val1[2] - val2[2])**2
	size2 = abs(val1[3] - val2[3])**2
	angle = abs(val1[4] - val2[4])**2
	return (dist1 + size1 + dist2 + size2 + angle*16)/5

def match(star1, star2, thrnum, thrval):
	d1 = star1["descriptor"]
	d2 = star2["descriptor"]
	mismatches = 0.0
	matches = 0.0
	sum = 0
	for val1 in d1:
		for val2 in d2:
			d = difference(val1, val2)
			if d < thrval**2:
				matches += 1
				sum += d
				
		else:
			mismatches += 1
	if matches / (matches + mismatches) > thrnum:
		return True, (sum / matches)**0.5
	else:
		return False, None


def build_match(image1, image2, name2):
	main1 = image1["main"]
	main2 = image2["main"]
	# match stars of main1 to stars of main2
	for i in range(len(main1)):
		star1 = main1[i]
		if "matches" not in star1:
			star1["matches"] = {}
		star1["matches"][name2] = None
		minj = None
		mins = None
		for j in range(len(main2)):
			star2 = main2[j]
			m, s = match(star1, star2, thr_num, thr_val)
			if m == False:
				continue
			if mins is None or s < mins:
				mins = s
				minj = j
		star1["matches"][name2] = minj
	image1["main"] = main1
	return image1


def matchStars(image, starsfiles, lock):
	name = image[0]
	filename = image[1]
	print(name)
	lock.acquire()
	with open(filename) as f:
		stars = json.load(f)
	lock.release()
	for name0, _ in starsfiles:
		print("Base: ", name0)
		starsfn0 = os.path.join(starsdir, name0 + ".json")
		lock.acquire()
		with open(starsfn0) as f:
			stars0 = json.load(f)
		lock.release()
		stars = build_match(stars, stars0, name0)

	lock.acquire()
	with open(filename, "w") as f:
		json.dump(stars, f, indent=4)
	lock.release()


if __name__ == "__main__":
	starsdir = sys.argv[1]

	filelock = mp.Manager().Lock()
	
	shots = {}

	starsfiles = common.listfiles(starsdir, ".json")

	pool = mp.Pool(ncpu)
	pool.starmap(matchStars, [(image, starsfiles, filelock) for image in starsfiles])
	pool.close()

