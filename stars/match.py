import os
import sys
import json
import math
import cfg
import common
import multiprocessing as mp

import usage

ncpu = max(int(mp.cpu_count())-1, 1)
#ncpu = 10

debug = False

thr_num = cfg.stars["match"]["threshold_num"]
thr_val = cfg.stars["match"]["threshold_value"]

def ditem_difference(val1, val2):
	angle1_1 = val1[1]
	size1_1  = val1[2]
	angle2_1 = val1[4]
	size2_1  = val1[5]
	dangle_1 = val1[6]

	angle1_2 = val2[1]
	size1_2  = val2[2]
	angle2_2 = val2[4]
	size2_2  = val2[5]
	dangle_2 = val2[6]

	da1 = (angle1_1 - angle1_2)**2
	ds1 = (size1_1 - size1_2)**2
	da2 = (angle2_1 - angle2_2)**2
	ds2 = (size2_1 - size2_2)**2
	dda = (dangle_1 - dangle_2)**2

	return da1 + ds1 + da2 + ds2 + dda

def best_ditem_match(ditem, descriptor):
	mind = None
	mini = None
	for i in range(len(descriptor)):
		val2 = descriptor[i]
		d = ditem_difference(ditem, val2)
		if mind is not None and d > mind:
			continue
		mind = d
		mini = i
	return mini, mind

def star_difference(star1, star2):
	class FoundMatchException(Exception):
		pass

	d1 = list(star1["descriptor"])
	d2 = list(star2["descriptor"])
	errs = []
	has_new_match = None
	while has_new_match != False:
		has_new_match = False
		try:
			for i in range(len(d1)):
				val1 = d1[i]
				minj, mind = best_ditem_match(val1, d2)
				if mind is not None:
					errs.append(mind)
					d1.pop(i)
					d2.pop(minj)
					raise FoundMatchException()
		except FoundMatchException:
			has_new_match = True
	return errs

def best_star_match(star, stars, thr_val, thr_num):
	min_i = None
	min_sum_small_err = None
	max_len_small_err = None
	for i in range(len(stars)):
		#print("Compare to star", i)
		star2 = stars[i]
		errs = star_difference(star, star2)
#		print(errs)
		small_errs = []
		for err in errs:
			if err < thr_val:
				small_errs.append(err)
		sum_small_err = sum(small_errs)
		len_small_err = len(small_errs)
#		print(len_small_err, sum_small_err)
		if len_small_err < thr_num * len(errs):
			continue

		if max_len_small_err is not None and len_small_err < max_len_small_err:
			continue

		if debug:
			print(len_small_err, sum_small_err)

		if max_len_small_err is None or max_len_small_err < len_small_err:
			min_sum_small_err = None
			max_len_small_err = len_small_err

		if min_sum_small_err is None or sum_small_err < min_sum_small_err:
			min_sum_small_err = sum_small_err
			min_i = i
	if debug:
		if min_i is not None:
			print("Result: ", min_i, stars[min_i]["y"], stars[min_i]["x"])
		else:
			print("Result: None")
	return min_i


def build_match(image1, image2, name2, thr_val, thr_num):
	main1 = image1["main"]
	main2 = image2["main"]
	# match stars of main1 to stars of main2
	for i in range(len(main1)):
		star = main1[i]
		if debug:
			print(i, star["y"], star["x"])
		if "matches" not in star:
			star["matches"] = {}
		best = best_star_match(star, main2, thr_val, thr_num)
		star["matches"][name2] = best
	image1["main"] = main1
	return image1

def matchStars(image, starsfiles, starsdir, lock):
	name = image[0]
	filename = image[1]
	lock.acquire()
	with open(filename) as f:
		stars = json.load(f)
	lock.release()
	for name0, _ in starsfiles:
#		print("%s / %s" % (name, name0))
		if debug:
			print("\n")
		starsfn0 = os.path.join(starsdir, name0 + ".json")
		lock.acquire()
		with open(starsfn0) as f:
			stars0 = json.load(f)
		print("Images: %s / %s" % (name, name0))
		lock.release()
		stars = build_match(stars, stars0, name0, thr_val, thr_num)

	lock.acquire()
	with open(filename, "w") as f:
		json.dump(stars, f, indent=4)
	lock.release()

def process(argv):
	starsdir = argv[0]

	filelock = mp.Manager().Lock()
	
	shots = {}

	starsfiles = common.listfiles(starsdir, ".json")
	total = len(starsfiles)**2
	print("total = %i" % total)
	pool = mp.Pool(ncpu)
	pool.starmap(matchStars, [(image, starsfiles, starsdir, filelock) for image in starsfiles])
	pool.close()

commands = {
	"*" : (process, "find stars matches", "descs_dir/"),
}

def run(argv):
	usage.run(argv, "stars match", commands)

