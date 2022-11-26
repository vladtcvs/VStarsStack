#
# Copyright (c) 2022 Vladislav Tsendrovskii
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

import json
import math
import vstarstack.cfg
import vstarstack.common
import multiprocessing as mp

import vstarstack.usage

ncpu = max(int(mp.cpu_count())-1, 1)
#ncpu = 10

debug = False

def items_compare(desc_item1, desc_item2, max_angle_diff, max_dangle_diff, max_size_diff):
	angle1_1 = desc_item1[1]
	size1_1  = desc_item1[2]
	angle2_1 = desc_item1[4]
	size2_1  = desc_item1[5]
	dangle_1 = desc_item1[6]

	angle1_2 = desc_item2[1]
	size1_2  = desc_item2[2]
	angle2_2 = desc_item2[4]
	size2_2  = desc_item2[5]
	dangle_2 = desc_item2[6]

	dangle1 = abs(angle1_1 - angle1_2) / max_angle_diff
	if dangle1 > 1:
		return False, None

	dangle2 = abs(angle2_1 - angle2_2) / max_angle_diff
	if dangle2 > 1:
		return False, None

	dsize1 = abs(size1_1 - size1_2) / max_size_diff
	if dsize1 > 1:
		return False, None
	
	dsize2 = abs(size2_1 - size2_2) / max_size_diff
	if dsize2 > 1:
		return False, None
	
	ddangle = abs(dangle_1 - dangle_2) / max_dangle_diff
	if ddangle > 1:
		return False, None
	
	return True, (dangle1 + dangle2 + dsize1 + dsize2 + ddangle)

def best_ditem_match(ditem, descriptor, used, max_angle_diff, max_dangle_diff, max_size_diff):
	mind = None
	mini = None
	for i in range(len(descriptor)):
		if i in used:
			continue
		ditem2 = descriptor[i]
		matched, d = items_compare(ditem, ditem2, max_angle_diff, max_dangle_diff, max_size_diff)
		if not matched:
			continue
		if (mind is None) or (d > mind):
			mind = d
			mini = i

	return mini, mind

def descriptor_match(desc1,
					 desc2,
					 max_angle_diff,
					 max_dangle_diff,
					 max_size_diff,
					 min_matched_ditems):
	count_matched = 0
	used = []
	for ditem in desc1:
		matched_i, _ = best_ditem_match(ditem, desc2, used, max_angle_diff, max_dangle_diff, max_size_diff)
		if matched_i is not None:
			count_matched += 1
			used.append(matched_i)

		if count_matched >= min_matched_ditems:
			return True

	if count_matched >= len(desc1):
		return True
	if count_matched >= len(desc2):
		return True
	return False

def find_star_match(star,
					stars,
					used,
					max_angle_diff,
					max_dangle_diff,
					max_size_diff,
					min_matched_ditems):
	desc1 = star["descriptor"]
	for i in range(len(stars)):
		if i in used:
			continue
		star2 = stars[i]
		desc2 = star2["descriptor"]
		matched = descriptor_match(desc1, desc2, max_angle_diff, max_dangle_diff, max_size_diff, min_matched_ditems)
		if matched:
			return i
	return None

def build_match(image1,
				image2,
				name2,
				max_angle_diff,
				max_dangle_diff,
				max_size_diff,
				min_matched_ditems):
	main1 = image1["main"]
	main2 = image2["main"]

	used = []
	# match stars of main1 to stars of main2
	for i in range(len(main1)):
		star = main1[i]
		if debug:
			print(i, star["y"], star["x"])
		if "matches" not in star:
			star["matches"] = {}
		matched = find_star_match(star, main2, used, max_angle_diff, max_dangle_diff, max_size_diff, min_matched_ditems)
		star["matches"][name2] = matched
		if matched is not None:
			used.append(matched)
	image1["main"] = main1
	return image1

def matchStars(description, descriptions):
	name = description[0]
	desc = dict(description[1])
	print(name)
	if desc["projection"] == "perspective":
		fov = 2*math.atan((desc["H"]**2 + desc["W"]**2)**0.5/2 / desc["F"])

	max_angle_diff = vstarstack.cfg.stars["match"]["max_angle_diff"] * fov
	max_dangle_diff = vstarstack.cfg.stars["match"]["max_dangle_diff"] * math.pi/180
	max_size_diff = vstarstack.cfg.stars["match"]["max_size_diff"]
	min_matched_ditems = vstarstack.cfg.stars["match"]["min_matched_ditems"]
	
	for name_i, desc_i in descriptions:
		if debug:
			print("%s / %s" % (name, name_i))
			print("\n")
		desc = build_match(desc, desc_i, name_i, max_angle_diff, max_dangle_diff, max_size_diff, min_matched_ditems)
	return name, desc

def process(argv):
	if len(argv) >= 1:
		starsdir = argv[0]
	else:
		starsdir = vstarstack.cfg.config["stars"]["paths"]["descs"]

	filelock = mp.Manager().Lock()
	
	shots = {}

	starsfiles = vstarstack.common.listfiles(starsdir, ".json")
	descs = []
	name_fname = {}
	for name, fname in starsfiles:
		with open(fname) as f:
			desc = json.load(f)
		descs.append((name, desc))
		name_fname[name] = fname

	total = len(starsfiles)**2
	print("total = %i" % total)
	pool = mp.Pool(ncpu)
	results = pool.starmap(matchStars, [(desc, descs) for desc in descs])
	for name, desc in results:
		with open(name_fname[name], "w") as f:
			json.dump(desc, f, indent=4, ensure_ascii=False)
	pool.close()

def run(argv):
	process(argv)

