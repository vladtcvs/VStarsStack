import os
import sys
import json
import math

import common

def build_description(image, num_main):
	main = image["stars"][:num_main]

	for i in range(len(main)):
		star = main[i]
		star["descriptor"] = []
		other = []
		for j in range(len(main)):
			if i != j:
				other.append(main[j])
		for j in range(len(other)-1):
			other1 = other[j]
			dir1 = (other1["y"] - star["y"], other1["x"] - star["x"])
			dist1 = common.length(dir1)
			dir1 = common.norm(dir1)
			size1 = other1["size"]
			for k in range(j+1, len(other)):
				other2 = other[k]
				dir2 = (other2["y"] - star["y"], other2["x"] - star["x"])
				dist2 = common.length(dir2)
				dir2 = common.norm(dir2)
				size2 = other2["size"]
				angle = math.acos(dir1[0]*dir2[0] + dir1[1]*dir2[1])*180/math.pi
				if dist1 < dist2:
					star["descriptor"].append((dist1, size1, dist2, size2, angle))
				else:
					star["descriptor"].append((dist2, size2, dist1, size1, angle))

	image.pop("stars")
	image["main"] = main
	return image


if __name__ == "__main__":

	path=sys.argv[1]
	outpath=sys.argv[2]
	files = common.listfiles(path, ".json")

	for name, filename in files:
		with open(filename) as f:
			image = json.load(f)

		image = build_description(image, 16)

		with open(os.path.join(outpath, name + ".json"), "w") as f:
			json.dump(image, f, indent=4)


