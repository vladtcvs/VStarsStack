import os
import json

def run(argv):
	if len(argv) > 0:
		dir = argv[0]
	else:
		dir = os.getcwd()

	projf = os.path.join(dir, "project.json")
	with open(projf) as f:
		proj = json.load(f)

	proj["mode"] = "stars"
	proj["stars"] = {
		"describe": {
			"num_main" : 20,
			"mindist" : 0.1,
		},
		"match" : {
			"max_angle_diff" : 0.01,
            "max_size_diff" : 0.1,
            "max_dangle_diff" : 4,
            "min_matched_ditems" : 15,
		},
		"paths" : {
			"stars" : "stars",
			"descs" : "descs",
			"net"   : "net.json",
		},
		"use_angles" : True,
		"brightness_over_neighbours" : 0.04,
	}
	proj["cluster"] = {
		"path" : "clusters.json"
	}

	with open(projf, "w") as f:
		json.dump(proj, f, indent=4, ensure_ascii=False)

