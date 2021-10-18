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
		"match" : {
			"threshold_value": 0.5,
			"threshold_num": 0.3,
		},
		"paths" : {
			"stars" : "stars",
			"descs" : "descs",
			"net"   : "net.json",
		},
		"use_angles" : True,
	}
	proj["cluster"] = {
		"path" : "clusters.json"
	}

	with open(projf, "w") as f:
		json.dump(proj, f, indent=4, ensure_ascii=False)

