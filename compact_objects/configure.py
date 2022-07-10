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

	proj["mode"] = "compact objects"
	proj["compact objects"] = {
		"paths" : {
			"descs" : "descs",
			"crop"  : "crop",
		},
	}

	with open(projf, "w") as f:
		json.dump(proj, f, indent=4, ensure_ascii=False)
