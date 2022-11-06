import os
import json

def dircheck(name):
	if not os.path.isdir(name):
		os.mkdir(name)

def run(argv):
	if len(argv) > 0:
		dir = argv[0]
	else:
		dir = os.getcwd()

	projf = os.path.join(dir, "project.json")
	with open(projf) as f:
		proj = json.load(f)

	proj["planets"] = {
		"map_resolution" : 360,
		"paths" : {
			"cutted" : "cutted",
			"maps"   : "maps",
		},
	}

	dircheck(dir + '/cutted')
	dircheck(dir + '/maps')

	with open(projf, "w") as f:
		json.dump(proj, f, indent=4, ensure_ascii=False)
