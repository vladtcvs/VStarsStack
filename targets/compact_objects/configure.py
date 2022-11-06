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

	proj["mode"] = "compact_objects"
	proj["compact_objects"] = {
		"threshold" : 200,
		"minPixels" : 200,
		"maxPixels" : 1000,
		"detect_channels" : ["R", "G", "B"],
		"cutsize" : 200,
		"paths" : {
			"descs" : "descs",
			"cutted" : "cutted",
		},
	}

	dircheck(dir + '/descs')
	dircheck(dir + '/cutted')

	with open(projf, "w") as f:
		json.dump(proj, f, indent=4, ensure_ascii=False)
