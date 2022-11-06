import math
import numpy as np
import os
import json
import data
import multiprocessing as mp

def getval(config, name, default):
	if name in config:
		return config[name]
	return default

cfgdir = os.getcwd()
cfgpath = os.path.join(cfgdir, "project.json")
nthreads = max(int(mp.cpu_count())-1, 1)

if os.path.exists(cfgpath):
	with open(cfgpath) as f:
		config = json.load(f)

	use_sphere = getval(config, "use_sphere", True)
	compress = getval(config, "compress", True)

	if "stars" in config:
		stars      = config["stars"]
		use_angles = getval(stars, "use_angles", True)

	telescope = config["telescope"]

	camerad    = telescope["camera"]
	scope      = telescope["scope"]

	if "distorsion" in telescope:
		distorsion = telescope["distorsion"]
	else:
		distorsion = None

else:
	pass
