import math
import numpy as np
import os
import json

def getval(config, name, default):
	if name in config:
		return config[name]
	return default

cfgdir = os.getcwd()
cfgpath = os.path.join(cfgdir, "project.json")

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

	if "distorsion" in telescope:
		distorsion = telescope["distorsion"]
	else:
		distorsion = None

	if "vignetting" in telescope:
		vf = os.path.join(cfgdir, telescope["vignetting"])
	else:
		vf = None

	if vf is not None and os.path.exists(vf):
		vignetting = np.load(vf)["arr_0"]
	else:
		vignetting = np.zeros((camerad["h"], camerad["w"]))+1
else:
	pass

