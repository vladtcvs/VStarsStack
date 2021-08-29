import math
import numpy as np
import os
import json

#cfgdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
cfgdir = os.path.join(os.getcwd(), "config")
cfgpath = os.path.join(cfgdir, "config.json")

if os.path.exists(cfgpath):
	with open(cfgpath) as f:
		config = json.load(f)

	use_sphere = config["use_sphere"]
	stars      = config["stars"]

	with open(os.path.join(cfgdir, config["telescope"])) as f:
		telescope = json.load(f)

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

