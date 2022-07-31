import math
import numpy as np
import os
import json
import data

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

else:
	pass

_flat = None
def get_flat():
	if _flat is not None:
		return _flat

	if "flat" in telescope:
		vf = os.path.join(cfgdir, telescope["flat"])
	else:
		vf = None

	if vf is not None and os.path.exists(vf):
		_flat = data.data_load(vf)
	else:
		_flat = data.data_create()

	return _flat
