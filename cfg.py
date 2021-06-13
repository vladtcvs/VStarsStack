import math
import numpy as np
import os
import json

cfgdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

with open(os.path.join(cfgdir, "config.json")) as f:
	config = json.load(f)

use_sphere = config["use_sphere"]
stars      = config["stars"]

with open(os.path.join(cfgdir, config["telescope"])) as f:
	telescope = json.load(f)

camerad    = telescope["camera"]
distorsion = telescope["distorsion"]
vignetting = np.load(os.path.join(cfgdir, telescope["vignetting"]))["arr_0"]

