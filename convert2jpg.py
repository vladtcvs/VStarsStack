import common
import imageio
import sys
import os
import rawpy
import numpy as np

files = common.listfiles(sys.argv[1], ".nef")
for name, filename in files:
	print(name)
	options = {
		"half_size" : True,
		"four_color_rgb" : False,
		"use_camera_wb" : False,
                "use_auto_wb" : False,
		"user_wb" : (1,1,1,1),
		"user_flip" : 0,
		"output_color" : rawpy.ColorSpace.raw,
		"output_bps" : 16,
		"user_black" : None,
		"user_sat" : None,
		"no_auto_bright" : True,
		"auto_bright_thr" : 0.0,
		"adjust_maximum_thr" : 0,
		"bright" : 100.0,
		"highlight_mode" : rawpy.HighlightMode.Ignore,
		"exp_shift" : None,
		"exp_preserve_highlights" : 0.0,
		"no_auto_scale" : True,
		"gamma" : (1, 1),
		"chromatic_aberration" : None,
		"bad_pixels_path" : None
	}

	image = rawpy.imread(filename)
	rgb = image.postprocess(**options)

	am = np.amax(rgb)
	rgb = (rgb / am * 255).astype(np.uint8)
		
	imageio.imsave(os.path.join(sys.argv[2], name + ".jpg"), rgb)

