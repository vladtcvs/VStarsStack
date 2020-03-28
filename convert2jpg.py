import common
import imageio
import sys
import os
import rawpy

files = common.listfiles(sys.argv[1], ".nef")
for name, filename in files:
	print(name)
	with rawpy.imread(filename) as raw:
		rgb = raw.postprocess()
		
	imageio.imsave(os.path.join(sys.argv[2], name + ".jpg"), rgb)

