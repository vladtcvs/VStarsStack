import os
import sys
import common
import math

import multiprocessing as mp
nps = max(int(mp.cpu_count())-1, 1)

minsize = nps

orig=sys.argv[1]
base=sys.argv[2]

files = common.listfiles(orig, ".nef")

num = len(files)
splits = math.ceil(num**(2/3)/2**(1/3))
inblock = max(math.ceil(num / splits), minsize)

blocks=[]
block=[]

for name, filename in files:
	block.append(filename)
	if len(block) >= inblock:
		blocks.append(block)
		block=[]

if len(block) > 0:
	blocks.append(block)

for i in range(len(blocks)):
	blockdir = os.path.join(base, "block%04i" % i)
	origdir = os.path.join(blockdir, "orig")
	os.mkdir(blockdir)
	os.mkdir(origdir)
	os.mkdir(os.path.join(blockdir, "npy"))
	os.mkdir(os.path.join(blockdir, "stars"))
	os.mkdir(os.path.join(blockdir, "descs"))
	os.mkdir(os.path.join(blockdir, "shifted"))
	for filename in blocks[i]:
		basename = os.path.basename(filename)
		os.rename(filename, os.path.join(origdir, basename))

sumdir = os.path.join(base, "sum")
os.mkdir(sumdir)

os.mkdir(os.path.join(sumdir, "npy"))
os.mkdir(os.path.join(sumdir, "stars"))
os.mkdir(os.path.join(sumdir, "descs"))
os.mkdir(os.path.join(sumdir, "shifted"))

