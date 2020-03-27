import sys
import common
import random
import os

n=3000
files = common.listfiles(sys.argv[1], ".png")
sample = random.sample(files, n)
toremove = []
for item in files:
	if item in sample:
		continue
	toremove.append(item[1])

for f in toremove:
	os.remove(f)

