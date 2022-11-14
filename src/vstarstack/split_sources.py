#
# Copyright (c) 2022 Vladislav Tsendrovskii
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

import os
import sys
import math

import vstarstack.common

import multiprocessing as mp
nps = max(int(mp.cpu_count())-1, 1)

minsize = nps

orig=sys.argv[1]
base=sys.argv[2]

files = vstarstack.common.listfiles(orig, ".npz")

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
	npydir = os.path.join(blockdir, "npy")
	os.mkdir(blockdir)
	os.mkdir(os.path.join(blockdir, "npy"))
	os.mkdir(os.path.join(blockdir, "stars"))
	os.mkdir(os.path.join(blockdir, "descs"))
	os.mkdir(os.path.join(blockdir, "aligned"))
	for filename in blocks[i]:
		basename = os.path.basename(filename)
		os.rename(filename, os.path.join(npydir, basename))

sumdir = os.path.join(base, "sum")
os.mkdir(sumdir)

os.mkdir(os.path.join(sumdir, "npy"))
os.mkdir(os.path.join(sumdir, "stars"))
os.mkdir(os.path.join(sumdir, "descs"))
os.mkdir(os.path.join(sumdir, "aligned"))
