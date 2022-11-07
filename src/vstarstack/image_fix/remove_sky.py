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

import multiprocessing as mp

import vstarstack.sky_model.gradient
import vstarstack.sky_model.gauss
import vstarstack.sky_model.isoline
import vstarstack.sky_model.quadratic

import vstarstack.usage
import os
import vstarstack.common
import vstarstack.cfg
import vstarstack.data

def remove_sky(name, infname, outfname, model):
	print(name)

	img = vstarstack.data.DataFrame.load(infname)

	for channel in img.get_channels():
		
		image, opts = img.get_channel(channel)
		if not opts["brightness"]:
			continue

		print("\t%s" % channel)
		sky = model(image)

		result = image - sky
		img.add_channel(result, channel, **opts)

	img.store(outfname)

def process_file(argv, model):
	infname = argv[0]
	outfname = argv[1]
	name = os.path.splitext(os.path.basename(infname))[0]
	remove_sky(name, infname, outfname, model)

def process_dir(argv, model):
	inpath = argv[0]
	outpath = argv[1]
	files = vstarstack.common.listfiles(inpath, ".zip")
	pool = mp.Pool(vstarstack.cfg.nthreads)
	pool.starmap(remove_sky, [(name, fname, os.path.join(outpath, name + ".zip"), model) for name, fname in files])
	pool.close()

def process(argv, model):
	if len(argv) > 0:
		if os.path.isdir(argv[0]):
			process_dir(argv, model)
		else:
			process_file(argv, model)
	else:
		process_dir([vstarstack.cfg.config["paths"]["npy-fixed"],
						vstarstack.cfg.config["paths"]["npy-fixed"]], model)

commands = {
	"isoline"   : (lambda argv : process(argv, vstarstack.sky_model.isoline.model),  "use isoline model"),
	"gauss"     : (lambda argv : process(argv, vstarstack.sky_model.gauss.model),    "use gauss blur model"),
	"gradient"  : (lambda argv : process(argv, vstarstack.sky_model.gradient.model), "use gradient model"),
	"quadratic" : (lambda argv : process(argv, vstarstack.sky_model.quadratic.model), "use quadratic gradient model"),
}

def run(argv):
	usage.run(argv, "image-fix remove-sky", commands, autohelp=True)
