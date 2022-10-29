import numpy as np
import os
import imageio

import matplotlib.pyplot as plt

import data
import common
import usage

def make_frames(data, channel):
	if channel == "RGB":
		r,_ = data.get_channel("R")
		g,_ = data.get_channel("G")
		b,_ = data.get_channel("B")
		
		rgb = np.zeros((r.shape[0], r.shape[1], 3))
		rgb[:,:,0] = r
		rgb[:,:,1] = g
		rgb[:,:,2] = b
		amax = np.amax(rgb)
		rgb = rgb / amax
		frames = {"RGB" : rgb}

	else:
		frames = {}
		if channel is None:
			channels = data.get_channels()
		else:
			channels = [channel]

		for channel in channels:
			print("Channel = ", channel)
			img,options = data.get_channel(channel)
			print("Shape = ", img.shape)

			if options["brightness"]:
				amin = np.amin(img)
				amax = np.amax(img)
				print("%s: %f - %f" % (channel, amin, amax))
				if amax - amin > 0:
					img = (img - amin)/(amax-amin)
	
			frames[channel] = img
	return frames

def show(argv):
	power = 1
	slope = 1

	path = argv[0]
	if len(argv) > 1:
		channel = argv[1]
	else:
		channel = None

	dataframe = data.DataFrame.load(path)
	frames = make_frames(dataframe, channel)

	nch = len(frames)		
	fig, axs = plt.subplots(1, nch)
	fig.patch.set_facecolor('#222222')

	id = 0
	for channel in frames:
		if nch > 1:
			sp = axs[id]
		else:
			sp = axs
		img = frames[channel]
		img = img / np.amax(img)
		img = img * slope
		img = np.clip(img, 0, 1)
		img = np.power(img, power)
		sp.imshow(img, cmap="gray")
		sp.set_title(channel)
		id += 1

	plt.show()


def convert(argv):
	path = argv[0]
	if len(argv) > 1:
		channel = argv[1]
	else:
		channel = None

	out = argv[2]

	dataframe = data.DataFrame.load(path)
	frames = make_frames(dataframe, channel)

	nch = len(frames)

	out = os.path.abspath(out)
	dir = os.path.dirname(out)
	name, ext = os.path.splitext(os.path.basename(out))
	for channel in frames:
		if nch > 1:
			fname = os.path.join(dir, "%s_%s.%s" % (name, channel, ext))
		else:
			fname = out
		img = frames[channel]
		img = img*65535
		img = np.clip(img, 0, 65535).astype('uint16')
		imageio.imwrite(fname, img)

def cut(argv):
	path = argv[0]
	left = int(argv[1])
	top = int(argv[2])
	right = int(argv[3])
	bottom = int(argv[4])
	out = argv[5]

	dataframe = data.DataFrame.load(path)
	channels = dataframe.get_channels()

	outdata = data.DataFrame(params=dataframe.params)
	for channel in channels:
		print("Channel = ", channel)
		img,options = dataframe.get_channel(channel)
		sub = img[top:bottom+1, left:right+1]
		outdata.add_channel(sub, channel, **options)
	for link_type in dataframe.links:
		for name in dataframe.links[link_type]:
			outdata.add_channel_link(name, dataframe.links[link_type][name], link_type)
	outdata.store(out)

def rename_channel(argv):
	name = argv[0]
	channel = argv[1]
	target = argv[2]
	print(name)
	dataframe = data.DataFrame.load(name)
	dataframe.rename_channel(channel, target)
	dataframe.store(name)

commands = {
	"show"     : (show, "show image"),
	"convert"  : (convert, "convert image"),
	"cut"      : (cut, "cut part of image"),
	"rename-channel" : (rename_channel, "filename.zip original_name target_name - rename channel"),
}

def run(argv):
	usage.run(argv, "image", commands, autohelp=True)
