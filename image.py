import numpy as np
import os
import imageio

import matplotlib.pyplot as plt

import common
import usage

channel_light = {
	"Yb" : False,
	"Yr" : False,
}

def make_frames(data, channel):
	if channel == "RGB":
		r = data["channels"]["R"]
		g = data["channels"]["G"]
		b = data["channels"]["B"]

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
			channels = data["meta"]["channels"]
		else:
			channels = [channel]

		for channel in channels:
			print("Channel = ", channel)
			img = data["channels"][channel]
			print("Shape = ", img.shape)

			if channel in channel_light and channel_light[channel] == False:
				amin = np.amin(img)
				amax = np.amax(img)
				print("%s: %f - %f" % (channel, amin, amax))
				img = (img - amin)/(amax-amin)
			else:
				amax = np.amax(img)
				print("%s: 0 - %f" % (channel, amax))
				img = img / amax

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

	data = common.data_load(path)
	frames = make_frames(data, channel)

	nch = len(frames)		
	fig, axs = plt.subplots(1, nch)
	fig.patch.set_facecolor('#222222')

	id = 0
	for channel in frames:
		if nch > 1:
			sp = axs[id]
		else:
			sp = axs
		sp.imshow(frames[channel], cmap="gray")
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

	data = common.data_load(path)
	frames = make_frames(data, channel)

	nch = len(frames)		
	
	images = {}
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
	data = common.data_load(path)
	channels = data["meta"]["channels"]

	outdata = common.data_create(data["meta"]["tags"], data["meta"]["params"])
	for channel in channels:
		print("Channel = ", channel)
		img = data["channels"][channel]
		sub = img[top:bottom+1, left:right+1]
		encoded = channel in data["meta"]["encoded_channels"]
		common.data_add_channel(outdata, sub, channel, encoded)
	common.data_store(outdata, out)

commands = {
	"show"     : (show, "show image"),
	"convert"  : (convert, "convert image"),
	"cut"      : (cut, "cut part of image"),
}

def run(argv):
	usage.run(argv, "image", commands, autohelp=True)
