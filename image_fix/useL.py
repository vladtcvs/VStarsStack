import data
import numpy as np

import matplotlib.pyplot as plt

def process(infname, outfname, colors, L):
    dataframe = data.DataFrame.load(infname)
    print("L: %s" % L)
    print("Colors: %s" % (" ".join(colors)))
    images = {}
    opts = {}
    for channel in colors:
        images[channel],opts[channel] = dataframe.get_channel(channel)
        images[channel] = np.clip(images[channel], 0, 1e9)
    L_image_real,_ = dataframe.get_channel(L)
    L_image_real /= np.amax(L_image_real)

    L_image_synth = sum([images[channel] for channel in images])
    L_image_synth /= np.amax(L_image_synth)

    k = L_image_real / L_image_synth
    k[np.where(L_image_synth == 0)] = 0
    for channel in images:
        images[channel] *= k
        dataframe.add_channel(images[channel], channel, **opts[channel])
    dataframe.store(outfname)

def process_file(argv):
    infname = argv[0]
    outfname = argv[1]
    L = argv[2]
    colors = argv[3:]
    process(infname, outfname, colors, L)

def run(argv):
    process_file(argv)
