from calendar import c
import data
import numpy as np

import matplotlib.pyplot as plt

def has_flag(opts, name):
    if name not in opts:
        return False
    if not opts[name]:
        return False
    return True

def process(infname, outfname, colors, L):
    dataframe = data.DataFrame.load(infname)
    print("L: %s" % L)
    print("Colors: %s" % (" ".join(colors)))
    images = {}
    opts = {}
    L_image_synth = None

    for channel in colors:
        image,opts[channel] = dataframe.get_channel(channel)
        image = np.clip(image, 0, 1e12)
        images[channel] = image

        if not has_flag(opts[channel], "normalized"):
            weight,_ = dataframe.get_channel(dataframe.links["weight"][channel])
            image /= weight
            image[np.where(weight == 0)] = 0

        if L_image_synth is None:
            L_image_synth = image
        else:
            L_image_synth += image

    L_image_synth /= np.amax(L_image_synth)
    
    L_image_real,_ = dataframe.get_channel(L)
    L_image_real /= np.amax(L_image_real)

    k = L_image_real / L_image_synth
    k[np.where(L_image_synth == 0)] = 0
    for channel in images:
        dataframe.add_channel(images[channel]*k, channel, **opts[channel])
    dataframe.store(outfname)

def process_file(argv):
    infname = argv[0]
    outfname = argv[1]
    L = argv[2]
    colors = argv[3:]
    process(infname, outfname, colors, L)

def run(argv):
    process_file(argv)
