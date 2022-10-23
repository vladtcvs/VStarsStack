import common
import data

import stars
import skimage.restoration
import sys

import matplotlib.pyplot as plt
import numpy as np

def build_psf(image, rect):
    image = image[rect[1]:rect[3], rect[0]:rect[2]]
    image = np.array(image)
    low = np.where(image < np.amax(image)*0.7)
    image[low] = 0
    image = image / np.sum(image)
    return image

def fix(image, star_rect):
    niterations = 20
    psf = build_psf(image, star_rect)
    return skimage.restoration.richardson_lucy(image, psf, niterations)

def process_file(argv):
    fname = argv[0]
    outname = argv[1]
    rect = (int(argv[2]), int(argv[3]), int(argv[4]), int(argv[5]))
    dataframe = data.DataFrame.load(fname)
    channels = dataframe.get_channels()
    for channel in channels:
        img, options = dataframe.get_channel(channel)
        if options["encoded"]:
            continue
        if options["weight"]:
            continue
        img = img / np.amax(img)
        fixed = fix(img, rect)
        dataframe.add_channel(fixed, channel, **options)
    dataframe.store(outname)

def run(argv):
    process_file(argv)
