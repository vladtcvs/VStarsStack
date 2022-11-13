import vstarstack.fine_shift.image_wave
import vstarstack.usage

import json

def align_features(argv):
    images = argv[0]
    clusters = argv[1]
    outpath = argv[2]

    with open(clusters) as f:
        clusters = json.load(f)

    

commands = {
    "align-features" : (align_features, "align features on images", "npy/ clusters.json shifted/"),
}

def run(argv):
    vstarstack.usage.run(argv, "fine-shift", commands, autohelp=True)
