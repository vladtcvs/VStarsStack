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

import vstarstack.fine_shift.image_wave
import vstarstack.usage
import vstarstack.cfg
import vstarstack.data
import vstarstack.common

import multiprocessing as mp
import numpy as np
import os
import json
#import tqdm

ncpu = vstarstack.cfg.nthreads

def cluster_average(cluster):
    xs = []
    ys = []
    for name in cluster:
        xs.append(cluster[name]["x"])
        ys.append(cluster[name]["y"])
    pos = {
        "x" : sum(xs)/len(xs),
        "y" : sum(ys)/len(ys),
    }
    return pos

def process_alignment(name, outpath, Nsteps, plen, dh, W, H, gridW, gridH, good_clusters):
    print("Processing: %s" % name)
    wave = vstarstack.fine_shift.image_wave.ImageWave(W, H, gridW, gridH)
    points = []
    targets = []
    for cluster in good_clusters:
        if name not in cluster["images"]:
            continue

        # we need reverse transformation
        x = cluster["average"]["x"]
        y = cluster["average"]["y"]
        points.append((x,y))
        x = cluster["images"][name]["x"]
        y = cluster["images"][name]["y"]
        targets.append((x,y))

    print("\tusing %i points" % len(points))
    if len(points) < plen:
        print("\tskip - too low points")
        return
    wave.approximate(targets, points, Nsteps, dh)
    data = wave.data()

    out = os.path.join(outpath, name+".json")
    if os.path.isfile(out):
        with open(out) as f:
            descriptor = json.load(f)
    else:
        descriptor = {}

    descriptor["fine_shift"] = {}
    descriptor["fine_shift"]["image_wave"] = data

    with open(out, "w") as f:
        json.dump(descriptor, f, indent=4, ensure_ascii=False)

def process_alignment_wrapper(arg):
    return process_alignment(*arg)

def find_alignment(argv):
    clusters = argv[0]
    outpath = argv[1]

    with open(clusters) as f:
        clusters = json.load(f)

    W = vstarstack.cfg.camerad["w"]
    H = vstarstack.cfg.camerad["h"]

    Nsteps = vstarstack.cfg.config["fine_shift"]["Nsteps"]
    dh = vstarstack.cfg.config["fine_shift"]["dh"]
    gridW = vstarstack.cfg.config["fine_shift"]["gridW"]
    gridH = vstarstack.cfg.config["fine_shift"]["gridH"]
    cllen = vstarstack.cfg.config["fine_shift"]["cluster_len_k"]
    plen = vstarstack.cfg.config["fine_shift"]["points_min_len"]

    maxcllen = max([len(cluster) for cluster in clusters])    
    names = []
    good_clusters = []
    for cluster in clusters:
        if len(cluster.keys()) < maxcllen * cllen:
            continue

        names += list(cluster.keys())
        gcluster = {
            "average" : cluster_average(cluster),
            "images" : cluster,
        }
        good_clusters.append(gcluster)
    names = sorted(list(set(names)))

    pool = mp.Pool(ncpu)
    args = [(name, outpath, Nsteps, plen, dh, W, H, gridW, gridH, good_clusters) for name in names]
    for _ in pool.imap_unordered(process_alignment_wrapper, args):
        pass
    pool.close()

def apply_alignment_file(name, npy, align_data, output):
    print(name)
    if not os.path.exists(align_data):
        return
    with open(align_data) as f:
        descriptor = json.load(f)
        align_data = descriptor["fine_shift"]["image_wave"]

    wave = vstarstack.fine_shift.image_wave.ImageWave.from_data(align_data)
    dataframe = vstarstack.data.DataFrame.load(npy)
    links = dict(dataframe.links)
    for channel in dataframe.get_channels():
        image, opts = dataframe.get_channel(channel)
        if opts["encoded"]:
            continue
        fixed = np.zeros(image.shape)
        for y in range(fixed.shape[0]):
            for x in range(fixed.shape[1]):
                ox,oy = wave.interpolate(x,y)
                fixed[y,x] = vstarstack.common.getpixel(image, oy, ox)[1]

        dataframe.add_channel(fixed, channel, **opts)
    dataframe.links = links
    dataframe.store(output)

def apply_alignment_file_wrapper(arg):
    apply_alignment_file(*arg)

def apply_alignment(argv):
    if len(argv) >= 3:
        npys = argv[0]
        descs = argv[1]
        outputs = argv[2]
    else:
        npys = vstarstack.cfg.config["npy-fixed"]
        descs = vstarstack.cfg.config["descs"]
        outputs = vstarstack.cfg.config["aligned"]

    if os.path.isdir(npys):
        files = vstarstack.common.listfiles(npys, ".zip")
        pool = mp.Pool(ncpu)
        args = [(name, fname, os.path.join(descs, name + ".json"), os.path.join(outputs, name + ".zip"))
                    for name, fname in files]
        for _ in pool.imap_unordered(apply_alignment_file_wrapper, args):
            pass
        pool.close()
    else:
        apply_alignment_file(os.path.basename(npys), npys, descs, outputs)

commands = {
    "align-features" : (find_alignment, "find alignment of images", "clusters.json alignments/"),
    "apply-aligns" : (apply_alignment, "apply alignments to images", "npys/ alignments/ output/"),
}

def run(argv):
    vstarstack.usage.run(argv, "fine-shift", commands, autohelp=True)
