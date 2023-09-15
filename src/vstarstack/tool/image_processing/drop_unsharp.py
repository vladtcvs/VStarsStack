import os
import numpy as np
import scipy.ndimage

import vstarstack.library.data
import vstarstack.tool.cfg
import vstarstack.tool.common

def measure_sharpness(img : np.ndarray) -> float:
    sx = scipy.ndimage.sobel(img, axis=0, mode='constant')
    sy = scipy.ndimage.sobel(img, axis=1, mode='constant')
    sobel = np.sqrt(sx**2 + sy**2)
    metric = np.sum(sobel)
    summ = np.sum(img)
    return metric / summ

def measure_sharpness_df(df : vstarstack.library.data.DataFrame) -> float:
    metric = 0
    nch = 0
    for channel in df.get_channels():
        img, opts = df.get_channel(channel)
        if not opts["brightness"]:
            continue
        amax = np.amax(img)
        amin = np.amin(img)
        img = (img - amin)/(amax - amin)
        metric += measure_sharpness(img)
        nch += 1
    if nch == 0:
        return 0
    return metric / nch

def select_sharpests(fnames : list[str], percent : int):
    metrics = []
    for fname in fnames:
        df = vstarstack.library.data.DataFrame.load(fname)
        metric = measure_sharpness_df(df)
        print(f"{fname} : {metric}")
        metrics.append((fname, metric))
    metrics = sorted(metrics, key=lambda item: item[1], reverse=True)
    metrics = metrics[:int(len(metrics)*percent/100)]
    return [item[0] for item in metrics]

def run(project : vstarstack.tool.cfg.Project, argv : list[str]):
    path = argv[0]
    percent = int(argv[1])
    files = vstarstack.tool.common.listfiles(path, ".zip")
    fnames = [item[1] for item in files]
    sharpests = select_sharpests(fnames, percent)
    for i,fname in enumerate(sharpests):
        basename = os.path.basename(fname)
        dirname = os.path.dirname(fname)
        basename = "%06i_%s" % (i, basename)
        os.rename(fname,  os.path.join(dirname, basename))
        fnames.remove(fname)

    for fname in fnames:
        print(f"Removing {fname}")
        os.remove(fname)
