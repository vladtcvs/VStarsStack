#
# Copyright (c) 2024 Vladislav Tsendrovskii
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

import logging
import os
import numpy as np
import scipy.ndimage
from enum import Enum
import cv2

import vstarstack.library.data
import vstarstack.tool.cfg
import vstarstack.tool.common

logger = logging.getLogger(__name__)

class EstimationMethod(Enum):
    """Sharpness estimation method"""
    SOBEL = 0
    LAPLACE = 1

def measure_sharpness_sobel(img : np.ndarray, mask : np.ndarray | None) -> float:
    sx = scipy.ndimage.sobel(img, axis=0, mode='constant')
    sy = scipy.ndimage.sobel(img, axis=1, mode='constant')
    sobel = np.sqrt(sx**2 + sy**2)
    if mask is not None:
        sobel = sobel * mask
        img = img * mask
    metric = np.sum(sobel)
    summ = np.sum(img)
    return metric / summ

def measure_sharpness_laplace(img : np.ndarray, mask : np.ndarray | None) -> float:
    laplace = cv2.Laplacian(img, cv2.CV_64F).var()
    if mask is not None:
        laplace = laplace * mask
        img = img * mask
    metric = np.sum(laplace)
    summ = np.sum(img)
    return metric / summ

def measure_sharpness_df(df : vstarstack.library.data.DataFrame, method : EstimationMethod) -> float:
    metric = 0
    nch = 0
    if method not in [EstimationMethod.LAPLACE, EstimationMethod.SOBEL]:
        logger.error(f"Invalid method {method}")
        return None

    for channel in df.get_channels():
        img, opts = df.get_channel(channel)
        if not df.get_channel_option(channel, "brightness"):
            continue
        mask, _, _ = df.get_linked_channel(channel, "mask")
        if method == EstimationMethod.SOBEL:
            metric += measure_sharpness_sobel(img, mask)
        elif method == EstimationMethod.LAPLACE:
            metric += measure_sharpness_laplace(img, mask)
        nch += 1
    if nch == 0:
        return 0
    return metric / nch

def select_sharpests(fnames : list[str], percent : int, method : EstimationMethod):
    metrics = []
    for fname in fnames:
        df = vstarstack.library.data.DataFrame.load(fname)
        metric = measure_sharpness_df(df, method)
        if metric is None:
            continue
        logger.info(f"{fname} : {metric}")
        metrics.append((fname, metric))
    metrics = sorted(metrics, key=lambda item: item[1], reverse=True)
    metrics = metrics[:int(len(metrics)*percent/100)]
    return [item[0] for item in metrics]

def _process(project : vstarstack.tool.cfg.Project, argv : list[str], method : EstimationMethod):
    path = argv[0]
    percent = int(argv[1])
    files = vstarstack.tool.common.listfiles(path, ".zip")
    fnames = [item[1] for item in files]
    sharpests = select_sharpests(fnames, percent, method)
    for i,fname in enumerate(sharpests):
        basename = os.path.basename(fname)
        dirname = os.path.dirname(fname)
        basename = "%06i_%s" % (i, basename)
        os.rename(fname,  os.path.join(dirname, basename))
        fnames.remove(fname)

    for fname in fnames:
        logger.info(f"Removing {fname}")
        os.remove(fname)

commands = {
    "sobel" : (lambda project, argv : _process(project, argv, EstimationMethod.SOBEL), "Use Sobel filter for estimating sharpness", "path/ percent"),
    "laplace" : (lambda project, argv : _process(project, argv, EstimationMethod.LAPLACE), "Use Laplace filter for estimating sharpness", "path/ percent"),
}
