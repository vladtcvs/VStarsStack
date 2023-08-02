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

import os
import numpy as np
import imageio

import matplotlib.pyplot as plt

from vstarstack.library.image_process.border import border
from vstarstack.library.image_process.cut import cut
import vstarstack.library.data
import vstarstack.tool.common

import vstarstack.tool.usage
import vstarstack.tool.cfg

POWER = 1
SLOPE = vstarstack.tool.cfg.get_param("multiply", float, 1)
FLOOR = vstarstack.tool.cfg.get_param("clip_floor", bool, False)

def _make_frames(dataframe, channels, *, slope=1, power=1, clip_floor=False):
    if channels == "RGB":
        r, _ = dataframe.get_channel("R")
        g, _ = dataframe.get_channel("G")
        b, _ = dataframe.get_channel("B")

        rgb = np.zeros((r.shape[0], r.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        amax = np.amax(rgb)
        rgb = rgb / amax
        rgb = np.clip(rgb*slope, 0, 1)**power

        frames = {"RGB": rgb}

    else:
        frames = {}
        if channels is None:
            channels = dataframe.get_channels()

        for channel in channels:
            print("Channel = ", channel)
            img, options = dataframe.get_channel(channel)
            print("Shape = ", img.shape)

            if options["brightness"]:
                img = img.astype(np.float64)
                amin = max(np.amin(img), 0)
                amax = np.amax(img)
                print(f"{channel}: {amin} - {amax}")
                if clip_floor:
                    if amax - amin > 0:
                        img = (img - amin)/(amax-amin)
                else:
                    img = img / amax
                img = np.clip(img*slope, 0, 1)**power

            frames[channel] = img
    return frames


def _show(_project, argv):
    path = argv[0]
    if len(argv) > 1:
        if argv[1] == "RGB":
            channel = "RGB"
        else:
            channel = argv[1:]
    else:
        channel = None

    dataframe = vstarstack.library.data.DataFrame.load(path)
    frames = _make_frames(dataframe,
                          channel,
                          slope=SLOPE,
                          power=POWER,
                          clip_floor=FLOOR)

    nch = len(frames)
    fig, axs = plt.subplots(1, nch)
    fig.patch.set_facecolor('#222222')

    index = 0
    for channel,img in frames.items():
        if nch > 1:
            subplot = axs[index]
        else:
            subplot = axs
        img = frames[channel].astype(np.float64)
        img = img / np.amax(img)
        subplot.imshow(img, cmap="gray", vmin=0, vmax=1.0)
        subplot.set_title(channel)
        index += 1

    plt.show()


def _convert(_project, argv):
    path = argv[0]
    out = argv[1]

    if len(argv) > 2:
        if argv[2] == "RGB":
            channels = "RGB"
        else:
            channels = argv[2:]
    else:
        channels = None

    dataframe = vstarstack.library.data.DataFrame.load(path)
    frames = _make_frames(dataframe, channels, slope=SLOPE, power=POWER)

    nch = len(frames)

    out = os.path.abspath(out)
    path = os.path.dirname(out)
    name, ext = os.path.splitext(os.path.basename(out))
    for channels, img in frames.items():
        if nch > 1:
            fname = os.path.join(path, f"{name}_{channels}.{ext}")
        else:
            fname = out

        img = img*65535
        img = img.astype('uint16')
        vstarstack.tool.common.check_dir_exists(fname)
        imageio.imwrite(fname, img)


def _cut(_project, argv):
    path = argv[0]
    left = int(argv[1])
    top = int(argv[2])
    right = int(argv[3])
    bottom = int(argv[4])
    out = argv[5]

    dataframe = vstarstack.library.data.DataFrame.load(path)
    result = cut(dataframe, left, top, right, bottom)
    vstarstack.tool.common.check_dir_exists(out)
    result.store(out)

def _rename_channel(_project, argv):
    name = argv[0]
    channel = argv[1]
    target = argv[2]
    print(name)
    dataframe = vstarstack.library.data.DataFrame.load(name)
    dataframe.rename_channel(channel, target)
    vstarstack.tool.common.check_dir_exists(name)
    dataframe.store(name)

def _exposures(_project, argv):
    fname = argv[0]
    dataframe = vstarstack.library.data.DataFrame.load(fname)
    channels = dataframe.get_channels()

    for channel in channels:
        _, opts = dataframe.get_channel(channel)
        if not opts["brightness"]:
            continue

        weight_channel = dataframe.links["weight"][channel]
        weight, _ = dataframe.get_channel(weight_channel)
        amaxw = np.amax(weight)
        print(f"{channel} : {amaxw}")


commands = {
    "show": (_show, "show image"),
    "convert": (_convert, "convert image"),
    "cut": (_cut, "cut part of image"),
    "rename-channel": (_rename_channel, "filename.zip original_name target_name - rename channel"),
    "exposure": (_exposures, "display image exposures per channel", "file.zip"),
}
