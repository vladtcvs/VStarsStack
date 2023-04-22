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

import vstarstack.cfg
import vstarstack.common
import vstarstack.data
import vstarstack.usage

import os
import multiprocessing as mp

ncpu = vstarstack.cfg.nthreads


def remove_dark(name, fname, out, dark_file):
    print(name)

    img = vstarstack.data.DataFrame.load(fname)
    dark_img = vstarstack.data.DataFrame.load(dark_file)

    for channel in img.get_channels():
        image, opts = img.get_channel(channel)
        if not opts["brightness"]:
            continue

        if channel in dark_img.get_channels():
            image = image - dark_img.get_channel(channel)[0]

        img.add_channel(image, channel, **opts)

    img.store(out)


def process_file(input, output, flat_file):
    name = os.path.splitext(os.path.basename(input))[0]
    remove_dark(name, input, output, flat_file)


def process_dir(input, output, flat_file):
    files = vstarstack.common.listfiles(input, ".zip")
    pool = mp.Pool(ncpu)
    args = [(name, fname, os.path.join(output, name + ".zip"), flat_file)
            for name, fname in files]
    pool.starmap(remove_dark, args)
    pool.close()


def process(project: vstarstack.cfg.Project, argv: list):
    if len(argv) >= 3:
        input = argv[0]
        output = argv[1]
        flat_file = argv[2]
    else:
        input = project.config["paths"]["npy-fixed"]
        output = project.config["paths"]["npy-fixed"]
        flat_file = project.config["calibration"]["dark"]["path"]

    if os.path.isdir(input):
        process_dir(input, output, flat_file)
    else:
        process_file(input, output, flat_file)


def prepare_darks(project: vstarstack.cfg.Project, argv: list):
    if len(argv) >= 2:
        npys = argv[0]
        result = argv[1]
    else:
        npys = project.config["calibration"]["dark"]["npy"]
        result = project.config["calibration"]["dark"]["path"]

    channels = {}
    files = vstarstack.common.listfiles(npys, ".zip")
    for name, fname in files:
        print(name)
        dark_frame = vstarstack.data.DataFrame.load(fname)
        for channel in dark_frame.get_channels():
            image, options = dark_frame.get_channel(channel)
            if not options["brightness"]:
                continue
            if channel not in channels:
                channels[channel] = []
            channels[channel].append(image)

    result_image = vstarstack.data.DataFrame()
    for channel in channels:
        avg = sum(channels[channel]) / len(channels[channel])
        result_image.add_channel(avg, channel, brightness=True)
    result_image.store(result)


commands = {
    "prepare": (prepare_darks, "dark prepare", "prepare dark frames"),
    "*": (process, "dark", "(input.file output.file | input/ output/) dark.zip"),
}


def run(project: vstarstack.cfg.Project, argv: list):
    vstarstack.usage.run(project, argv, "image-fix dark",
                         commands, autohelp=False)
