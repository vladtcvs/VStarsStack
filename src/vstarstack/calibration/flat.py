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
import multiprocessing as mp
import numpy as np
import cv2
import scipy.signal

import vstarstack.cfg
import vstarstack.common
import vstarstack.data
import vstarstack.usage


ncpu = max(1, mp.cpu_count()-1)


def flatten(name, fname, out, flat_file):
    print(name)

    img = vstarstack.data.DataFrame.load(fname)
    flat_img = vstarstack.data.DataFrame.load(flat_file)

    for channel in img.get_channels():
        image, opts = img.get_channel(channel)
        if not opts["brightness"]:
            continue

        if channel in flat_img.get_channels():
            image = image / flat_img.get_channel(channel)[0]

        img.add_channel(image, channel, **opts)
    img.store(out)


def process_file(input_file, output_file, flat_file):
    """Process single file"""
    name = os.path.splitext(os.path.basename(input_file))[0]
    flatten(name, input_file, output_file, flat_file)


def process_dir(input_dir, output_dir, flat_file):
    """Process directory"""
    files = vstarstack.common.listfiles(input_dir, ".zip")
    pool = mp.Pool(ncpu)
    args = [(name, fname, os.path.join(output_dir, name + ".zip"), flat_file)
            for name, fname in files]
    pool.starmap(flatten, args)
    pool.close()


def process(project: vstarstack.cfg.Project, argv: list):
    """Process path"""
    if len(argv) >= 3:
        input_path = argv[0]
        output_file = argv[1]
        flat_file = argv[2]
    else:
        input_path = project.config["paths"]["npy-fixed"]
        output_file = project.config["paths"]["npy-fixed"]
        flat_file = project.config["calibration"]["flat"]["path"]

    if os.path.isdir(input_path):
        process_dir(input_path, output_file, flat_file)
    else:
        process_file(input_path, output_file, flat_file)


def prepare_flats(project: vstarstack.cfg.Project, argv: list):
    """Prepare flat files for processing"""
    if len(argv) >= 2:
        npys = argv[0]
        result = argv[1]
    else:
        npys = project.config["calibration"]["flat"]["npy"]
        result = project.config["calibration"]["flat"]["path"]

    channels = {}
    files = vstarstack.common.listfiles(npys, ".zip")
    for _, fname in files:
        flat_frame = vstarstack.data.DataFrame.load(fname)
        for channel_name in flat_frame.get_channels():
            image, opts = flat_frame.get_channel(channel_name)
            if not opts["brightness"]:
                continue
            if channel_name not in channels:
                channels[channel_name] = []
            channels[channel_name].append(image)

    result_image = vstarstack.data.DataFrame()
    for channel_name, channel in channels.items():
        channel_images_sum = sum(channel)
        channel_images_sum = cv2.GaussianBlur(channel_images_sum, (51, 51), 0)
        channel_images_sum = channel_images_sum / np.amax(channel_images_sum)
        result_image.add_channel(channel_images_sum, channel_name, brightness=True)
    result_image.store(result)


def prepare_sky(_project: vstarstack.cfg.Project, argv: list):
    """Generate flat image"""
    out = argv[1]
    imgs = argv[0]
    files = vstarstack.common.listfiles(imgs, ".zip")
    channels = {}
    for _, fname in files:
        frame = vstarstack.data.DataFrame.load(fname)
        for channel_name in frame.get_channels():
            image, opts = frame.get_channel(channel_name)
            if not opts["brightness"]:
                continue

            image = cv2.GaussianBlur(image, (5, 5), 0)
            if channel_name not in channels:
                channels[channel_name] = []
            channels[channel_name].append(image / np.amax(image))

    thr = 0.006
    median_filter_size = 31
    blur_size = 301

    result_image = vstarstack.data.DataFrame()
    for channel_name, channel in channels.items():
        print(channel_name)
        avg = sum(channel) / len(channel)
        skyes = []
        for i,img in enumerate(channel):
            img = channel[i]
            diff = abs(img - avg)
            mask_idx = np.where(diff > thr)

            sky = img
            sky[mask_idx] = np.average(sky)
            print("Apply median filter")
            sky_fixed = scipy.signal.medfilt2d(sky, median_filter_size)
            print("\tDone")
            sky_fixed = cv2.GaussianBlur(sky_fixed, (blur_size, blur_size), 0)

            skyes.append(sky_fixed)

        sky_fixed = sum(skyes) / len(skyes)
        sky_fixed = sky_fixed / np.amax(sky_fixed)
        result_image.add_channel(sky_fixed, channel_name, brightness=True)
    result_image.store(out)


commands = {
    "prepare": (prepare_flats,
                "flat prepare",
                "prepare flat frames"),
    "prepare-starsky": (prepare_sky,
                        "flat prepare-starsky inputs/ output.zip",
                        "prepare flat frames from N images with stars"),
    "*": (process,
          "flat",
          "(input.file output.file | input/ output/) flat.zip"),
}


def run(project: vstarstack.cfg.Project, argv: list):
    vstarstack.usage.run(project, argv, "image-fix flat", commands, autohelp=False)
