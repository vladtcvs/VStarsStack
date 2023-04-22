"""Apply shift to image"""
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


import json
import os
import multiprocessing as mp
import numpy as np

from vstarstack.movement.movement import MovementException
from vstarstack.movement.flat import Movement as mf
from vstarstack.movement.sphere import Movement as ms
import vstarstack.cfg

import vstarstack.projection.perspective
import vstarstack.common
import vstarstack.data

import vstarstack.shift.shift_image

ncpu = vstarstack.cfg.nthreads


def make_shift(name, infname, image_shift, outfname):
    """Make shifted image"""
    print(name)
    if not os.path.exists(infname) or image_shift is None:
        print("skip")
        return

    dataframe = vstarstack.data.DataFrame.load(infname)

    proj = dataframe.params["projection"]
    if proj == "perspective":
        h = dataframe.params["h"]
        w = dataframe.params["w"]
        W = dataframe.params["perspective_kw"] * w
        H = dataframe.params["perspective_kh"] * h
        F = dataframe.params["perspective_F"]
        proj = vstarstack.projection.perspective.Projection(W, H, F, w, h)
    else:
        raise vstarstack.data.InvalidParameterException("projection",  proj)

    for channel in dataframe.get_channels():
        image, opts = dataframe.get_channel(channel)
        if opts["weight"]:
            continue
        if opts["encoded"]:
            continue

        weight_channel = None
        if channel in dataframe.links["weight"]:
            weight_channel = dataframe.links["weight"][channel]

        if weight_channel:
            weight, _ = dataframe.get_channel(weight_channel)
        else:
            weight = np.ones(image.shape)*1

        shifted, shifted_weight = vstarstack.shift.shift_image.shift_image(
            image, image_shift, proj, weight)
        dataframe.add_channel(shifted, channel, **opts)
        dataframe.add_channel(shifted_weight, weight_channel, weight=True)
        dataframe.add_channel_link(channel, weight_channel, "weight")

    dataframe.store(outfname)


def run(project: vstarstack.cfg.Project, argv: list):
    """Run shift applyment on path"""
    if len(argv) > 0:
        npy_dir = argv[0]
        shifts_fname = argv[1]
        shifted_dir = argv[2]
    else:
        npy_dir = project.config["paths"]["npy-fixed"]
        shifts_fname = project.config["paths"]["absolute-shifts"]
        shifted_dir = project.config["paths"]["aligned"]

    with open(shifts_fname, encoding='utf8') as file:
        data = json.load(file)
        shifts = data["movements"]

    shift_type = data["shift_type"]
    if shift_type == "flat":
        Movement = mf
    elif shift_type == "sphere":
        Movement = ms
    else:
        raise MovementException(shift_type, "Unknown movement type!")

    for name in shifts:
        if shifts[name] is not None:
            shifts[name] = Movement.deserialize(shifts[name])

    images = vstarstack.common.listfiles(npy_dir, ".zip")

    for name, _ in images:
        if name not in shifts:
            shifts[name] = None

    with mp.Pool(ncpu) as pool:
        pool.starmap(make_shift, [(name, filename, shifts[name], os.path.join(
                     shifted_dir, name + ".zip")) for name, filename in images])
