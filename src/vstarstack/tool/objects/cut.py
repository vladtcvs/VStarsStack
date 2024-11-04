#
# Copyright (c) 2022-2024 Vladislav Tsendrovskii
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
import numpy as np
import cv2

import vstarstack.tool.cfg
import vstarstack.library.common
import vstarstack.library.data
import vstarstack.tool.common

def run(project: vstarstack.tool.cfg.Project, argv: list[str]):
    if len(argv) < 3:
        npypath = project.config.paths.light.npy
        jsonpath = project.config.paths.descs
        cutpath = project.config.paths.light.npy
    else:
        npypath = argv[0]
        jsonpath = argv[1]
        cutpath = argv[2]

    if len(argv) > 3:
        margin = int(argv[3])
    else:
        margin = project.config.objects.margin

    files = vstarstack.tool.common.listfiles(jsonpath, ".json")
    maxr = 0
    for name, filename in files:
        with open(filename, encoding='utf8') as f:
            detection = json.load(f)
        r = int(detection["object"]["r"])
        print(f"Loading info: {name}, r = {r}")
        if r > maxr:
            maxr = r
    disk_radius=int(maxr*1.1+0.5)
    maxr = int(maxr+0.5)+margin
    size = 2*maxr+1
    print("maxr = ", maxr, " size = ", size)

    mask = np.zeros((size,size))
    cv2.circle(mask, (maxr,maxr), disk_radius, 1, -1)

    for name, filename in files:
        print(name)
        with open(filename, encoding='utf8') as f:
            detection = json.load(f)

        x = int(detection["object"]["x"])
        y = int(detection["object"]["y"])
        left = int(x - maxr)
        right = left + size
        top = int(y - maxr)
        bottom = top + size

        imagename = os.path.join(npypath, name + ".zip")
        try:
            image = vstarstack.library.data.DataFrame.load(imagename)
        except Exception:
            print("Can not load ", name)
            continue

        image.add_channel(mask, "mask", mask=True)
        for channel in list(image.get_channels()):
            layer, opts = image.get_channel(channel)
            if image.get_channel_option(channel, "encoded"):
                image.remove_channel(channel)
                continue
            if image.get_channel_option(channel, "mask"):
                continue

            w = layer.shape[1]
            h = layer.shape[0]

            source_top = top
            source_left = left
            target_top = 0
            target_left = 0

            copy_w = size
            copy_h = size

            if source_top < 0:
                space = 0 - source_top
                source_top += space
                target_top += space
                copy_h -= space

            if source_left < 0:
                space = 0 - source_left
                source_left += space
                target_left += space
                copy_w -= space

            if source_top + copy_h > h:
                space = source_top + copy_h - h
                copy_h -= space

            if source_left + copy_w > w:
                space = source_left + copy_w - w
                copy_w -= space

            img = np.zeros((size, size))
            img[target_top:target_top+copy_h, target_left:target_left+copy_w] = layer[source_top:source_top+copy_h, source_left:source_left+copy_w]
            image.replace_channel(img, channel, **opts)
            image.add_channel_link(channel, "mask", "mask")

            detection["roi"] = {
                "x1": left,
                "y1": top,
                "x2": right,
                "y2": bottom
            }
            vstarstack.tool.common.check_dir_exists(filename)
            with open(filename, "w", encoding='utf8') as f:
                json.dump(detection, f, indent=4, ensure_ascii=False)

        image.params["w"] = size
        image.params["h"] = size

        outname = os.path.join(cutpath, name + ".zip")
        if len(image.get_channels()) != 0:
            vstarstack.tool.common.check_dir_exists(outname)
            image.store(outname)
        else:
            print(f"Skipping {outname}")
            if os.path.exists(outname):
                os.remove(outname)
