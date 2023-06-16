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

import vstarstack.tool.cfg
import vstarstack.library.common
import vstarstack.library.data

def run(project: vstarstack.tool.cfg.Project, argv: list[str]):
    if len(argv) < 3:
        npypath = project.config["paths"]["npy-fixed"]
        jsonpath = project.config["paths"]["descs"]
        cutpath = project.config["paths"]["npy-fixed"]
    else:
        npypath = argv[0]
        jsonpath = argv[1]
        cutpath = argv[2]

    if len(argv) > 3:
        margin = int(argv[3])
    else:
        margin = project.config["objects"]["margin"]

    require_size = project.config["objects"]["require_size"]

    files = vstarstack.library.common.listfiles(jsonpath, ".json")
    for name, filename in files:
        print(f"Loading info: {name}")
        with open(filename, encoding='utf8') as f:
            detection = json.load(f)

    maxr = 0
    for name, filename in files:
        with open(filename, encoding='utf8') as f:
            detection = json.load(f)

        r = int(detection["object"]["r"])
        if r > maxr:
            maxr = r
    maxr = int(maxr+0.5)+margin

    for name, filename in files:
        print(name)
        with open(filename, encoding='utf8') as f:
            detection = json.load(f)

        x = int(detection["object"]["x"])
        y = int(detection["object"]["y"])
        r = int(detection["object"]["r"])
        left = int(x - maxr)
        right = int(x + maxr)
        top = int(y - maxr)
        bottom = int(y + maxr)

        left = max(left, 0)
        top = max(top, 0)

        imagename = os.path.join(npypath, name + ".zip")
        try:
            image = vstarstack.library.data.DataFrame.load(imagename)
        except Exception:
            print("Can not load ", name)
            continue

        weight_links = dict(image.links["weight"])

        for channel in image.get_channels():
            img, opts = image.get_channel(channel)
            if opts["encoded"]:
                image.remove_channel(channel)
                continue

            img = img[top:bottom+1, left:right+1]

            if require_size:
                if img.shape[0] != 2*maxr + 1:
                    print("\tSkip %s" % channel)
                    image.remove_channel(channel)
                    continue
                if img.shape[1] != 2*maxr + 1:
                    print("\tSkip %s" % channel)
                    image.remove_channel(channel)
                    continue

            detection["roi"] = {
                "x1": left,
                "y1": top,
                "x2": right,
                "y2": bottom
            }
            image.add_channel(img, channel, **opts)
            with open(filename, "w", encoding='utf8') as f:
                json.dump(detection, f, indent=4, ensure_ascii=False)

        for ch, value in weight_links.items():
            image.add_channel_link(ch, value, "weight")

        outname = os.path.join(cutpath, name + ".zip")
        image.store(outname)
