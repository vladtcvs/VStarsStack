#
# Copyright (c) 2023 Vladislav Tsendrovskii
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

import vstarstack.tool.common
import vstarstack.tool.cfg
import vstarstack.library.common

from vstarstack.library.stars import describe

def get_brightest(stars, N, mindistance):
    """Get first N brightest stars"""
    sample = []
    for star in stars:
        for selected in sample:
            if abs(selected["y"] - star["y"]) < mindistance and \
               abs(selected["x"] - star["x"]) < mindistance:
                break
        else:
            sample.append(star)
            if len(sample) >= N:
                break
    return sample

def build_descriptions(image_description: dict,
                       num_main : int,
                       mindist : float,
                       use_angles : bool):
    """Build descriptors for first num_main brightest stars"""
    mindistance = min(image_description["h"], image_description["w"]) * mindist
    stars = image_description["stars"]
    stars = sorted(stars, key=lambda item: item["size"], reverse=True)
    main = get_brightest(stars, num_main, mindistance)
    descriptors = describe.build_descriptors(main, use_angles)

    image_description["main"] = []
    for item, desc in zip(main, descriptors):
        record = {
            "star" : item,
            "descriptor" : desc.serialize(),
        }
        image_description["main"].append(record)
    return image_description

def run(project: vstarstack.tool.cfg.Project, argv: list):
    if len(argv) >= 2:
        path = argv[0]
        outpath = argv[1]
    else:
        path = project.config.paths.descs
        outpath = project.config.paths.descs

    num_main = project.config.stars.describe.num_main
    mindist = project.config.stars.describe.mindist

    files = vstarstack.tool.common.listfiles(path, ".json")

    for name, filename in files:
        print(name)
        with open(filename, encoding='utf8') as f:
            description = json.load(f)
        description = build_descriptions(description,
                                         num_main,
                                         mindist,
                                         project.config.stars.use_angles)

        jsonfname = os.path.join(outpath, name + ".json")
        vstarstack.tool.common.check_dir_exists(jsonfname)
        with open(jsonfname, "w", encoding='utf8') as f:
            json.dump(description, f, indent=4)
