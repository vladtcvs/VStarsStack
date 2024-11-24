#
# Copyright (c) 2023-2024 Vladislav Tsendrovskii
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
import logging

import vstarstack.tool.common
import vstarstack.tool.cfg
import vstarstack.library.common

from vstarstack.library.stars import describe

logger = logging.getLogger(__name__)

def build_descriptions(image_description: dict,
                       use_angles : bool):
    """Build descriptors for first num_main brightest stars"""
    stars = [item["keypoint"] for item in image_description["points"]]
    stars = sorted(stars, key=lambda item: item["size"], reverse=True)
    descriptors = describe.build_descriptors(stars, use_angles)

    image_description["points"] = []

    for point, desc in zip(stars, descriptors):
        record = {
            "keypoint" : point,
            "descriptor-type" : "star",
            "descriptor" : desc.serialize(),
        }
        image_description["points"].append(record)
    return image_description

def run(project: vstarstack.tool.cfg.Project, argv: list):
    if len(argv) >= 2:
        path = argv[0]
        outpath = argv[1]
    else:
        path = project.config.paths.descs
        outpath = project.config.paths.descs

    files = vstarstack.tool.common.listfiles(path, ".json")

    for name, filename in files:
        logger.info(f"Processing {name}")
        with open(filename, encoding='utf8') as f:
            description = json.load(f)
        description = build_descriptions(description, project.config.stars.use_angles)
        jsonfname = os.path.join(outpath, name + ".json")
        vstarstack.tool.common.check_dir_exists(jsonfname)
        with open(jsonfname, "w", encoding='utf8') as f:
            json.dump(description, f, indent=4)
