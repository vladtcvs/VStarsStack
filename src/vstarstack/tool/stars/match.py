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

import multiprocessing as mp
import json
import math

import vstarstack.tool.usage
import vstarstack.tool.cfg
import vstarstack.library.common

from vstarstack.library.stars.match import DescriptorMatcher
from vstarstack.library.stars import describe

def match_stars(matcher : DescriptorMatcher,
                name1 : str, name2 : str,
                desc1 : list[describe.Descriptor],
                desc2 : list[describe.Descriptor]):
    """match stars between images"""
    match = matcher.build_match(desc1, desc2)
    return (name1, name2, match)

def process(project: vstarstack.tool.cfg.Project, argv: list):
    if len(argv) >= 2:
        starsdir = argv[0]
        matchfile = argv[1]
    else:
        starsdir = project.config.paths.descs
        matchfile = project.config.stars.paths.matchfile

    starsfiles = vstarstack.library.common.listfiles(starsdir, ".json")
    descs = []
    name_fname = {}
    w = None
    h = None
    for name, fname in starsfiles:
        with open(fname, encoding='utf8') as file:
            desc = json.load(file)
        if w is None or w > desc["w"]:
            w = desc["w"]
        if h is None or h > desc["h"]:
            h = desc["h"]

        desc = [describe.Descriptor.deserialize(item["descriptor"]) for item in desc["main"]]
        descs.append((name, desc))
        name_fname[name] = fname

    W = project.config.telescope.camera.pixel_W / 1000 * w
    H = project.config.telescope.camera.pixel_H / 1000 * h
    F = project.config.telescope.scope.F
    fov1 = math.atan(W / F)
    fov2 = math.atan(H / F)
    fov = min(fov1, fov2)

    print(f"W = {W:.2f} mm")
    print(f"H = {H:.2f} mm")
    print(f"F = {F:.2f} mm")
    print(f"Fov = {fov*180/math.pi:.2f}°")

    max_angle_diff = project.config.stars.match.max_angle_diff_k * fov
    max_dangle_diff = project.config.stars.match.max_dangle_diff * math.pi/180
    max_size_diff = project.config.stars.match.max_size_diff
    min_matched_ditems = project.config.stars.match.min_matched_ditems

    matcher = DescriptorMatcher(min_matched_ditems,
                                max_angle_diff,
                                max_dangle_diff,
                                max_size_diff)
    total = len(starsfiles)**2
    print(f"total = {total}")
    args = []
    for desc1 in descs:
        for desc2 in descs:
            args.append((matcher, desc1[0], desc2[0], desc1[1], desc2[1]))
    match_table = {}
    with mp.Pool(vstarstack.tool.cfg.nthreads) as pool:
        results = pool.starmap(match_stars, args)
        for name1, name2, match in results:
            if name1 not in match_table:
                match_table[name1] = {}
            match_table[name1][name2] = match

    with open(matchfile, "w", encoding='utf8') as file:
        json.dump(match_table, file, indent=4, ensure_ascii=False)

def run(project: vstarstack.tool.cfg.Project, argv: list):
    process(project, argv)
