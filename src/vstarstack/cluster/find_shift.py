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

import vstarstack.usage

import vstarstack.movement.flat
import vstarstack.movement.sphere
import vstarstack.cfg
import vstarstack.common

percent = 60


def select_movement(project: vstarstack.cfg.Project):
    if project.use_sphere:
        return vstarstack.movement.sphere.Movement
    else:
        return vstarstack.movement.flat.Movement


def find_shift(movement, star1, star2):
    star1_from = star1[0]
    star1_to = star1[1]

    star2_from = star2[0]
    star2_to = star2[1]
    t = movement.build(star1_from, star2_from, star1_to, star2_to)
    return t


def run(project: vstarstack.cfg.Project, argv: list):
    if len(argv) > 1:
        clusters_fname = argv[0]
        shifts_fname = argv[1]
    else:
        clusters_fname = project.config["cluster"]["path"]
        shifts_fname = project.config["paths"]["relative-shifts"]

    with open(clusters_fname, encoding='utf8') as file:
        clusters = json.load(file)

    print(clusters)
    names = []
    for cluster in clusters:
        for name in cluster:
            if name not in names:
                names.append(name)

    movements = {}

    for name1 in names:
        movements[name1] = {}
        for name2 in names:
            print("%s / %s" % (name1, name2))
            stars = []
            for cluster in clusters:
                if name1 not in cluster:
                    continue
                if name2 not in cluster:
                    continue
                if project.use_sphere:
                    star_to = (cluster[name1]["lat"], cluster[name1]["lon"])
                    star_from = (cluster[name2]["lat"], cluster[name2]["lon"])
                else:
                    star_to = (cluster[name1]["y"], cluster[name1]["x"])
                    star_from = (cluster[name2]["y"], cluster[name2]["x"])

                stars.append((star_from, star_to))

            ts = []

            for i in range(len(stars)-1):
                star1_from, star1_to = stars[i]
                for j in range(i+1, len(stars)):
                    star2_from, star2_to = stars[j]
                    try:
                        t = find_shift(select_movement(
                            project), (star1_from, star1_to), (star2_from, star2_to))
                        ts.append(t)
                    except Exception as _:
                        print("Can not find movement")
                        continue

            if len(ts) >= 1:
                t = select_movement(project).average(ts, percent)
                movements[name1][name2] = t.serialize()
            else:
                movements[name1][name2] = None
    data = {
        "movements": movements
    }
    if project.use_sphere:
        data["shift_type"] = "sphere"
    else:
        data["shift_type"] = "flat"
    data["format"] = "relative"
    with open(shifts_fname, "w") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
