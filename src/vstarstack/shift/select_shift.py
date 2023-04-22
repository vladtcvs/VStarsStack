"""Select reference image for shift"""
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

import vstarstack.cfg
from vstarstack.movement.sphere import Movement as ms
from vstarstack.movement.flat import Movement as mf
from vstarstack.movement.movement import MovementException


def select_base_image(infname, outfname):
    """Select image which wouldn't move"""
    with open(infname, encoding='utf8') as file:
        data = json.load(file)

    shiftsf = data["movements"]
    if data["shift_type"] == "flat":
        Movement = mf
    elif data["shift_type"] == "sphere":
        Movement = ms
    else:
        raise MovementException(data["shift_type"], "Unknown shift type!")

    shifts = {}
    names = []

    for name1 in shiftsf:
        shifts[name1] = {}
        names.append(name1)
        for name2 in shiftsf:
            if shiftsf[name1][name2] is None:
                continue
            shifts[name1][name2] = Movement.deserialize(shiftsf[name1][name2])

    # find image with minimal shift to other
    name0 = None
    mindistance = None
    max_size_of_cluster = None
    for name in names:
        size_of_cluster = len(shifts[name])
        print(name, size_of_cluster)
        if max_size_of_cluster is not None and size_of_cluster < max_size_of_cluster:
            continue
        if max_size_of_cluster is None or size_of_cluster > max_size_of_cluster:
            max_size_of_cluster = size_of_cluster
            mindistance = None
        distance = 0
        for name2 in shifts[name]:
            shift = shifts[name][name2]
            distance += shift.magnitude()

        if mindistance is None or distance < mindistance:
            mindistance = distance
            name0 = name

    print("Select:", name0, max_size_of_cluster)

    result = {}
    result["format"] = "absolute"
    result["shift_type"] = data["shift_type"]
    result["movements"] = data["movements"][name0]

    with open(outfname, "w", encoding='utf8') as file:
        json.dump(result, file, indent=4, ensure_ascii=False)


def run(project: vstarstack.cfg.Project, argv: list):
    "Run selection os base image"
    if len(argv) < 2:
        relative_name = project.config["paths"]["relative-shifts"]
        absolute_name = project.config["paths"]["absolute-shifts"]
    else:
        relative_name = argv[0]
        absolute_name = argv[1]

    select_base_image(relative_name, absolute_name)
