#
# Copyright (c) 2025 Vladislav Tsendrovskii
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
import datetime
import logging

import vstarstack.tool.cfg
import vstarstack.tool.common
import vstarstack.library.data
import vstarstack.library.projection.tools
import vstarstack.library.movement.sphere

logger = logging.getLogger(__name__)

def linear(project : vstarstack.tool.cfg.Project, argv: list[str]):
    # path with image files
    path = argv[0]
    
    # object position on first file - with lowest UTC
    x1 = int(argv[1])
    y1 = int(argv[2])

    # object position of last file - with highest UTC
    x2 = int(argv[3])
    y2 = int(argv[4])

    # output shift file
    shift_file = argv[5]

    # we align all files so object position is the same on all images
    # we use UTC parameter for determine interpolation position
    utcs = {}
    projections = {}
    lowest_utc = None
    highest_utc = None
    lowest_name = None
    highest_name = None
    imgs = vstarstack.tool.common.listfiles(path, ".zip")
    for name, fname in imgs:
        logging.info(f"Processing {name}")
        df = vstarstack.library.data.DataFrame.load(fname)
        utc = df.get_parameter("UTC")
        if utc is None:
            logging.warning(f"File {name} doesn't have UTC")
            continue
        try:
            logging.info(f"UTC {utc}")
            utc = datetime.datetime.fromisoformat(utc)
        except:
            logging.error(f"File {name} has incorrect UTC {utc}")
            continue
        utcs[name] = utc
        projections[name] = vstarstack.library.projection.tools.get_projection(df)
        if lowest_utc is None or utc < lowest_utc:
            lowest_utc = utc
            lowest_name = name
        if highest_utc is None or utc > highest_utc:
            highest_utc = utc
            highest_name = name

    first_lonlat = projections[lowest_name].project(x1, y1)
    last_lonlat = projections[highest_name].project(x2, y2)
    movement = vstarstack.library.movement.sphere.Movement.build_by_single(last_lonlat, first_lonlat)
    identity = vstarstack.library.movement.sphere.Movement.identity()

    delta = highest_utc - lowest_utc
    shifts = {}
    for name in utcs:
        interpolation = (utcs[name] - lowest_utc) / delta
        move = vstarstack.library.movement.sphere.Movement.interpolate([movement, identity], [interpolation, 1-interpolation])
        shifts[name] = move.serialize()
    with open(shift_file, "w", encoding='utf8') as f:
        json.dump(shifts, f, ensure_ascii=False, indent=4)

commands = {
    "linear-interpolation": (linear, "Interpolate moving object position linear between first and last images", "path/ X1 Y1 X2 Y2 shift-linear.json"),
}
