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
import csv

from vstarstack.library.photometry.star_background import calculate_bgs_df
from vstarstack.library.data import DataFrame


def process_image(project, argv : list):
    image_path = argv[0]
    stars_path = argv[1]
    report_fname = argv[2]
    with open(stars_path) as f:
        stars = json.load(f)
    stars = [{"x" : star["keypoint"]["x"], "y" : star["keypoint"]["y"], "radius" : star["keypoint"]["radius"]} for star in stars["points"]]

    df = DataFrame.load(image_path)
    backgrounds, channels = calculate_bgs_df(df, stars)
    with open(report_fname, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        header1 = ["x", "y", "radius","","name","mag","lum","", "star pixels", "background pixels",""]
        header2 = ["","","","","","","","","","",""]
        for channel in channels:
            header2.append(channel)
            header2.append("")
            header2.append("")
            header1 += ["star value", "background value", ""]
        header2 += ["Total", "", "", ""]
        header1 += ["star value", "background value", "mean star pixel", "mean background pixel"]
        writer.writerow(header2)
        writer.writerow(header1)

        for item in backgrounds:
            bg_npix = 0
            star_npix = 0
            row = None
            sum_star = 0
            sum_bg = 0
            for channel in channels:
                if row is None:
                    mag = ""
                    lum = ""
                    name = ""
                    row = [item["x"], item["y"], int(item["radius"]+0.5), "", name, mag, lum, ""]
                    star_npix = int(item["channels"][channel]["star_npix"])
                    bg_npix = int(item["channels"][channel]["bg_npix"])
                    row += [star_npix, bg_npix, ""]
                row += [int(item["channels"][channel]["star"])]
                row += [int(item["channels"][channel]["bg"])]
                row += [""]

                sum_star += int(item["channels"][channel]["star"])
                sum_bg += int(item["channels"][channel]["bg"])

            row += [sum_star, sum_bg, int(sum_star/star_npix), int(sum_bg/bg_npix)]
            writer.writerow(row)

commands = {
    "image": (process_image,
             "build report for clusters",
             "image.zip stars.json report.csv"),
}
