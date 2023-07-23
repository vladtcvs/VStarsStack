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

import os
import json
import matplotlib.pyplot as plt

import vstarstack.library.common
import vstarstack.library.projection.perspective
import vstarstack.library.loaders.classic
import vstarstack.library.stars.detect
import vstarstack.library.stars.describe
import vstarstack.library.stars.match
import vstarstack.library.cluster
import vstarstack.library.movement.find_shift
import vstarstack.library.movement.select_shift
import vstarstack.library.movement.move_image
import vstarstack.library.merge

from vstarstack.library.movement.sphere import Movement

dir_path = os.path.dirname(os.path.realpath(__file__))

def test_1():
    W = 12.7
    H = 12.7
    F = 10000
    w = 1270
    h = 1270
    proj = vstarstack.library.projection.perspective.Projection(W, H, F, w, h)

    names = ["1.png", "2.png", "3.png", "4.png"]
    images = []
    for name in names:
        fname = os.path.join(dir_path, os.path.join("matching", name))
        df = next(vstarstack.library.loaders.classic.readjpeg(fname))
        images.append(df)

    stars = []
    for image in images:
        layer,_ = image.get_channel("L")
        s = vstarstack.library.stars.detect.detect_stars(layer)
        assert len(s) == 4
        stars.append(s)

    used_stars = stars
    descs = []
    for s in used_stars:
        d = vstarstack.library.stars.describe.build_descriptors(s, True, proj)
        assert len(d) == 4
        descs.append(d)

    matcher = vstarstack.library.stars.match.DescriptorMatcher(3, 1e-3, 1e-3, 3e-1)
    match_table = vstarstack.library.stars.match.build_stars_match_table(matcher, descs, 0)
    id_clusters = vstarstack.library.cluster.find_clusters_in_match_table(match_table)
    star_clusters = []
    for cluster in id_clusters:
        star_cluster = {}
        for desc_id, star_id in cluster.items():
            name = names[desc_id]
            star_cluster[name] = used_stars[desc_id][star_id]
        star_clusters.append(star_cluster)

    assert len(star_clusters) == 4
    for sc in star_clusters:
        assert len(sc) == 4

    shifts, errors = vstarstack.library.movement.find_shift.build_movements(Movement, star_clusters)
    assert len(errors) == 0
    assert len(shifts) == 4
    for name in shifts:
        assert len(shifts[name]) == 4

    shifts["1.png"].pop("1.png")
    shifts["1.png"].pop("3.png")
    shifts["1.png"].pop("4.png")

    shifts["2.png"].pop("1.png")
    shifts["2.png"].pop("2.png")
    shifts["2.png"].pop("4.png")

    shifts["3.png"].pop("1.png")
    shifts["3.png"].pop("2.png")
    shifts["3.png"].pop("3.png")

    shifts["4.png"].pop("1.png")
    shifts["4.png"].pop("2.png")
    shifts["4.png"].pop("3.png")
    shifts["4.png"].pop("4.png")

    shifts = vstarstack.library.movement.find_shift.complete_movements(Movement, shifts, True)
    assert len(shifts) == 4
    for name in shifts:
        assert len(shifts[name]) == 4


    basic_name = vstarstack.library.movement.select_shift.select_base_image(shifts)
    assert basic_name == "1.png"

    shifts = shifts[basic_name]
    moved = []
    for id, image in enumerate(images):
        t = shifts[names[id]]
        mvd = vstarstack.library.movement.move_image.move_dataframe(image, t, proj)
        moved.append(mvd)

    assert len(moved) == 4

    source = vstarstack.library.common.ListImageSource(moved)
    merged = vstarstack.library.merge.simple_add(source)
    layer,_ = merged.get_channel("L")

    merged_stars = vstarstack.library.stars.detect.detect_stars(layer)
    assert len(merged_stars) == 4
