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


from vstarstack.library.stars import describe
from vstarstack.library.stars import match
from vstarstack.library import cluster

thr = 1e-6

def test_match_table_1():
    star1_1 = {"id" : 1, "size" : 1, "lon" : 0, "lat" : 0}
    star2_1 = {"id" : 2, "size" : 2, "lon" : 1e-3, "lat" : 0}
    star3_1 = {"id" : 3, "size" : 3, "lon" : 0, "lat" : 1e-3}
    desc_1 = describe.build_descriptor_angled(star1_1, [star2_1, star3_1])

    star1_2 = {"id" : 1, "size" : 1, "lon" : 0, "lat" : 0}
    star2_2 = {"id" : 2, "size" : 2, "lon" : 1e-3, "lat" : 0}
    star3_2 = {"id" : 3, "size" : 3, "lon" : 0, "lat" : 1e-3}
    desc_2 = describe.build_descriptor_angled(star1_2, [star2_2, star3_2])

    matcher = match.DescriptorMatcher(1, 1e-3, 1e-3, 1e-2)
    match_table = matcher._build_match_table(desc_1.items, desc_2.items)

    assert 0 in match_table
    assert 0 in match_table[0]
    assert match_table[0][0] == 0

def test_match_items_1():
    star1_1 = {"id" : 1, "size" : 1, "lon" : 0, "lat" : 0}
    star2_1 = {"id" : 2, "size" : 2, "lon" : 1e-3, "lat" : 0}
    star3_1 = {"id" : 3, "size" : 3, "lon" : 0, "lat" : 1e-3}
    desc_1 = describe.build_descriptor_angled(star1_1, [star2_1, star3_1])

    star1_2 = {"id" : 1, "size" : 1, "lon" : 0, "lat" : 0}
    star2_2 = {"id" : 2, "size" : 2, "lon" : 1e-3, "lat" : 0}
    star3_2 = {"id" : 3, "size" : 3, "lon" : 0, "lat" : 1e-3}
    desc_2 = describe.build_descriptor_angled(star1_2, [star2_2, star3_2])

    matcher = match.DescriptorMatcher(1, 1e-3, 1e-3, 1e-2)
    matches = matcher._get_matched_items(desc_1.items, desc_2.items)

    assert 0 in matches
    assert matches[0] == 0

def test_match_descs_1():
    star1_1 = {"id" : 1, "size" : 1, "lon" : 0, "lat" : 0}
    star2_1 = {"id" : 2, "size" : 2, "lon" : 1e-3, "lat" : 0}
    star3_1 = {"id" : 3, "size" : 3, "lon" : 0, "lat" : 1e-3}
    desc_1 = describe.build_descriptor_angled(star1_1, [star2_1, star3_1])

    star1_2 = {"id" : 1, "size" : 1, "lon" : 0, "lat" : 0}
    star2_2 = {"id" : 2, "size" : 2, "lon" : 1e-3, "lat" : 0}
    star3_2 = {"id" : 3, "size" : 3, "lon" : 0, "lat" : 1e-3}
    desc_2 = describe.build_descriptor_angled(star1_2, [star2_2, star3_2])

    matcher = match.DescriptorMatcher(1, 1e-3, 1e-3, 1e-2)
    assert matcher.check_match(desc_1, desc_2) is True

def test_match_1():
    stars1 = [
        {
            "id" : 1,
            "lon" : 0,
            "lat" : 0,
            "size" : 1
        },
        {
            "id" : 2,
            "lon" : 1e-3,
            "lat" : 0,
            "size" : 2
        },
        {
            "id" : 3,
            "lon" : 0,
            "lat" : 2e-3,
            "size" : 3
        },
    ]
    stars2 = [
        {
            "id" : 1,
            "lon" : 0,
            "lat" : 0,
            "size" : 1
        },
        {
            "id" : 2,
            "lon" : 1e-3,
            "lat" : 0,
            "size" : 2
        },
        {
            "id" : 3,
            "lon" : 0,
            "lat" : 2e-3,
            "size" : 3
        },
    ]
    descriptors_1 = describe.build_descriptors(stars1, True, None)
    assert len(descriptors_1) == 3
    descriptors_2 = describe.build_descriptors(stars2, True, None)
    assert len(descriptors_2) == 3

    matcher = match.DescriptorMatcher(1, 1e-3, 1e-3, 1e-2)
    matched = matcher.build_match(descriptors_1, descriptors_2)    
    assert matched[0] == 0
    assert matched[1] == 1
    assert matched[2] == 2

def test_match_2():
    stars1 = [
        {
            "id" : 1,
            "lon" : 0,
            "lat" : 0,
            "size" : 1
        },
        {
            "id" : 2,
            "lon" : 1e-3,
            "lat" : 0,
            "size" : 2
        },
        {
            "id" : 3,
            "lon" : 0,
            "lat" : 2e-3,
            "size" : 3
        },
    ]
    stars2 = [
        {
            "id" : 1,
            "lon" : 0,
            "lat" : 0,
            "size" : 1
        },
        {
            "id" : 3,
            "lon" : 0,
            "lat" : 2e-3,
            "size" : 3
        },
        {
            "id" : 2,
            "lon" : 1e-3,
            "lat" : 0,
            "size" : 2
        }
    ]
    descriptors_1 = describe.build_descriptors(stars1, True, None)
    assert len(descriptors_1) == 3
    descriptors_2 = describe.build_descriptors(stars2, True, None)
    assert len(descriptors_2) == 3

    matcher = match.DescriptorMatcher(1, 1e-3, 1e-3, 1e-2)
    matched = matcher.build_match(descriptors_1, descriptors_2)
    assert matched[0] == 0
    assert matched[1] == 2
    assert matched[2] == 1

def test_match_net_1():
    stars1 = [
        {
            "id" : 1,
            "lon" : 0,
            "lat" : 0,
            "size" : 1
        },
        {
            "id" : 2,
            "lon" : 1e-3,
            "lat" : 0,
            "size" : 2
        },
        {
            "id" : 3,
            "lon" : 0,
            "lat" : 2e-3,
            "size" : 3
        },
    ]
    stars2 = [
        {
            "id" : 1,
            "lon" : 0,
            "lat" : 0,
            "size" : 1
        },
        {
            "id" : 3,
            "lon" : 0,
            "lat" : 2e-3,
            "size" : 3
        },
        {
            "id" : 2,
            "lon" : 1e-3,
            "lat" : 0,
            "size" : 2
        }
    ]
    stars3 = [
        {
            "id" : 1,
            "lon" : 0,
            "lat" : 0,
            "size" : 1
        },
        {
            "id" : 2,
            "lon" : 1e-3,
            "lat" : 0,
            "size" : 2
        },
        {
            "id" : 3,
            "lon" : 0,
            "lat" : 2e-3,
            "size" : 3
        }
    ]

    descriptors_1 = describe.build_descriptors(stars1, True, None)
    assert len(descriptors_1) == 3
    descriptors_2 = describe.build_descriptors(stars2, True, None)
    assert len(descriptors_2) == 3
    descriptors_3 = describe.build_descriptors(stars3, True, None)
    assert len(descriptors_3) == 3

    lists = [descriptors_1, descriptors_2, descriptors_3]

    matcher = match.DescriptorMatcher(1, 1e-3, 1e-3, 1e-2)

    match_table = match.build_stars_match_table(matcher, lists)

    assert match_table[0][1][0]==0
    assert match_table[0][1][1]==2
    assert match_table[0][1][2]==1

    assert match_table[0][2][0]==0
    assert match_table[0][2][1]==1
    assert match_table[0][2][2]==2

    assert match_table[1][0][0]==0
    assert match_table[1][0][1]==2
    assert match_table[1][0][2]==1

    assert match_table[1][2][0]==0
    assert match_table[1][2][1]==2
    assert match_table[1][2][2]==1

    assert match_table[2][0][0]==0
    assert match_table[2][0][1]==1
    assert match_table[2][0][2]==2

    assert match_table[2][1][0]==0
    assert match_table[2][1][1]==2
    assert match_table[2][1][2]==1

def assert_has_cluster(clusters, cl):
    for c in clusters:
        if c == cl:
            return
    assert False

def test_cluster_1():
    stars1 = [
        {
            "id" : 1,
            "lon" : 0,
            "lat" : 0,
            "size" : 1
        },
        {
            "id" : 2,
            "lon" : 1e-3,
            "lat" : 0,
            "size" : 2
        },
        {
            "id" : 3,
            "lon" : 0,
            "lat" : 2e-3,
            "size" : 3
        }
    ]
    stars2 = [
        {
            "id" : 1,
            "lon" : 0,
            "lat" : 0,
            "size" : 1
        },
        {
            "id" : 3,
            "lon" : 0,
            "lat" : 2e-3,
            "size" : 3
        },
        {
            "id" : 2,
            "lon" : 1e-3,
            "lat" : 0,
            "size" : 2
        }
    ]
    descriptors_1 = describe.build_descriptors(stars1, True, None)
    assert len(descriptors_1) == 3
    descriptors_2 = describe.build_descriptors(stars2, True, None)
    assert len(descriptors_2) == 3
    lists = [descriptors_1, descriptors_2]

    matcher = match.DescriptorMatcher(1, 1e-3, 1e-3, 1e-2)
    match_table = match.build_stars_match_table(matcher, lists)
    clusters = cluster.find_clusters_in_match_table(match_table)
    assert len(clusters) == 3
    assert_has_cluster(clusters, {0:0, 1:0})
    assert_has_cluster(clusters, {0:1, 1:2})
    assert_has_cluster(clusters, {0:2, 1:1})
