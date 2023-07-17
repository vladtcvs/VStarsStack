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

import numpy as np
import os
import csv

import vstarstack.library.loaders.classic
import vstarstack.library.stars.detect as detect

dir_path = os.path.dirname(os.path.realpath(__file__))

def get_expected_position(id):
    with open(os.path.join(dir_path, "stars/stars.csv")) as f:
        reader = csv.reader(f)
        next(reader)
        for cid, x, y in reader:
            if cid == id:
                return int(x), int(y)
    return None, None

def run_test(id):
    detect.configure_detector(max_r=10)
    x, y = get_expected_position(id)
    fname = os.path.join(dir_path, "stars/star_%s.png" % id)
    image = next(vstarstack.library.loaders.classic.readjpeg(fname))
    gray = image.get_channel("R")[0]
    stars = detect.detect_stars(gray)
    assert len(stars) == 1
    assert stars[0]["x"] == x
    assert stars[0]["y"] == y

def test_1():
    run_test("001")

def test_2():
    run_test("002")

def test_3():
    run_test("003")

def test_4():
    run_test("004")

def test_5():
    run_test("005")

def test_6():
    run_test("006")

def test_7():
    run_test("007")

def test_8():
    run_test("008")

def test_9():
    run_test("009")

def test_10():
    run_test("010")

def test_11():
    run_test("011")

#test_1()