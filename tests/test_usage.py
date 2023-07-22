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
from vstarstack.tool.usage import complete_path_in_dir, autocomplete_files

dir_path = os.path.dirname(os.path.realpath(__file__))

def test_path_autocompletion_1():
    variants = complete_path_in_dir(dir_path, "stars")
    assert len(variants) == 1
    assert variants[0] == ("stars", True)

def test_path_autocompletion_2():
    variants = complete_path_in_dir(dir_path, "s")
    assert len(variants) == 1
    assert variants[0] == ("stars", True)

def test_path_autocompletion_3():
    variants = complete_path_in_dir(os.path.join(dir_path, "stars"), "star_")
    assert len(variants) == 8
    assert variants[0] == ("star_001.png", False)

def test_path_autocompletion_4():
    variants = complete_path_in_dir(os.path.join(dir_path, "stars"), "")
    assert len(variants) == 9
    assert variants[0] == ("star_001.png", False)

def test_path_autocompletion_5():
    variants = autocomplete_files(os.path.join(dir_path, "stars"))
    assert len(variants) == 1
    assert variants[0] == os.path.join(dir_path, "stars/")

def test_path_autocompletion_6():
    variants = autocomplete_files(os.path.join(dir_path, "sta"))
    assert len(variants) == 1
    assert variants[0] == os.path.join(dir_path, "stars/")

def test_path_autocompletion_7():
    variants = autocomplete_files(os.path.join(dir_path, "stars/"))
    assert len(variants) == 9
    assert variants[0] == os.path.join(dir_path, "stars/star_001.png")
