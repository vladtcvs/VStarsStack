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

from vstarstack.tool.configuration import Configuration

def test_1():
    decl = {
        "int_value" : (int, 1),
        "float_value" : (float, 2)
    }
    cfg = Configuration(decl)
    assert cfg.int_value == 1
    assert cfg.float_value == 2.0

def test_2():
    decl = {
        "int_value" : (int, 1),
        "float_value" : (float, 2),
        "submodule" : {
            "subvalue" : (int, 3)
        }
    }
    cfg = Configuration(decl)
    assert cfg.int_value == 1
    assert cfg.float_value == 2.0
    assert cfg.submodule.subvalue == 3

def test_3():
    decl = {
        "int_value" : (int, 1),
        "float_value" : (float, 2),
        "submodule" : ("module", {
            "subvalue" : (int, 3)
        })
    }
    cfg = Configuration(decl)
    assert cfg.int_value == 1
    assert cfg.float_value == 2.0
    assert cfg.submodule.subvalue == 3

def test_4():
    decl = {
        "int_value" : (int, 1),
        "float_value" : (float, 2),
        "use_modules" : (list, ["submodule"]),
        "submodule" : {
            "subvalue" : (int, 3)
        },
        "submodule2" : {
            "subvalue" : (int, 3)
        },
    }
    cfg = Configuration(decl)
    assert cfg.int_value == 1
    assert cfg.float_value == 2.0
    assert cfg.submodule.subvalue == 3
    assert cfg.submodule2.subvalue == 3
    assert len(cfg.used_modules) == 1
    assert cfg.used_modules[0] == "submodule"

def test_5():
    subdecl = {
        "value" : (float, 8.0),
    }
    subcfg = Configuration(subdecl)
    decl = {
        "int_value" : (int, 1),
        "float_value" : (float, 2),
        "use_modules" : (list, ["submodule"]),
        "submodule" : ("module", subcfg),
    }
    cfg = Configuration(decl)
    assert cfg.int_value == 1
    assert cfg.float_value == 2.0
    assert cfg.submodule.value == 8.0

def test_6():
    decl = {
        "int_value" : (int, 1),
        "float_value" : (float, 2)
    }
    cfg = Configuration(decl)
    assert cfg.int_value == 1
    assert cfg.float_value == 2.0
    data = cfg.write_configuration()
    assert data["int_value"] == 1
    assert data["float_value"] == 2.0

def test_7():
    decl = {
        "int_value" : (int, 1),
        "float_value" : (float, 2),
        "submodule" : {
            "value" : (int, 2)
        }
    }
    cfg = Configuration(decl)
    data = cfg.write_configuration()
    assert data["int_value"] == 1
    assert data["float_value"] == 2.0
    assert data["submodule"]["value"] == 2

def test_8():
    decl = {
        "int_value" : (int, 1),
        "float_value" : (float, 2),
        "use_modules" : (list, ["submodule"]),
        "submodule" : ("module", {
            "value" : (int, 2)
        }),
        "submodule2" : ("module", {
            "value" : (int, 3)
        }),
    }
    cfg = Configuration(decl)
    data = cfg.write_configuration()
    assert data["int_value"] == 1
    assert data["float_value"] == 2.0
    assert data["submodule"]["value"] == 2
    assert "submodule2" not in data

def test_9():
    decl = {
        "int_value" : (int, 1),
        "float_value" : (float, 2)
    }
    cfg = Configuration(decl)

    data = {
        "int_value" : 4,
        "float_value" : 3.0,
    }
    cfg.load_configuration(data)
    assert cfg.int_value == 4
    assert cfg.float_value == 3.0

def test_10():
    decl = {
        "int_value" : (int, 1),
        "float_value" : (float, 2),
        "submodule" : ("module", {
            "value" : (int, 2)
        }),
    }
    data = {
        "int_value" : 4,
        "float_value" : 3.0,
        "submodule" : {
            "value" : 3
        }
    }

    cfg = Configuration(decl)
    cfg.load_configuration(data)
    assert cfg.int_value == 4
    assert cfg.float_value == 3.0
    assert cfg.submodule.value == 3
    assert len(cfg.used_modules) == 1
    assert cfg.used_modules[0] == "submodule"
