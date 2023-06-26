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
from vstarstack.library.data import DataFrame

def test_1():
    dataframe = DataFrame()
    assert len(dataframe.params) == 0
    assert len(dataframe.get_channels()) == 0

def test_2():
    dataframe = DataFrame()

    layer = np.zeros((100,100))
    dataframe.add_channel(layer, "data", brightness=True)
    assert len(dataframe.params) == 0
    assert len(dataframe.get_channels()) == 1
    layer, opts = dataframe.get_channel("data")
    assert opts["brightness"] is True

def test_3():
    dataframe = DataFrame()

    layer = np.zeros((100,100))
    dataframe.add_channel(layer, "data", brightness=True)
    assert len(dataframe.params) == 0
    assert len(dataframe.get_channels()) == 1
    dataframe.remove_channel("data")
    assert len(dataframe.get_channels()) == 0

def test_4():
    dataframe = DataFrame()

    layer = np.zeros((100,100))
    dataframe.add_channel(layer, "data", brightness=True)
    assert len(dataframe.params) == 0
    assert len(dataframe.get_channels()) == 1
    dataframe.rename_channel("data", "values")
    assert len(dataframe.get_channels()) == 1
    assert "values" in dataframe.get_channels()

def test_5():
    dataframe = DataFrame()

    layer = np.zeros((100,100))
    weight = np.zeros((100,100))

    dataframe.add_channel(layer, "data", brightness=True)
    dataframe.add_channel(weight, "weight", brightness=True)
    dataframe.add_channel_link("data", "weight", "weight")
    assert len(dataframe.params) == 0
    assert len(dataframe.get_channels()) == 2
    assert dataframe.links["weight"]["data"] == "weight"

def test_6():
    dataframe = DataFrame()

    layer = np.zeros((100,100))
    weight = np.zeros((100,100))

    dataframe.add_channel(layer, "data", brightness=True)
    dataframe.add_channel(weight, "weight", brightness=True)
    dataframe.add_channel_link("data", "weight", "weight")
    assert len(dataframe.params) == 0
    assert len(dataframe.get_channels()) == 2
    assert dataframe.links["weight"]["data"] == "weight"

    dataframe.remove_channel("weight")
    assert len(dataframe.get_channels()) == 1
    assert "data" not in dataframe.links["weight"]

def test_7():
    dataframe = DataFrame()

    layer = np.zeros((100,100))
    weight = np.zeros((100,100))

    dataframe.add_channel(layer, "data", brightness=True)
    dataframe.add_channel(weight, "weight", brightness=True)
    dataframe.add_channel_link("data", "weight", "weight")
    assert len(dataframe.params) == 0
    assert len(dataframe.get_channels()) == 2
    assert dataframe.links["weight"]["data"] == "weight"

    dataframe.rename_channel("weight", "w")
    assert dataframe.links["weight"]["data"] == "w"

def test_8():
    dataframe = DataFrame()

    layer = np.zeros((100,100))
    dataframe.add_channel(layer, "data")
    dataframe.add_parameter(1, "param")
    assert len(dataframe.params) == 1
    assert len(dataframe.get_channels()) == 1
    assert dataframe.params["param"] == 1
