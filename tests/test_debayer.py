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
import vstarstack.library.debayer.bayer
from vstarstack.library.data import DataFrame
from vstarstack.library.debayer.bayer import DebayerMethod

def EQ(a, b):
    return abs(a-b) < 1e-12

def test_1():
    layer = np.ones((16, 16))
    layer[0::2,0::2] = 0.5 # R
    layer[0::2,1::2] = 0.1 # G
    layer[1::2,0::2] = 0.3 # G
    layer[1::2,1::2] = 1.0 # B
    mask = vstarstack.library.debayer.bayer.generate_mask("RGGB")
    df = DataFrame(params = {"weight":1, "w" : 16, "h" : 16})
    df.add_channel(layer, "raw", brightness=True, signal=True, normed=False)
    result = vstarstack.library.debayer.bayer.debayer_dataframe(df, mask, "raw", DebayerMethod.SUBSAMPLE)

    assert "R" in result.get_channels()
    assert "G" in result.get_channels()
    assert "B" in result.get_channels()

    assert result.get_channel_option("R", "normed") == False
    assert result.get_channel_option("G", "normed") == False
    assert result.get_channel_option("B", "normed") == False

    R,_ = result.get_channel("R")
    assert R.shape == (8, 8)
    assert R[0,0] == 0.5
    G,_ = result.get_channel("G")
    assert G.shape == (8, 8)
    assert G[0,0] == 0.4
    B,_ = result.get_channel("B")
    assert B.shape == (8, 8)
    assert B[0,0] == 1.0
    wR,_,_ = result.get_linked_channel("R", "weight")
    assert wR.shape == (8, 8)
    assert wR[0,0] == 1
    wG,_,_ = result.get_linked_channel("G", "weight")
    assert wG.shape == (8, 8)
    assert wG[0,0] == 2
    wB,_,_ = result.get_linked_channel("B", "weight")
    assert wB.shape == (8, 8)
    assert wB[0,0] == 1

def test_2():
    layer = np.ones((16, 16))
    layer[0::2,0::2] = 0.5 # R
    layer[0::2,1::2] = 0.1 # G
    layer[1::2,0::2] = 0.3 # G
    layer[1::2,1::2] = 1.0 # B
    mask = vstarstack.library.debayer.bayer.generate_mask("RGGB")
    df = DataFrame(params = {"weight":1, "w" : 16, "h" : 16})
    df.add_channel(layer, "raw", brightness=True, signal=True, normed=False)
    result = vstarstack.library.debayer.bayer.debayer_dataframe(df, mask, "raw", DebayerMethod.MASK)

    assert "R" in result.get_channels()
    assert "G" in result.get_channels()
    assert "B" in result.get_channels()

    assert result.get_channel_option("R", "normed") == False
    assert result.get_channel_option("G", "normed") == False
    assert result.get_channel_option("B", "normed") == False

    R,_ = result.get_channel("R")
    wR,_,_ = result.get_linked_channel("R", "weight")
    assert R.shape == (16, 16)
    assert wR.shape == (16, 16)

    assert R[0,0] == 0.5
    assert wR[0,0] == 1
    assert R[0,1] == 0
    assert wR[0,1] == 0
    assert R[1,0] == 0
    assert wR[1,0] == 0
    assert R[1,1] == 0
    assert wR[1,1] == 0

    G,_ = result.get_channel("G")
    wG,_,_ = result.get_linked_channel("G", "weight")
    assert G.shape == (16, 16)
    assert wG.shape == (16, 16)

    assert G[0,0] == 0
    assert wG[0,0] == 0
    assert G[0,1] == 0.1
    assert wG[0,1] == 1
    assert G[1,0] == 0.3
    assert wG[1,0] == 1
    assert G[1,1] == 0
    assert wG[1,1] == 0

    B,_ = result.get_channel("B")
    wB,_,_ = result.get_linked_channel("B", "weight")
    assert B.shape == (16, 16)
    assert wB.shape == (16, 16)

    assert B[0,0] == 0    
    assert wB[0,0] == 0
    assert B[0,1] == 0    
    assert wB[0,1] == 0
    assert B[1,0] == 0    
    assert wB[1,0] == 0
    assert B[1,1] == 1.0    
    assert wB[1,1] == 1

def test_3():
    layer = np.ones((16, 16))
    layer[0::2,0::2] = 0.5 # R
    layer[0::2,1::2] = 0.1 # G
    layer[1::2,0::2] = 0.3 # G
    layer[1::2,1::2] = 1.0 # B
    mask = vstarstack.library.debayer.bayer.generate_mask("RGGB")
    df = DataFrame(params = {"weight":1, "w" : 16, "h" : 16})
    df.add_channel(layer, "raw", brightness=True, signal=True, normed=False)
    result = vstarstack.library.debayer.bayer.debayer_dataframe(df, mask, "raw", DebayerMethod.INTERPOLATE)

    assert "R" in result.get_channels()
    assert "G" in result.get_channels()
    assert "B" in result.get_channels()

    assert result.get_channel_option("R", "normed") == False
    assert result.get_channel_option("G", "normed") == False
    assert result.get_channel_option("B", "normed") == False

    R,_ = result.get_channel("R")
    wR,_,_ = result.get_linked_channel("R", "weight")
    assert R.shape == (16, 16)
    assert wR.shape == (16, 16)

    assert EQ(R[0,0], 0.5)
    assert EQ(wR[0,0], 1)
    assert EQ(R[0,1], 1)
    assert EQ(wR[0,1], 2)
    assert EQ(R[1,0], 1)
    assert EQ(wR[1,0], 2)
    assert EQ(R[1,1], 2)
    assert EQ(wR[1,1], 4)

    G,_ = result.get_channel("G")
    wG,_,_ = result.get_linked_channel("G", "weight")
    assert G.shape == (16, 16)
    assert wG.shape == (16, 16)

    assert EQ(G[0,0], 0.4)
    assert EQ(wG[0,0], 2)
    assert EQ(G[0,1], 0.1)
    assert EQ(wG[0,1],1)
    assert EQ(G[1,0], 0.3)
    assert EQ(wG[1,0], 1)
    assert EQ(G[1,1], 0.8)
    assert EQ(wG[1,1], 4)

    B,_ = result.get_channel("B")
    wB,_,_ = result.get_linked_channel("B", "weight")
    assert B.shape == (16, 16)
    assert wB.shape == (16, 16)

    assert EQ(B[0,0], 1)
    assert EQ(wB[0,0], 1)
    assert EQ(B[0,1], 1)
    assert EQ(wB[0,1], 1)
    assert EQ(B[1,0], 1)
    assert EQ(wB[1,0], 1)
    assert EQ(B[1,1], 1)
    assert EQ(wB[1,1], 1)
