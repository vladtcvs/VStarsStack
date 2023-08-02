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

import numpy as np

import vstarstack.library.data


def _indexes(h, w, dy, dx, sy, sx):
    return range(sy, h+sy, dy), range(sx, w+sx, dx)

def yuv_422_image(frame, weight):
    """Split raw YUV image to L, Cb, Cr"""
    frame = frame.astype(np.float32)
    h = frame.shape[0]
    w = frame.shape[1]

    Y1y, Y1x = _indexes(h, w, 2, 2, 0, 0)
    Y1 = frame[:, Y1x]
    Y1 = Y1[Y1y, :]

    Y2y, Y2x = _indexes(h, w, 2, 2, 1, 0)
    Y2 = frame[:, Y2x]
    Y2 = Y2[Y2y, :]

    Cb1y, Cb1x = _indexes(h, w, 2, 4, 0, 1)
    Cb1 = frame[:, Cb1x]
    Cb1 = Cb1[Cb1y, :]

    Cb2y, Cb2x = _indexes(h, w, 2, 4, 1, 1)
    Cb2 = frame[:, Cb2x]
    Cb2 = Cb2[Cb2y, :]

    Cr1y, Cr1x = _indexes(h, w, 2, 4, 0, 3)
    Cr1 = frame[:, Cr1x]
    Cr1 = Cr1[Cr1y, :]

    Cr2y, Cr2x = _indexes(h, w, 2, 4, 1, 3)
    Cr2 = frame[:, Cr2x]
    Cr2 = Cr2[Cr2y, :]

    Y = np.concatenate((Y1, Y2), axis=1)
    Cb = np.concatenate((Cb1, Cb2), axis=1)
    Cr = np.concatenate((Cr1, Cr2), axis=1)

    Cb = np.repeat(Cb, axis=1, repeats=2)
    Cr = np.repeat(Cr, axis=1, repeats=2)

    Cb = (Cb.astype(np.float32) - 128) / Y
    Cr = (Cr.astype(np.float32) - 128) / Y

    w_Y1 = weight[:, Y1x]
    w_Y1 = w_Y1[Y1y, :]
    w_Y2 = weight[:, Y2x]
    w_Y2 = w_Y2[Y2y, :]
    w_Y = np.concatenate((w_Y1, w_Y2), axis=1)

    return Y, Cb, Cr, w_Y

def yuv_422_dataframe(dataframe : vstarstack.library.data.DataFrame,
                      raw_channel_name : str):
    """Split dataframe with raw YUV image to L, Cb, Cr"""
    raw, _ = dataframe.get_channel(raw_channel_name)
    weight, _ = dataframe.get_channel(dataframe.links["weight"][raw_channel_name])

    Y, Cb, Cr, w_Y = yuv_422_image(raw, weight)

    dataframe.add_channel(Y, "L", brightness=True)
    dataframe.add_channel(Cb, "Cb")
    dataframe.add_channel(Cr, "Cr")

    dataframe.add_channel(w_Y, "weight-L", weight=True)
    dataframe.add_channel_link("L", "weight-L", "weight")

    dataframe.params["format"] = "flat"
    dataframe.remove_channel(dataframe.links["weight"][raw_channel_name])
    dataframe.remove_channel(raw_channel_name)
    return dataframe
