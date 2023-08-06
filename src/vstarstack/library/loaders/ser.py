"""Read SER images"""
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

import sys
import math
import numpy as np

import vstarstack.library.data

def _serread(file, integer_size, little_endian):
    """Read integer from SER file"""
    block = list(file.read(integer_size))
    if little_endian:
        block = block[::-1]
    val = 0
    for i in range(integer_size):
        val *= 256
        val += block[i]
    return val


def _serread4(file):
    """Read 4-byte integer"""
    return _serread(file, 4, True)

def _read_to_npy(file, bpp, little_endian, shape):
    """Read block of SER file"""
    num = 1
    for _, dim in enumerate(shape):
        num *= dim
    num_b = bpp * num
    block = np.array(list(file.read(num_b)), dtype=np.uint32)
    block = block.reshape((num, bpp))
    for i in range(bpp):
        if not little_endian:
            block[:, i] *= 2**(8*i)
        else:
            block[:, i] *= 2**(8*(bpp-i))
    block = np.sum(block, axis=1)
    block = block.reshape(shape)
    return block


def readser(fname: str):
    """Read SER file"""
    with open(fname, "rb") as file:
        fileid = file.read(14)
        if fileid != b'LUCAM-RECORDER':
            print(f"Possibly invalid header {fileid.decode('utf8')}", file=sys.stderr)

        _luid = _serread4(file)
        colorid = _serread4(file)
        le16bit = _serread4(file)
        width = _serread4(file)
        height = _serread4(file)
        depth = _serread4(file)
        bpp = (int)(math.ceil(depth / 8))
        frames = _serread4(file)
        observer = file.read(40).decode('utf8')
        instrume = file.read(40).decode('utf8')
        telescope = file.read(40).decode('utf8')
        datetime = _serread(file, 8, True)
        datetime_utc = _serread(file, 8, True)
        opts = {}
        if colorid == 0:
            shape = (height, width, 1)
            channels = ["L"]        # luminocity
            image_format = "flat"
            opts["brightness"] = True
        elif colorid == 8:
            shape = (height, width, 1)
            channels = ["raw"]
            image_format = "bayerRGGB"
            opts["encoded"] = True
        elif colorid == 9:
            shape = (height, width, 1)
            channels = ["raw"]
            image_format = "bayerGRBG"
            opts["encoded"] = True
        elif colorid == 10:
            shape = (height, width, 1)
            channels = ["raw"]
            image_format = "bayerGBRG"
            opts["encoded"] = True
        elif colorid == 11:
            shape = (height, width, 1)
            channels = ["raw"]
            image_format = "bayerBGGR"
            opts["encoded"] = True
        elif colorid == 16:
            shape = (height, width, 1)
            channels = ["raw"]
            image_format = "bayerCYYM"
            opts["encoded"] = True
        elif colorid == 17:
            shape = (height, width, 1)
            channels = ["raw"]
            image_format = "bayerYCMY"
            opts["encoded"] = True
        elif colorid == 18:
            shape = (height, width, 1)
            channels = ["raw"]
            image_format = "bayerYMCY"
            opts["encoded"] = True
        elif colorid == 19:
            shape = (height, width, 1)
            channels = ["raw"]
            image_format = "bayerMYYC"
            opts["encoded"] = True
        elif colorid == 100:
            shape = (height, width, 3)
            channels = ["R", "G", "B"]
            image_format = "flat"
            opts["brightness"] = True
        elif colorid == 101:
            shape = (height, width, 3)
            channels = ["B", "G", "R"]
            image_format = "flat"
            opts["brightness"] = True
        else:
            print(f"Unsupported colorid = {colorid}")
            return
        tags = {
            "depth": depth,
            "observer": observer,
            "instrument": instrume,
            "telescope": telescope,
            "dateTime": datetime,
            "dateTimeUTC": datetime_utc,
        }

        params = {
            "w": width,
            "h": height,
            "format" : image_format,
        }

        for frame_id in range(frames):
            print(f"\tprocessing frame {frame_id}")
            frame = _read_to_npy(file, bpp, le16bit, shape)
            dataframe = vstarstack.library.data.DataFrame(params, tags)
            exptime = 1
            weight = np.ones(frame.data.shape[0:2])*exptime
            index = 0
            for index, channel in enumerate(channels):
                dataframe.add_channel(frame[:, :, index], channel, **opts)
                dataframe.add_channel(weight, "weight-"+channel, weight=True)
                dataframe.add_channel_link(
                    channel, "weight-"+channel, "weight")
            yield dataframe
