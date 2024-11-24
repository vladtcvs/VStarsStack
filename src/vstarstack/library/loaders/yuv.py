"""Read YUV video to npy frames"""
#
# Copyright (c) 2022-2024 Vladislav Tsendrovskii
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

import logging
import numpy as np

import vstarstack.library.data
logger = logging.getLogger(__name__)

def readyuv(fname: str, width: int, height: int):
    """Read YUV file"""
    frame_len = int(width*height*2)
    shape = (int(height*2), width)

    with open(fname, "rb") as files:
        tags = {}

        params = {
            "w": width,
            "h": height,
            "format" : "yuv_422",
            "exposure" : 1,
            "gain" : 1,
            "weight" : 1,
        }

        frame_id = 0
        while True:
            frame = files.read(frame_len)
            if not frame:
                break
            yuv = np.frombuffer(frame, dtype=np.uint8)

            yuv = yuv.reshape(shape)
            logger.info(f"processing frame {frame_id}")

            dataframe = vstarstack.library.data.DataFrame(params, tags)
            dataframe.add_channel(yuv, "raw", encoded=True, signal=True)
            yield dataframe
            frame_id += 1
