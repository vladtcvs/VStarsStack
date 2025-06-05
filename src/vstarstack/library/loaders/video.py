"""Read video source file"""
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

import cv2
import numpy as np

import vstarstack.library.data
from vstarstack.library.loaders.datatype import check_datatype

def read_video(fname: str):
    """Read frames from video file"""
    vidcap = cv2.VideoCapture(fname)
    frame_id = 0

    # vidcap.set(cv2.CAP_PROP_FORMAT, -1)
    while True:
        success, frame = vidcap.read()
        if not success:
            break

        tags = {}

        params = {
            "w": frame.shape[1],
            "h": frame.shape[0],
            "exposure" : 1,
            "gain" : 1,
            "weight" : 1,
        }

        max_value = np.iinfo(frame.dtype).max
        dataframe = vstarstack.library.data.DataFrame(params, tags)
        for channel_name, channel_index in [("R",0), ("G", 1), ("B", 2)]:
            data = frame[:,:,channel_index]
            dataframe.add_channel(check_datatype(data), channel_name, brightness=True, signal=True)
            overlight_idx = np.where(data >= max_value*0.99)
            if len(overlight_idx) > 0:
                weight = np.ones(data.shape)*params["weight"]
                weight[overlight_idx] = 0
                dataframe.add_channel(weight, f"weight-{channel_name}", weight=True)
                dataframe.add_channel_link(channel_name, f"weight-{channel_name}", "weight")

        yield dataframe
        frame_id += 1
