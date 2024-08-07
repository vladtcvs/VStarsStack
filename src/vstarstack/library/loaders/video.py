"""Read video source file"""
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
import cv2

import vstarstack.library.data

def read_video(fname: str):
    """Read frames from video file"""
    vidcap = cv2.VideoCapture(fname)
    frame_id = 0

    # vidcap.set(cv2.CAP_PROP_FORMAT, -1)
    while True:
        success, frame = vidcap.read()
        if not success:
            break

        tags = {
            "depth": 8,
        }

        params = {
            "w": frame.shape[1],
            "h": frame.shape[0],
        }

        exptime = 1
        weight = np.ones((frame.shape[0], frame.shape[1]))*exptime

        dataframe = vstarstack.library.data.DataFrame(params, tags)
        dataframe.add_channel(frame[:, :, 0], "R", brightness=True, signal=True)
        dataframe.add_channel(frame[:, :, 1], "G", brightness=True, signal=True)
        dataframe.add_channel(frame[:, :, 2], "B", brightness=True, signal=True)
        dataframe.add_channel(weight, "weight", weight=True)
        dataframe.add_channel_link("R", "weight", "weight")
        dataframe.add_channel_link("G", "weight", "weight")
        dataframe.add_channel_link("B", "weight", "weight")
        yield dataframe
        frame_id += 1
