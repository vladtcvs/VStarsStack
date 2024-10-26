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

        tags = {}

        params = {
            "w": frame.shape[1],
            "h": frame.shape[0],
            "exposure" : 1,
            "gain" : 1,
            "weight" : 1,
        }

        dataframe = vstarstack.library.data.DataFrame(params, tags)
        dataframe.add_channel(frame[:, :, 0], "R", brightness=True, signal=True)
        dataframe.add_channel(frame[:, :, 1], "G", brightness=True, signal=True)
        dataframe.add_channel(frame[:, :, 2], "B", brightness=True, signal=True)
        yield dataframe
        frame_id += 1
