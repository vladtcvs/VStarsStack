"""Generate lists for matching"""
#
# Copyright (c) 2024 Vladislav Tsendrovskii
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

import math
import re
import csv

import vstarstack.tool.common
import vstarstack.tool.cfg

def _group_by_video(project: vstarstack.tool.cfg.Project, argv: list[str]):
    regex = re.compile("(.*)_([0-9][0-9][0-9][0-9][0-9][0-9])_keypoints")
    features_path = argv[0]
    matches_between_videos = int(argv[1])
    sequence_length = int(argv[2])
    output = argv[3]

    files = vstarstack.tool.common.listfiles(features_path, ".json")

    with open(output, "w", encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(["image1","image2"])

        groups = {}
        for name, fname in files:
            m = regex.match(name)
            video_name = m.group(1)
            frame_id = m.group(2)
            if video_name not in groups:
                groups[video_name] = []
            groups[video_name].append(frame_id)

        # add records for frames inside same video
        for video_name, frames in groups.items():
            frames = sorted(frames)
            for i in range(len(frames)-1):
                use_frames = frames[i+1:i+1+sequence_length]
                reference_frame = frames[i]
                for frame in use_frames:
                    writer.writerow([video_name + "_" + reference_frame, video_name + "_" + frame])

        # select frames from each video
        selected = {}
        for video_name, frames in groups.items():
            frames = sorted(frames)
            step = math.floor(len(frames) / matches_between_videos)
            step = max(step, 1)
            frames = frames[::step]
            if len(frames) > 0:
                selected[video_name] = frames

        # add records for frames from different videos
        for name1 in selected:
            frames1 = selected[name1]
            for name2 in selected:
                if name2 <= name1:
                    continue
                frames2 = selected[name2]
                for frame1 in frames1:
                    for frame2 in frames2:
                        writer.writerow([name1 + "_" + frame1, name2 + "_" + frame2])

commands = {
    "group_video_frames": (_group_by_video,
                           "Group video frames to sequences and several matches between videos",
                           "features/ <matches between videos> <sequence length> match_list.csv"),
}
