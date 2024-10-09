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

import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

import vstarstack.library.common
import vstarstack.library.image_process
import vstarstack.library.image_process.togray
import vstarstack.tool.cfg
import vstarstack.library.data

def run(project : vstarstack.tool.cfg.Project, argv: list[str]):
    fname = argv[0]
    desc = argv[1]
    df = vstarstack.library.data.DataFrame.load(fname)    
    with open(desc, encoding='utf8') as f:
        detection = json.load(f)
        x = int(detection["object"]["x"]+0.5)
        y = int(detection["object"]["y"]+0.5)
        r = int(detection["object"]["r"]+0.5)

    gray,_ = vstarstack.library.image_process.togray.df_to_gray(df)
    gray = gray / np.amax(gray)
    rgb = np.zeros((gray.shape[0], gray.shape[1], 3))
    rgb[:,:,0] = gray
    rgb[:,:,1] = gray
    rgb[:,:,2] = gray

    cv2.circle(rgb, (x, y), 5, (255, 0, 0))
    cv2.circle(rgb, (x, y), r, (255, 0, 0))
    plt.imshow(rgb)
    plt.show()
    