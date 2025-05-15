#
# Copyright (c) 2025 Vladislav Tsendrovskii
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

from typing import Tuple
import numpy as np
import cv2
from vstarstack.library.data import DataFrame

def calculate_bg(image : np.ndarray, x : int, y : int, star_r : int, bg_r_big : int, bg_r_small : int) -> dict:
    star_mask = np.zeros(image.shape)
    cv2.circle(star_mask, (x,y), star_r, 1, -1)
    bg_mask = np.zeros(image.shape)
    cv2.circle(bg_mask, (x,y), bg_r_big, 1, -1)
    cv2.circle(bg_mask, (x,y), bg_r_small, 0, -1)

    bg_pixels = image * bg_mask
    bg_sum = np.sum(bg_pixels)
    bg_npix = np.sum(bg_mask)
    bg_mean = bg_sum / bg_npix

    star_pixels = (image - bg_mean) * star_mask
    star_sum = np.sum(star_pixels)
    star_npix = np.sum(star_mask)
    return {
        "star" : star_sum,
        "star_npix" : star_npix,
        "bg" : bg_sum,
        "bg_npix" : bg_npix
    }

def calculate_bgs(image : np.ndarray, stars : list) -> list:
    bgs = []
    for star in stars:
        x = star["x"]
        y = star["y"]
        r = int(star["radius"])
        bg = calculate_bg(image, x, y, r, 2*r, 4*r)
        bgs.append(bg)
    return bgs

def calculate_bgs_df(df : DataFrame, stars : list) -> Tuple[list, set]:
    values = []
    all_channels = set()
    for star in stars:
        channels = {
            "x" : star["x"],
            "y" : star["y"],
            "radius" : star["radius"],
            "channels" : {},
        }
        for channel in df.get_channels():
            if not df.get_channel_option(channel, "brightness"):
                continue
            all_channels.add(channel)
            image, _ = df.get_channel(channel)
            x = star["x"]
            y = star["y"]
            sr = int(star["radius"]+0.5)
            bgr_small = sr * 2
            bgr_big = sr * 3
            bg = calculate_bg(image, x, y, sr, bgr_big, bgr_small)
            channels["channels"][channel] = bg
        values.append(channels)
    return values, all_channels
