"""Remove continuum from the narrow-band image"""
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

import logging
import numpy as np
import math
import vstarstack.library.stars.detect
from vstarstack.library.photometry.magnitude import star_magnitude_summ_nobg

def calculate_coeff(narrow : np.ndarray, wide : np.ndarray) -> float:
    stars = vstarstack.library.stars.detect.detect_stars(wide)
    max_wide = np.amax(wide)
    max_narrow = np.amax(narrow)
    coeffs = []
    stars = sorted(stars, key=lambda item: item["radius"], reverse=True)[:40]
    logging.info("Stars")
    for star in stars:
        x = int(star["x"]+0.5)
        y = int(star["y"]+0.5)
        r = int(star["radius"]+0.5)
        
        narrow_mag, narrow_numpix, _, narrow_maxv = star_magnitude_summ_nobg(narrow, x, y, r)
        wide_mag, wide_numpix, _, wide_maxv = star_magnitude_summ_nobg(wide, x, y, r)
        if narrow_maxv is None or wide_maxv is None:
            continue
        if wide_maxv > 0.95 * max_wide:
            continue
        if narrow_maxv > 0.95 * max_narrow:
            continue
        k = (narrow_mag / narrow_numpix) / (wide_mag / wide_numpix)
        if math.isnan(k) or k < 0 or k >= 1:
            continue
        logging.info(f"  star {x}:{y}:{r}    k = {k}")
        coeffs.append(k)
    coeff = np.mean(coeffs)
    logging.info(f"Coeff = {coeff}")
    return coeff

def remove_continuum(narrow : np.ndarray, wide : np.ndarray, coeff : float = None):
    if coeff is None:
        coeff = calculate_coeff(narrow, wide)
    return narrow - np.clip(wide, 0, None) * coeff
