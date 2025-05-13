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

import numpy as np
import vstarstack.library.stars.detect
from vstarstack.library.photometry.magnitude import star_magnitude_summ_nobg

def remove_continuum(narrow : np.ndarray, wide : np.ndarray):
    stars = vstarstack.library.stars.detect.detect_stars(narrow)
    max_wide = np.amax(wide)
    max_narrow = np.amax(narrow)
    coeffs = []
    for star in stars:
        x = int(star["x"]+0.5)
        y = int(star["y"]+0.5)
        r = int(star["radius"]+0.5)
        narrow_mag, narrow_numpix, narrow_maxv = star_magnitude_summ_nobg(narrow, x, y, r)
        wide_mag, wide_numpix, wide_maxv = star_magnitude_summ_nobg(narrow, x, y, r)
        if wide_maxv > 0.95 * max_wide:
            continue
        if narrow_maxv > 0.95 * max_narrow:
            continue
        k = (narrow_mag / narrow_numpix) / (wide_mag / wide_numpix)
        coeffs.append(k)
    coeff = np.mean(coeffs)
    return narrow - wide * coeff
