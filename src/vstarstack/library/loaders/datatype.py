"""convert 8 and 16 bit data to 32 bit"""
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
def check_datatype(data : np.ndarray) -> np.ndarray:
    """If 8 bit or 16 bit convert to 32 bit"""
    if data.dtype == np.int8 or data.dtype == np.int16:
        return data.astype(np.int32)
    if data.dtype == np.uint8 or data.dtype == np.uint16:
        return data.astype(np.uint32)
    return data
