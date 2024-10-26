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

import json
import math

class DarksLibrary:
    """Library for managing darks"""
    def __init__(self, delta_temperature : float):
        self.delta_temp = delta_temperature
        self.darks = []

    def append_dark(self, name : str, exposure : float | None, gain : float | None, temperature : float | None) -> None:
        """Append dark to library"""
        self.darks.append({"exposure":exposure, "gain":gain, "temperature":temperature, "name":name})

    def store(self, fname):
        """Save library to file"""
        with open(fname, "w", encoding="utf8") as f:
            json.dump(self.darks, f)

    def load(self, fname):
        """Load library from file"""
        with open(fname, encoding="utf8") as f:
            self.darks = json.load(f)

    @staticmethod
    def _delta_temperature(dark_temp : float, target_temp : float) -> float:
        if target_temp is None:
            return 0
        if dark_temp is None:
            return math.inf
        return abs(dark_temp - target_temp)

    def find_darks(self, exposure : float | None, gain : float | None, temperature : float | None) -> list:
        """Find list of darks, which match parameters"""
        results = []
        for item in self.darks:
            if exposure != item["exposure"]:
                continue
            if gain != item["gain"]:
                continue
            if temperature is not None and item["temperature"] is not None:
                if abs(temperature - item["temperature"]) > self.delta_temp:
                    continue
            results.append(item)
        results = sorted(results, key=lambda item : self._delta_temperature(item["temperature"], temperature))
        return results
