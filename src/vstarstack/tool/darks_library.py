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

class DarksLibrary:
    """Library for managing darks"""
    def __init__(self, delta_temperature : float | None):
        self.delta_temp = delta_temperature
        self.darks = []

    def append_dark(self, name : str, exposure : float | None, gain : float | None, temperature : float | None) -> None:
        """Append dark to library"""
        self.darks.append({"exposure":exposure, "gain":gain, "temperature":temperature, "name":name})

    def find_darks(self, exposure : float | None, gain : float | None, temperature : float | None) -> list:
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
        return results
