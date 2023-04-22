"""Camera description"""
#
# Copyright (c) 2023 Vladislav Tsendrovskii
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

class Camera:
    """Camera description"""

    def __init__(self, config):
        self.w = config["w"]
        self.h = config["h"]
        if "W" in config and "H" in config:
            self.W = config["W"]
            self.H = config["H"]
            self.kw = self.W / self.w
            self.kh = self.H / self.h
        elif "pW" in config and "pH" in config:
            self.kw = config["pW"]*1e-3
            self.kh = config["pH"]*1e-3
            self.W = self.w * self.kw
            self.H = self.h * self.kh
        else:
            raise Exception("Insufficient camera parameters")

        if "encoding_format" in config:
            self.format = config["encoding_format"]
        else:
            self.format = "flat"
