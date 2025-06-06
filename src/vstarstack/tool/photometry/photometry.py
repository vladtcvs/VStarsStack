#
# Copyright (c) 2024-2025 Vladislav Tsendrovskii
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

commands = {
    "measure-mag" : ("vstarstack.tool.photometry.measure_mag", "measure star magnitude"),
    "measure-fwhm" : ("vstarstack.tool.photometry.measure_fwhm", "measure full width at half maximum"),
    "measure-background" : ("vstarstack.tool.photometry.measure_pollution", "measure background light level"),
}
