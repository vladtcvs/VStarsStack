#
# Copyright (c) 2022 Vladislav Tsendrovskii
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

import vstarstack.usage
import vstarstack.shift.select_shift
import vstarstack.shift.apply_shift

commands = {
	"select-shift" : (vstarstack.shift.select_shift.run, "Select base image and shift", "shifts.json shift.json"),
	"apply-shift"  : (vstarstack.shift.apply_shift.run, "Apply selected shifts", "shift.json npy/ shifted/"),
}

def run(argv):
	vstarstack.usage.run(argv, "shift", commands, autohelp=True)
