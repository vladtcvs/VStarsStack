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

import usage
import debayer.yuv422
import debayer.bayer

commands = {
	"yuv422" : (debayer.yuv422.run, "Consider RAW as YUV with 422 subsampling"),
    "bayer"  : (debayer.bayer.run, "Consider RAW as Bayer masked image")
}

def run(argv):
	usage.run(argv, "debayer", commands, autohelp=True)
