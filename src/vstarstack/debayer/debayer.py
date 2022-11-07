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
import vstarstack.debayer.yuv422
import vstarstack.debayer.bayer

commands = {
	"yuv422" : (vstarstack.debayer.yuv422.run, "Consider RAW as YUV with 422 subsampling"),
    "bayer"  : (vstarstack.debayer.bayer.run, "Consider RAW as Bayer masked image")
}

def run(argv):
	vstarstack.usage.run(argv, "debayer", commands, autohelp=True)
