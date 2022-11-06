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
import readimage.nef
import readimage.classic
import readimage.ser
import readimage.yuv
import readimage.video
import readimage.fits

commands = {
	"nef"   : (readimage.nef.run, "read Nikon NEF"),
	"classic"  : (readimage.classic.run, "read usual images (JPG, PNG, TIFF)"),
	"ser" : (readimage.ser.run, "read SER"),
	"yuv" : (readimage.yuv.run, "read YUV images"),
	"fits" : (readimage.fits.run, "read FITS images"),
	"video" : (readimage.video.run, "read VIDEO images"),
}

def run(argv):
	usage.run(argv, "readimage", commands, autohelp=True)
