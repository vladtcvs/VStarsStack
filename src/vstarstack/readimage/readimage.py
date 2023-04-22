"""Read source image files to internal format"""
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

import vstarstack.cfg

import vstarstack.usage
import vstarstack.readimage.nef
import vstarstack.readimage.classic
import vstarstack.readimage.ser
import vstarstack.readimage.yuv
import vstarstack.readimage.video
import vstarstack.readimage.fits

commands = {
    "nef": (vstarstack.readimage.nef.run, "read Nikon NEF"),
    "classic": (vstarstack.readimage.classic.run, "read usual images (JPG, PNG, TIFF)"),
    "ser": (vstarstack.readimage.ser.run, "read SER"),
    "yuv": (vstarstack.readimage.yuv.run, "read YUV images"),
    "fits": (vstarstack.readimage.fits.run, "read FITS images"),
    "video": (vstarstack.readimage.video.run, "read VIDEO images"),
}


def run(project: vstarstack.cfg.Project, argv: list):
    """Run reading image files"""
    vstarstack.usage.run(project, argv, "readimage", commands, autohelp=True)
