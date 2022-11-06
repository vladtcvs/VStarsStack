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

import os.path
import exifread

tags_names = {
	"shutter" : [("EXIF ExposureTime", 0)],
	"iso" : [("EXIF ISOSpeedRatings", 0), ("MakerNote ISOSetting", 1)],
}

def read_tags(filename):
	f = open(filename, 'rb')
	tags = exifread.process_file(f)
	f.close()

#	for key in tags:
#		print(key, tags[key])

	res = {}
	for tn in tags_names:
		for name, id in tags_names[tn]:
			if name in tags:
				res[tn] = float(tags[name].values[id])
				break

	print(res)
	return res

