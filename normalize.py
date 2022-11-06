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

import sys
import numpy as np

def normalize(image):
	if image.shape[2] == 3:
		return image
	elif image.shape[2] == 4:
		nums = image[:,:,3]
		nums = np.where(nums == 0, 1, nums)
		image[:,:,0] /= nums
		image[:,:,1] /= nums
		image[:,:,2] /= nums
		image[:,:,3] /= nums
		return image
	else:
		raise Exception("unknown shape")

if __name__ == "__main__":
	img = np.load(sys.argv[1])
	try:
		img = img["arr_0"]
	except:
		pass
	img = normalize(img)
	np.savez_compressed(sys.argv[2], img)

