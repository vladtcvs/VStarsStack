"""Common methods"""
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

import os

def listfiles(path, ext=None, recursive=False):
    images = []
    for f in os.listdir(path):
        filename = os.path.abspath(os.path.join(path, f))

        if recursive and os.path.isdir(filename):
            bdname = os.path.basename(filename)
            rimages = listfiles(os.path.join(path, filename), ext, True)
            rimages = [(bdname + "_" + item[0], item[1]) for item in rimages]
            images += rimages

        if not os.path.isfile(filename):
            continue
        if (ext is not None) and (f[-len(ext):].lower() != ext):
            continue

        name = os.path.splitext(f)[0]
        images.append((name, filename))
    images.sort(key=lambda item: item[0])
    return images

def check_dir_exists(filename):
    dirname = os.path.dirname(filename)
    if os.path.isdir(dirname):
        return
    os.makedirs(dirname, exist_ok=True)
