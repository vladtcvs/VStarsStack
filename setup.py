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

from setuptools import setup, Extension, find_packages

import os

perspective = Extension(name="vstarstack.projection.perspective",
       sources=["src/vstarstack/projection/perspective.c"])
image_wave = Extension(name="vstarstack.fine_shift.image_wave",
       sources=["src/vstarstack/fine_shift/image_wave.c"])


root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "src")
result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root) for f in filenames if os.path.splitext(f)[1] == '.py']
result = list(set([os.path.dirname(item[len(root)+1:]) for item in result]))
result = [item.replace("/",".") for item in result if "tests" not in item]

print("Packages: ", result)

packages = result

setup (name = 'vstarstack',
       version = '1.0',
       author='Vladislav Tsendrovskii',
       description = 'Stacking astrophotos',
       scripts = ['bin/vstarstack'],
       package_dir = {'': 'src'},
       packages=packages,
       ext_modules = [perspective, image_wave],
       install_requires = [
              'numpy',
              'astropy',
              'rawpy',
              'pillow',
              'imageio',
              'exifread',
              'opencv-python',
              'scikit-image',
              'scipy',
       ]
)