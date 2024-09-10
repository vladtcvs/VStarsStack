#
# Copyright (c) 2022-2024 Vladislav Tsendrovskii
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
import numpy as np
from setuptools import setup, Extension

clusterization = Extension(name="vstarstack.library.clusters.clusterization",
                           sources=[
                               "src/vstarstack/library/clusters/clusterization/module.cc",
                               "src/vstarstack/library/clusters/clusterization/lib/src/clusters.cc"
                           ],
                           include_dirs=[
                               "src/vstarstack/library/clusters/clusterization/lib/include"
                           ])

projection = Extension( name="vstarstack.library.projection.projections",
                        sources=[
                            "src/vstarstack/library/projection/projections/module.c",
                            "src/vstarstack/library/projection/projections/lib/perspective.c",
                            "src/vstarstack/library/projection/projections/lib/orthographic.c",
                            "src/vstarstack/library/projection/projections/lib/equirectangular.c",
                        ])

movements = Extension(  name="vstarstack.library.movement.movements",
                        sources=[
                            "src/vstarstack/library/movement/movements/module.c",
                            "src/vstarstack/library/movement/movements/lib/sphere.c",
                            "src/vstarstack/library/movement/movements/lib/flat.c",
                        ],
                        include_dirs=[
                            "src/vstarstack/library/projection/projections",
                            np.get_include(),
                        ])

libimagedeform_root = "src/vstarstack/library/fine_movement/libimagedeform"
libimagedeform_headers = [libimagedeform_root + "/include"]
libimagedeform_sources = [libimagedeform_root + "/src/interpolation.c",
                          libimagedeform_root + "/src/image_grid.c",
                          libimagedeform_root + "/src/image_deform.c",
                          libimagedeform_root + "/src/image_deform_gc.c",
                          libimagedeform_root + "/src/image_deform_lc.c",
                        ]

imagedeform_root = "src/vstarstack/library/fine_movement/module"
imagedeform_sources = [imagedeform_root + "/imagegrid.c",
                       imagedeform_root + "/imagedeform.c",
                       imagedeform_root + "/imagedeform_gc.c",
                       imagedeform_root + "/imagedeform_lc.c",
                       imagedeform_root + "/imagedeform_module.c",
                        ]

image_deform = Extension(name="vstarstack.library.fine_movement.module",
       sources=imagedeform_sources+libimagedeform_sources,
       include_dirs=[np.get_include()]+libimagedeform_headers)

root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "src")
result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root)
            for f in filenames if os.path.splitext(f)[1] == '.py']
result = list(set([os.path.dirname(item[len(root)+1:]) for item in result]))
result = [item.replace("/",".") for item in result if "tests" not in item]

print("Packages: ", result)

packages = result

setup (name = 'vstarstack',
       version = '0.2.1',
       author='Vladislav Tsendrovskii',
       description = 'Stacking astrophotos',
       package_dir = {'vstarstack': 'src/vstarstack'},
       packages=packages,
       ext_modules = [projection,
                      movements,
                      image_deform,
                      clusterization,
                    ],
       entry_points = {
           'console_scripts': [
               'vstarstack = vstarstack.entry:main',
           ],
       },
       install_requires = [
              'numpy',
              'astropy',
              'rawpy',
              'pillow',
              'imageio',
              'exifread',
              'opencv-python',
              'scikit-image',
              'scipy >= 1.11.0',
              'imutils',
              'matplotlib',
              'pytz',
       ],
       requires=["setuptools", "cython", "numpy"]
)
