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
from pathlib import Path

use_opencl = True

package_data = {}

clusterization = Extension(name="vstarstack.library.clusters.clusterization",
                           sources=[
                               "src/c_modules/clusterization/module.cc",
                               "src/c_modules/clusterization/lib/src/clusters.cc"
                           ],
                           include_dirs=[
                               "src/c_modules/clusterization/lib/include"
                           ])

projection = Extension( name="vstarstack.library.projection.projections",
                        sources=[
                            "src/c_modules/projections/module.c",
                            "src/c_modules/projections/lib/perspective.c",
                            "src/c_modules/projections/lib/orthographic.c",
                            "src/c_modules/projections/lib/equirectangular.c",
                        ])

movements = Extension(  name="vstarstack.library.movement.movements",
                        sources=[
                            "src/c_modules/movements/module.c",
                            "src/c_modules/movements/lib/sphere.c",
                            "src/c_modules/movements/lib/flat.c",
                        ],
                        include_dirs=[
                            "src/c_modules/projections",
                            np.get_include(),
                        ])

libimagedeform_root = "src/c_modules/fine_movement/libimagedeform"

if use_opencl:
    cl_kernel = libimagedeform_root + "/cl/image_deform_lc.cl"
    if not os.path.exists(cl_kernel):
        print(f"Kernel {cl_kernel} doesn't exist, fail!")
    package_data["vstarstack"] = [cl_kernel]

libimagedeform_headers = [libimagedeform_root + "/include"]
libimagedeform_sources = [libimagedeform_root + "/src/interpolation.c",
                          libimagedeform_root + "/src/image_grid.c",
                          libimagedeform_root + "/src/image_deform.c",
                          libimagedeform_root + "/src/image_deform_gc.c",
                          libimagedeform_root + "/src/image_deform_lc.c",
                          libimagedeform_root + "/src/image_deform_lc_const.c",
                          libimagedeform_root + "/src/image_deform_lc_grid.c"
                        ]

imagedeform_root = "src/c_modules/fine_movement/module"
imagedeform_sources = [imagedeform_root + "/imagegrid.c",
                       imagedeform_root + "/imagedeform.c",
                       imagedeform_root + "/imagedeform_gc.c",
                       imagedeform_root + "/imagedeform_lc.c",
                       imagedeform_root + "/imagedeform_module.c",
                        ]

image_deform = Extension(name="vstarstack.library.fine_movement.module",
                         sources=imagedeform_sources+libimagedeform_sources,
                         include_dirs=[np.get_include()]+libimagedeform_headers,
                         libraries=["OpenCL"] if use_opencl else [],
                         define_macros=[("USE_OPENCL", "1")] if use_opencl else [],
                         )

root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "src")
result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root)
            for f in filenames if os.path.splitext(f)[1] == '.py']
result = list(set([os.path.dirname(item[len(root)+1:]) for item in result]))
result = [item.replace("/",".") for item in result if "tests" not in item]

packages = sorted(result)
print("Packages:")
for package in packages:
    if package in package_data:
        print(package, " : ", package_data[package])
    else:
        print(package)
    
print("-------------------")

setup (name = 'vstarstack',
       version = '0.3.7',
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
              'psutil',
              'photutils',
       ],
       requires=["setuptools", "cython", "numpy"],
       package_data=package_data,
       include_package_data=True
)
