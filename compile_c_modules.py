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

import numpy as np
from setuptools import Extension
from setuptools.command.build_py import build_py as _build_py

class build_py(_build_py):
    def run(self):
        self.run_command("build_ext")
        return super().run()

    def initialize_options(self):
        super().initialize_options()
        if self.distribution.ext_modules == None:
            self.distribution.ext_modules = []

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
        libimagedeform_headers = [libimagedeform_root + "/include"]
        libimagedeform_sources = [libimagedeform_root + "/src/interpolation.c",
                          libimagedeform_root + "/src/image_grid.c",
                          libimagedeform_root + "/src/image_deform.c",
                          libimagedeform_root + "/src/image_deform_gc.c",
                          libimagedeform_root + "/src/image_deform_lc.c",
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
        include_dirs=[np.get_include()]+libimagedeform_headers)


        self.distribution.ext_modules.append(clusterization)
        self.distribution.ext_modules.append(projection)
        self.distribution.ext_modules.append(movements)
        self.distribution.ext_modules.append(image_deform)
