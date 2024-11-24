#
# Copyright (c) 2024 Vladislav Tsendrovskii
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
import vstarstack.library.data
import vstarstack.library.merge.simple_add
import vstarstack.library.image_process.remove_sky
import vstarstack.library.sky_model.gradient
import vstarstack.library.common

def prepare_psf(images : vstarstack.library.common.IImageSource, threshold : float) -> vstarstack.library.data.DataFrame:
    """Prepare PSF from star image"""
    psf = vstarstack.library.merge.simple_add.simple_add(images)
    psf = vstarstack.library.image_process.remove_sky.remove_sky_with_model(psf, vstarstack.library.sky_model.gradient.model)
    for channel in list(psf.get_channels()):
        if psf.get_channel_option(channel, "weight"):
            psf.remove_channel(channel)
            continue
        image, opts = psf.get_channel(channel)
        image -= np.amax(image) * threshold
        image = np.clip(image, 0, None)
        sumv = np.sum(image)
        if sumv > 0:
            image = image / sumv
        psf.replace_channel(image, channel, **opts)
    return psf
