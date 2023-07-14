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

import numpy as np

from vstarstack.library.loaders.classic import readjpeg
import vstarstack.library.merge
from vstarstack.library.common import ListImageSource

def images_equal(img1, img2, thr = 0.0):
    return np.amax(abs(img1 - img2)) <= thr

def test_simple_add():
    original_image = next(readjpeg("test_image.png"))
    light = original_image.get_channel("L")[0]
    weight = np.ones(light.shape)
    copy1 = original_image.copy()
    copy2 = original_image.copy()
    source = ListImageSource([copy1, copy2])

    summ = vstarstack.library.merge.simple_add(source)
    summ_light, opts = summ.get_channel("L")
    wn = summ.links["weight"]["L"]
    summ_weight = summ.get_channel(wn)[0]
    assert images_equal(summ_light, light*2)
    assert images_equal(summ_weight, weight*2)


def test_simple_mean():
    original_image = next(readjpeg("test_image.png"))
    light = original_image.get_channel("L")[0]
    weight = np.ones(light.shape)
    copy1 = original_image.copy()
    copy2 = original_image.copy()
    source = ListImageSource([copy1, copy2])

    summ = vstarstack.library.merge.simple_mean(source)
    summ_light, opts = summ.get_channel("L")
    wn = summ.links["weight"]["L"]
    summ_weight = summ.get_channel(wn)[0]
    assert images_equal(summ_light, light)
    assert images_equal(summ_weight, weight*2)

def test_kappa_sigma_1():
    original_image = next(readjpeg("test_image.png"))
    light = original_image.get_channel("L")[0]
    weight = np.ones(light.shape)
    copy1 = original_image.copy()
    copy2 = original_image.copy()
    source = ListImageSource([copy1, copy2])

    summ = vstarstack.library.merge.kappa_sigma(source, 1, 1, 0)
    summ_light, opts = summ.get_channel("L")
    wn = summ.links["weight"]["L"]
    summ_weight = summ.get_channel(wn)[0]
    assert images_equal(summ_light, light)
    assert images_equal(summ_weight, weight*2)

def test_kappa_sigma_2():
    original_image = next(readjpeg("test_image.png"))
    light = original_image.get_channel("L")[0]
    weight = np.ones(light.shape)
    copy1 = original_image.copy()
    copy2 = original_image.copy()
    source = ListImageSource([copy1, copy2])

    summ = vstarstack.library.merge.kappa_sigma(source, 1, 1, 1)
    summ_light, opts = summ.get_channel("L")
    wn = summ.links["weight"]["L"]
    summ_weight = summ.get_channel(wn)[0]
    assert images_equal(summ_light, light)
    assert images_equal(summ_weight, weight*2)

def test_kappa_sigma_3():
    original_image = next(readjpeg("test_image.png"))
    light = original_image.get_channel("L")[0]
    weight = np.ones(light.shape)
    copy1 = original_image.copy()
    copy2 = original_image.copy()
    source = ListImageSource([copy1, copy2])

    summ = vstarstack.library.merge.kappa_sigma(source, 1, 1, 2)
    summ_light, opts = summ.get_channel("L")
    wn = summ.links["weight"]["L"]
    summ_weight = summ.get_channel(wn)[0]
    assert images_equal(summ_light, light)
    assert images_equal(summ_weight, weight*2)

def test_kappa_sigma_4():
    peak = 36
    N = 64
    original_image = next(readjpeg("test_image.png"))
    light = original_image.get_channel("L")[0]
    light = light / np.amax(light)
    original_image.replace_channel(light, "L")
    weight = np.ones(light.shape)

    # for repeatability
    np.random.seed(1)
    noised = []
    for _ in range(N):
        copy = original_image.copy()
        copy_light = copy.get_channel("L")[0]
        op = np.amax(copy_light)
        copy_light = np.random.poisson(copy_light / op * peak) / peak * op
        copy.replace_channel(copy_light, "L")
        noised.append(copy)
    source = ListImageSource(noised)

    summ = vstarstack.library.merge.kappa_sigma(source, 3, 3, 5)
    summ_light, opts = summ.get_channel("L")
    wn = summ.links["weight"]["L"]
    summ_weight = summ.get_channel(wn)[0]
    assert images_equal(summ_light, light, 0.1)
    assert images_equal(summ_weight, weight*N, 1)
