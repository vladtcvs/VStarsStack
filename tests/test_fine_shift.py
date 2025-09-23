#
# Copyright (c) 2023-2024 Vladislav Tsendrovskii
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

import math
import sys
import os
import numpy as np
import gc
import psutil

from vstarstack.library.loaders.classic import readjpeg
from vstarstack.library.fine_movement.module import ImageGrid
from vstarstack.library.fine_movement.module import ImageDeform
from vstarstack.library.fine_movement.module import ImageDeformGC
from vstarstack.library.fine_movement.module import ImageDeformLC

dir_path = os.path.dirname(os.path.realpath(__file__))
N = 200
dh = 0.01

kernel_path = "/home/vlad/Astronomy/software/VStarStack/src/c_modules/fine_movement/libimagedeform/cl/image_deform_lc.cl"

def test_identity():
    deform = ImageDeform(image_w=10, image_h=10,
                         grid_w=2, grid_h=2)
    
    x, y = deform.apply_point(2, 0)
    assert x == 2 and y == 0
    x, y = deform.apply_point(9, 5)
    assert x == 9 and y == 5
    x, y = deform.apply_point(10, 5)
    assert math.isnan(x)
    assert math.isnan(y)

def test_approximate_identity():
    gc = ImageDeformGC(image_w=10, image_h=10,
                           grid_w=2, grid_h=2,
                           spk=0.01)
    expected = np.array([[5.0, 5.0]], dtype=np.float32)
    points = np.array([[5.0, 5.0]], dtype=np.float32)
    deform = gc.find(points=points, expected_points=expected, dh=dh, Nsteps=N)
    x, y = deform.apply_point(5.5, 5)

    assert abs(x-5.5) < 5e-3
    assert abs(y-5) < 5e-3


def test_approximate_single():
    gc = ImageDeformGC(image_w=10, image_h=10,
                       grid_w=2, grid_h=2,
                       spk=0.01)
    points = np.array([(5.0, 5.2)], dtype=np.float32) # [(y,x), (y,x), ...]
    expected = np.array([(5.0, 5.0)], dtype=np.float32) # [(y,x), (y,x), ...]
    deform = gc.find(points=points, expected_points=expected, dh=dh, Nsteps=0)
    assert deform is not None
    x, y = deform.apply_point(5.2, 5.0)

    assert abs(x-5.0) < 5e-3
    assert abs(y-5.0) < 5e-3

def test_approximate_square():
    gc = ImageDeformGC(image_w=10, image_h=10,
                       grid_w=2, grid_h=2,
                       spk=0.01)
    points = np.array([(0.1, 0.1), (0.1, 8.9), (8.9, 0.1), (8.9, 8.9)], dtype=np.float32) # [(y,x), (y,x), ...]
    expected = np.array([(0.0, 0.0), (0.0, 9.0), (9.0, 0.0), (9.0, 9.0)], dtype=np.float32) # [(y,x), (y,x), ...]
    deform = gc.find(points=points, expected_points=expected, dh=dh, Nsteps=N)
    assert deform is not None

    x, y = deform.apply_point(4.5, 4.5)
    assert abs(x-4.5) < 5e-3
    assert abs(y-4.5) < 5e-3

    x, y = deform.apply_point(8.9, 0.1)
    assert abs(x-9) < 5e-3
    assert abs(y-0) < 5e-3


def test_approximate_parabola1():
    gc = ImageDeformGC(image_w=11, image_h=11,
                       grid_w=3, grid_h=3,
                       spk=0.01)

    expected = np.array([(5, 0), (5, 5), (5, 10)], dtype=np.float32).astype(np.float32) # [(y,x),...]
    points = np.array([(5, 0), (5, 5.1), (5, 10)], dtype=np.float32).astype(np.float32) # [(y,x),...]
    deform = gc.find(points=points, expected_points=expected, dh=dh, Nsteps=N)
    assert deform is not None

    x, y = deform.apply_point(5.1, 5)

    assert abs(x-5) < 5e-3
    assert abs(y-5) < 5e-3


def test_approximate_parabola2():
    gc = ImageDeformGC(image_w=11, image_h=11,
                       grid_w=3, grid_h=3,
                       spk=0.01)

    expected = np.array([(5, 0), (5, 5), (5, 10)], dtype=np.float32) # [(y,x),...]
    points = np.array([(5, 0), (5.1, 5.1), (5, 10)], dtype=np.float32) # [(y,x),...]
    deform = gc.find(points=points, expected_points=expected, dh=dh, Nsteps=N)
    assert deform is not None

    x, y = deform.apply_point(5.1, 5.1)

    assert abs(x-5) < 5e-3
    assert abs(y-5) < 5e-3

def test_approximate_parabola_long():
    gc = ImageDeformGC(image_w=11, image_h=11,
                       grid_w=7, grid_h=7,
                       spk=0.1)

    expected = np.array([(5, 0), (5, 5), (5, 10)], dtype=np.float32) # [(y,x),...]
    points = np.array([(5, 0), (5, 5.1), (5, 10)], dtype=np.float32) # [(y,x),...]
    deform = gc.find(points=points, expected_points=expected, dh=dh, Nsteps=N)
    assert deform is not None

    x, y = deform.apply_point(5.1, 5)

    assert abs(x-5) < 5e-3
    assert abs(y-5) < 5e-3

def test_deserialize():
    gc = ImageDeformGC(image_w=11, image_h=11,
                       grid_w=3, grid_h=3,
                       spk=0.001)
    expected = np.array([[5.0, 5.0]], dtype=np.float32)
    points = np.array([[5.0, 5.2]], dtype=np.float32)
    deform = gc.find(points=points, expected_points=expected, dh=dh, Nsteps=N)
    
    x, y = deform.apply_point(5.2, 5.0)

    assert abs(x-5.0) < 5e-3
    assert abs(y-5.0) < 5e-3

    data = deform.content()
    assert data is not None
    assert data.shape[0] == 3
    assert data.shape[1] == 3
    assert data.shape[2] == 2

    deform2 = ImageDeform(image_w=11, image_h=11,
                          grid_w=3, grid_h=3)
    deform2.fill(shift_array=data)

    x, y = deform2.apply_point(5.2, 5.0)

    assert abs(x-5.0) < 5e-3
    assert abs(y-5.0) < 5e-3

def compare_shift_array(array, reference):
    assert len(array) == len(reference)
    print(array, reference)
    for i, v in enumerate(reference):
        assert abs(v - array[i]) < 1e-3

def test_correlation1():
    df1 = next(readjpeg(os.path.join(dir_path, "fine_shift/image1.png")))
    image1 = df1.get_channel("L")[0].astype(np.float32)
    grid = ImageGrid(image_w=image1.shape[1], image_h=image1.shape[0])
    grid.fill(image1)
    correlation = ImageGrid.correlation(grid, grid)
    assert correlation == 1

def test_correlation2():
    df1 = next(readjpeg(os.path.join(dir_path, "fine_shift/image1.png")))
    df2 = next(readjpeg(os.path.join(dir_path, "fine_shift/image2.png")))

    image1 = df1.get_channel("L")[0].astype(np.float32)
    image2 = df2.get_channel("L")[0].astype(np.float32)

    image1_moved = np.zeros(image1.shape, dtype=np.float32)
    for y in range(image1.shape[0]):
        for x in range(1,image1.shape[1]):
            image1_moved[y,x] = image1[y,x-1]
        image1_moved[y,0] = np.nan

    grid1 = ImageGrid(image_w=image1_moved.shape[1], image_h=image1_moved.shape[0])
    grid1.fill(image1_moved)
    grid2 = ImageGrid(image_w=image2.shape[1], image_h=image2.shape[0])
    grid2.fill(image1_moved)

    correlation = ImageGrid.correlation(grid1, grid2)
    assert correlation == 1

def test_shift_image1():
    df1 = next(readjpeg(os.path.join(dir_path, "fine_shift/image1.png")))

    image1 = df1.get_channel("L")[0].astype(np.float32)
    grid1 = ImageGrid(image_w=image1.shape[1], image_h=image1.shape[0])
    grid1.fill(image1)

    deform = ImageDeform(image1.shape[1], image1.shape[0], 2, 2)
    grid2 = deform.apply_image(grid1, 1)
    correlation = ImageGrid.correlation(grid1, grid2)
    assert correlation == 1

def test_approximate_by_correlation_constant1():
    df1 = next(readjpeg(os.path.join(dir_path, "fine_shift/image3.tiff")))
    df2 = next(readjpeg(os.path.join(dir_path, "fine_shift/image3.tiff")))

    image = df1.get_channel("L")[0].astype(np.float32)
    image_ref = df2.get_channel("L")[0].astype(np.float32)

    w = image.shape[1]
    h = image.shape[0]

    grid = ImageGrid(image_w=w, image_h=h)
    grid.fill(image)
    grid_ref = ImageGrid(image_w=w, image_h=h)
    grid_ref.fill(image_ref)

    with open(kernel_path) as f:
        kernel = f.read()

    lc = ImageDeformLC(image_w=w, image_h=h, pixels=1, kernel_source=kernel)
    deform = lc.find_constant(img=grid,
                              pre_align=None,
                              ref_img=grid_ref,
                              ref_pre_align=None,
                              maximal_shift=2,
                              subpixels=1)
    assert deform is not None

    data = deform.content()
    assert data.shape[0] == h
    assert data.shape[1] == w
    assert data.shape[2] == 2
    assert np.amin(data) == 0
    assert np.amax(data) == 0

def test_approximate_by_correlation_constant2():
    df1 = next(readjpeg(os.path.join(dir_path, "fine_shift/image3.tiff")))
    df2 = next(readjpeg(os.path.join(dir_path, "fine_shift/image4.tiff")))

    image = df1.get_channel("L")[0].astype(np.float32)
    image_ref = df2.get_channel("L")[0].astype(np.float32)

    w = image.shape[1]
    h = image.shape[0]

    grid = ImageGrid(image_w=w, image_h=h)
    grid.fill(image)
    grid_ref = ImageGrid(image_w=w, image_h=h)
    grid_ref.fill(image_ref)

    with open(kernel_path) as f:
        kernel = f.read()

    lc = ImageDeformLC(image_w=w, image_h=h, pixels=1, kernel_source=kernel)
    deform = lc.find_constant(grid, None, grid_ref, None, 2, 1)
    assert deform is not None

    data = deform.content()
    assert data.shape[0] == h
    assert data.shape[1] == w
    assert data.shape[2] == 2
    assert np.amin(data[:,:,0]) == 0
    assert np.amax(data[:,:,0]) == 0
    assert np.amin(data[:,:,1]) == 1
    assert np.amax(data[:,:,1]) == 1

def test_approximate_by_correlation1():
    df1 = next(readjpeg(os.path.join(dir_path, "fine_shift/image1.png")))
    df2 = next(readjpeg(os.path.join(dir_path, "fine_shift/image1.png")))

    image = df1.get_channel("L")[0].astype(np.float32)
    image_ref = df2.get_channel("L")[0].astype(np.float32)

    w = image.shape[1]
    h = image.shape[0]

    grid = ImageGrid(image_w=w, image_h=h)
    grid.fill(image)
    grid_ref = ImageGrid(image_w=w, image_h=h)
    grid_ref.fill(image_ref)

    with open(kernel_path) as f:
        kernel = f.read()

    lc = ImageDeformLC(image_w=w, image_h=h, pixels=1, kernel_source=kernel)
    deform = lc.find(grid, None, grid_ref, None, 5, 5, 1)
    assert deform is not None

    data = deform.content()
    assert data.shape[0] == h
    assert data.shape[1] == w
    assert data.shape[2] == 2
    assert np.amin(data) == 0
    assert np.amax(data) == 0

def test_approximate_by_correlation2():
    df = next(readjpeg(os.path.join(dir_path, "fine_shift/image3.tiff")))
    df_ref = next(readjpeg(os.path.join(dir_path, "fine_shift/image4.tiff")))

    image = df.get_channel("L")[0].astype(np.float32)
    image_ref = df_ref.get_channel("L")[0].astype(np.float32)

    w = image.shape[1]
    h = image.shape[0]

    grid = ImageGrid(image_w=w, image_h=h)
    grid.fill(image)
    grid_ref = ImageGrid(image_w=w, image_h=h)
    grid_ref.fill(image_ref)

    with open(kernel_path) as f:
        kernel = f.read()

    lc = ImageDeformLC(image_w=w, image_h=h, pixels=1, kernel_source=kernel)
    deform = lc.find(img=grid,
                     pre_align=None,
                     ref_img=grid_ref,
                     ref_pre_align=None,
                     radius=5,
                     maximal_shift=3,
                     subpixels=1)
    assert deform is not None

    data = deform.content()
    assert data.shape[0] == h
    assert data.shape[1] == w
    assert data.shape[2] == 2
    for line_id in range(data.shape[0]):
        print(data[line_id,:,1])

    assert np.amin(data[:,:,0]) == 0
    assert np.amax(data[:,:,0]) == 0
    assert np.amin(data[:,:,1]) == 0
    assert np.amax(data[:,:,1]) == 1

def test_memory_leak():
    N = 5
    df1 = next(readjpeg(os.path.join(dir_path, "fine_shift/image1.png")))
    df2 = next(readjpeg(os.path.join(dir_path, "fine_shift/image1.png")))

    image = df1.get_channel("L")[0].astype(np.float32)
    image_ref = df2.get_channel("L")[0].astype(np.float32)

    w = image.shape[1]
    h = image.shape[0]

    prevd = 0
    memory1 = psutil.Process().memory_info().rss
    deltas = []

    for _ in range(N):
        grid = ImageGrid(image_w=w, image_h=h)
        grid.fill(image)
        grid_ref = ImageGrid(image_w=w, image_h=h)
        grid_ref.fill(image_ref)

        lc = ImageDeformLC(image_w=w, image_h=h, pixels=1)
        deform = lc.find(grid, None, grid_ref, None, 5, 5, 1)
    
        lc = None
        deform = None
        grid = None
        grid_ref = None

        gc.collect()
        memory2 = psutil.Process().memory_info().rss
        print(memory1, memory2, memory2 - memory1, memory2 - memory1 - prevd)
        deltas.append(memory2 - memory1 - prevd)
        prevd = memory2 - memory1
    assert deltas[-1] == 0

def test_divergence_1():
    df1 = next(readjpeg(os.path.join(dir_path, "fine_shift/image3.tiff")))
    df2 = next(readjpeg(os.path.join(dir_path, "fine_shift/image3.tiff")))

    image = df1.get_channel("L")[0].astype(np.float32)
    image_ref = df2.get_channel("L")[0].astype(np.float32)

    w = image.shape[1]
    h = image.shape[0]

    grid = ImageGrid(image_w=w, image_h=h)
    grid.fill(image)
    grid_ref = ImageGrid(image_w=w, image_h=h)
    grid_ref.fill(image_ref)

    lc = ImageDeformLC(image_w=w, image_h=h, pixels=1)
    deform = lc.find(grid, None, grid_ref, None, 5, 5, 1)
    assert deform is not None

    divergence = deform.divergence(subpixels=1)
    assert divergence is not None
    divergence = divergence.content()
    assert len(divergence.shape) == 2
    assert divergence.shape[0] == h
    assert divergence.shape[1] == w
    assert np.amin(divergence) == 0
    assert np.amax(divergence) == 0

def test_divergence_2():
    df1 = next(readjpeg(os.path.join(dir_path, "fine_shift/image3.tiff")))
    df2 = next(readjpeg(os.path.join(dir_path, "fine_shift/image4.tiff")))

    image = df1.get_channel("L")[0].astype(np.float32)
    image_ref = df2.get_channel("L")[0].astype(np.float32)

    w = image.shape[1]
    h = image.shape[0]

    grid = ImageGrid(image_w=w, image_h=h)
    grid.fill(image)
    grid_ref = ImageGrid(image_w=w, image_h=h)
    grid_ref.fill(image_ref)

    lc = ImageDeformLC(image_w=w, image_h=h, pixels=1)
    deform = lc.find(grid, None, grid_ref, None, 5, 5, 1)
    assert deform is not None

    divergence = deform.divergence(subpixels=1)
    assert divergence is not None
    divergence = divergence.content()

    assert len(divergence.shape) == 2
    assert divergence.shape[0] == h
    assert divergence.shape[1] == w
