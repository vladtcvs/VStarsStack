/*
 * Copyright (c) 2024 Vladislav Tsendrovskii
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include <image_deform_gc.h>
#include <image_deform.h>
#include <gtest/gtest.h>

TEST(imagedeform_gc, test_init)
{
    struct ImageDeformGlobalCorrelator gc;
    int res = image_deform_gc_init(&gc, 2, 2, 10, 10, 0.01);
    ASSERT_EQ(res, 0);
    image_deform_gc_finalize(&gc);
}

TEST(imagedeform_gc, test_single)
{
    struct ImageDeformGlobalCorrelator gc;
    int res = image_deform_gc_init(&gc, 2, 2, 10, 10, 0.01);
    ASSERT_EQ(res, 0);

    real_t points[2] = {5.0, 5.2};
    real_t expected[2] = {5.0, 5.0};

    ImageDeform *deform = image_deform_gc_find(&gc, 0.1, 0, points, expected, 1);

    real_t x, y;
    image_deform_apply_point(deform, 5.2, 5, &x, &y);
    EXPECT_NEAR(x, 5, 1e-3);
    EXPECT_NEAR(y, 5, 1e-3);

    image_deform_gc_finalize(&gc);
}

TEST(imagedeform_gc, test_square_shift_array)
{
    struct ImageDeformGlobalCorrelator gc;
    int res = image_deform_gc_init(&gc, 2, 2, 10, 10, 0.000);
    ASSERT_EQ(res, 0);

    real_t points[8] = {0.1f, 0.1f,
                        0.1f, 8.9f,
                        8.9f, 0.1f,
                        8.9f, 8.9f};

    real_t expected[8] = {0, 0,
                          0, 9,
                          9, 0,
                          9, 9};

    ImageDeform *deform = image_deform_gc_find(&gc, 0.01, 200, points, expected, 4);

    real_t dy = image_deform_get_shift(deform, 0.0, 0.0, 0);
    real_t dx = image_deform_get_shift(deform, 0.0, 0.0, 1);
    EXPECT_NEAR(dx, -0.1f/9, 1e-2);
    EXPECT_NEAR(dy, -0.1f/9, 1e-2);

    dy = image_deform_get_shift(deform, 0.0, 1.0, 0);
    dx = image_deform_get_shift(deform, 0.0, 1.0, 1);
    EXPECT_NEAR(dx, -0.1f/9, 1e-2);
    EXPECT_NEAR(dy, 0.1f/9, 1e-2);

    dy = image_deform_get_shift(deform, 1.0, 0.0, 0);
    dx = image_deform_get_shift(deform, 1.0, 0.0, 1);
    EXPECT_NEAR(dx, 0.1f/9, 1e-2);
    EXPECT_NEAR(dy, -0.1f/9, 1e-2);

    dy = image_deform_get_shift(deform, 1.0, 1.0, 0);
    dx = image_deform_get_shift(deform, 1.0, 1.0, 1);
    EXPECT_NEAR(dx, 0.1f/9, 1e-2);
    EXPECT_NEAR(dy, 0.1f/9, 1e-2);

    image_deform_gc_finalize(&gc);
}

TEST(imagedeform_gc, test_square)
{
    struct ImageDeformGlobalCorrelator gc;
    int res = image_deform_gc_init(&gc, 2, 2, 10, 10, 0.000);
    ASSERT_EQ(res, 0);

    real_t points[8] = {0.1f, 0.1f,
                        0.1f, 8.9f,
                        8.9f, 0.1f,
                        8.9f, 8.9f};

    real_t expected[8] = {0, 0,
                          0, 9,
                          9, 0,
                          9, 9};

    ImageDeform *deform = image_deform_gc_find(&gc, 0.001, 2000, points, expected, 4);
    real_t x, y;

    image_deform_apply_point(deform, 4.5f, 4.5f, &x, &y);
    EXPECT_NEAR(x, 4.5f, 1e-3);
    EXPECT_NEAR(y, 4.5f, 1e-3);

    image_deform_apply_point(deform, 0.1f, 0.1f, &x, &y);
    EXPECT_NEAR(x, 0, 1e-3);
    EXPECT_NEAR(y, 0, 1e-3);

    image_deform_apply_point(deform, 8.9f, 0.1f, &x, &y);
    EXPECT_NEAR(x, 9, 1e-3);
    EXPECT_NEAR(y, 0, 1e-3);

    image_deform_gc_finalize(&gc);
}
