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

    double points[2] = {5.0, 5.2};
    double expected[2] = {5.0, 5.0};

    ImageDeform *deform = image_deform_gc_find(&gc, 0.1, 0, points, expected, 1);

    double x, y;
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

    double points[8] = {0.1, 0.1,
                        0.1, 8.9,
                        8.9, 0.1,
                        8.9, 8.9};

    double expected[8] = {0, 0,
                          0, 9,
                          9, 0,
                          9, 9};

    ImageDeform *deform = image_deform_gc_find(&gc, 0.01, 200, points, expected, 4);

    double dy = image_deform_get_shift(deform, 0.0, 0.0, 0);
    double dx = image_deform_get_shift(deform, 0.0, 0.0, 1);
    EXPECT_NEAR(dx, -0.1/9, 1e-2);
    EXPECT_NEAR(dy, -0.1/9, 1e-2);

    dy = image_deform_get_shift(deform, 0.0, 1.0, 0);
    dx = image_deform_get_shift(deform, 0.0, 1.0, 1);
    EXPECT_NEAR(dx, -0.1/9, 1e-2);
    EXPECT_NEAR(dy, 0.1/9, 1e-2);

    dy = image_deform_get_shift(deform, 1.0, 0.0, 0);
    dx = image_deform_get_shift(deform, 1.0, 0.0, 1);
    EXPECT_NEAR(dx, 0.1/9, 1e-2);
    EXPECT_NEAR(dy, -0.1/9, 1e-2);

    dy = image_deform_get_shift(deform, 1.0, 1.0, 0);
    dx = image_deform_get_shift(deform, 1.0, 1.0, 1);
    EXPECT_NEAR(dx, 0.1/9, 1e-2);
    EXPECT_NEAR(dy, 0.1/9, 1e-2);

    image_deform_gc_finalize(&gc);
}

TEST(imagedeform_gc, test_square)
{
    struct ImageDeformGlobalCorrelator gc;
    int res = image_deform_gc_init(&gc, 2, 2, 10, 10, 0.000);
    ASSERT_EQ(res, 0);

    double points[8] = {0.1, 0.1,
                        0.1, 8.9,
                        8.9, 0.1,
                        8.9, 8.9};

    double expected[8] = {0, 0,
                          0, 9,
                          9, 0,
                          9, 9};

    ImageDeform *deform = image_deform_gc_find(&gc, 0.001, 2000, points, expected, 4);
    double x, y;

    image_deform_apply_point(deform, 4.5, 4.5, &x, &y);
    EXPECT_NEAR(x, 4.5, 1e-3);
    EXPECT_NEAR(y, 4.5, 1e-3);

    image_deform_apply_point(deform, 0.1, 0.1, &x, &y);
    EXPECT_NEAR(x, 0, 1e-3);
    EXPECT_NEAR(y, 0, 1e-3);

    image_deform_apply_point(deform, 8.9, 0.1, &x, &y);
    EXPECT_NEAR(x, 9, 1e-3);
    EXPECT_NEAR(y, 0, 1e-3);

    image_deform_gc_finalize(&gc);
}
