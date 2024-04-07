#include <image_deform.h>
#include <gtest/gtest.h>

TEST(imagedeform, test_init)
{
    struct ImageDeform deform;
    int res = image_deform_init(&deform, 2, 2, 10, 10);
    ASSERT_EQ(res, 0);

    EXPECT_EQ(deform.image_h, 10);
    EXPECT_EQ(deform.image_w, 10);
    EXPECT_EQ(deform.grid_h, 2);
    EXPECT_EQ(deform.grid_w, 2);

    image_deform_finalize(&deform);
}

TEST(imagedeform, test_identity_integer)
{
    struct ImageDeform deform;
    int res = image_deform_init(&deform, 2, 2, 10, 10);
    ASSERT_EQ(res, 0);
    double x, y;
    
    image_deform_apply_point(&deform, 0, 0, &x, &y);
    EXPECT_EQ(x, 0);
    EXPECT_EQ(y, 0);

    image_deform_apply_point(&deform, 0, 9, &x, &y);
    EXPECT_EQ(x, 0);
    EXPECT_EQ(y, 9);

    image_deform_apply_point(&deform, 9, 9, &x, &y);
    EXPECT_EQ(x, 9);
    EXPECT_EQ(y, 9);

    image_deform_apply_point(&deform, 10, 9, &x, &y);
    EXPECT_TRUE(isnan(x));
    EXPECT_TRUE(isnan(y));

    image_deform_finalize(&deform);
}

TEST(imagedeform, test_identity_float)
{
    struct ImageDeform deform;
    int res = image_deform_init(&deform, 2, 2, 10, 10);
    ASSERT_EQ(res, 0);
    double x, y;

    image_deform_apply_point(&deform, 4.5, 3.2, &x, &y);
    EXPECT_EQ(x, 4.5);
    EXPECT_EQ(y, 3.2);

    image_deform_finalize(&deform);
}
