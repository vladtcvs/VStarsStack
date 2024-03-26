/*
 * Copyright (c) 2023-2024 Vladislav Tsendrovskii
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

#include <math.h>
#include "image_grid.h"

#define SQR(x) ((x)*(x))

double image_grid_correlation(const struct ImageGrid *image1,
                              const struct ImageGrid *image2)
{
    double top = 0;
    double bottom1 = 0, bottom2 = 0;

    if (image1->w != image2->w || image1->h != image2->h)
    {
        printf("Error!\n");
        return NAN;
    }

    double average1 = 0, average2 = 0;
    int nump = 0;

    int i, j;
    for (i = 0; i < image1->h; i++)
    for (j = 0; j < image1->w; j++)
    {
        double pixel1 = image_wave_get_array(image1, j, i, 0);
        double pixel2 = image_wave_get_array(image2, j, i, 0);
        if (isnan(pixel1) || isnan(pixel2))
            continue;

        average1 += pixel1;
        average2 += pixel2;
        nump++;
    }

    average1 /= nump;
    average2 /= nump;

    for (i = 0; i < image1->h; i++)
    for (j = 0; j < image1->w; j++)
    {
        double pixel1 = image_wave_get_array(image1, j, i, 0);
        double pixel2 = image_wave_get_array(image2, j, i, 0);
        if (isnan(pixel1) || isnan(pixel2))
            continue;

        top += (pixel1 - average1)*(pixel2 - average2);
        bottom1 += SQR(pixel1 - average1);
        bottom2 += SQR(pixel2 - average2);
    }

    if (bottom1 == 0 || bottom2 == 0)
    {
        if (bottom1 == 0 && bottom2 == 0)
            return 1;
        return 0;
    }
    return top / sqrt(bottom1 * bottom2);
}

/**
 * \brief Get pixel at pos (x,y)
 * \param grid Shift array
 * \param x x
 * \param y y
 * \return value
 */
static double image_grid_get_array(const struct ImageGrid *grid,
                                   int x, int y)
{
    if (x >= grid->w)
        return NAN;
    if (x < 0)
        return NAN;
    if (y >= grid->h)
        return NAN;
    if (y < 0)
        return NAN;
    return grid->array[y * grid->w + x];
}

static double image_grid_interpolation(const struct ImageGrid *array,
                                       int xi, int yi,
                                       double dx, double dy)
{
    double x_m1m1 = image_grid_get_array(array, xi-1, yi-1);
    double x_0m1 = image_grid_get_array(array, xi, yi-1);
    double x_1m1 = image_grid_get_array(array, xi+1, yi-1);
    double x_2m1 = image_grid_get_array(array, xi+2, yi-1);

    double x_m10 = image_grid_get_array(array, xi-1, yi);
    double x_00 = image_grid_get_array(array, xi, yi);
    double x_10 = image_grid_get_array(array, xi+1, yi);
    double x_20 = image_grid_get_array(array, xi+2, yi);

    double x_m11 = image_grid_get_array(array, xi-1, yi+1);
    double x_01 = image_grid_get_array(array, xi, yi+1);
    double x_11 = image_grid_get_array(array, xi+1, yi+1);
    double x_21 = image_grid_get_array(array, xi+2, yi+1);
    
    double x_m12 = image_grid_get_array(array, xi-1, yi+2);
    double x_02 = image_grid_get_array(array, xi, yi+2);
    double x_12 = image_grid_get_array(array, xi+1, yi+2);
    double x_22 = image_grid_get_array(array, xi+2, yi+2);

    return interpolation_2d(x_m1m1, x_0m1, x_1m1, x_2m1,
                            x_m10,  x_00,  x_10,  x_20,
                            x_m11,  x_01,  x_11,  x_21,
                            x_m12,  x_02,  x_12,  x_22,
                            dx, dy);
}

double image_grid_get_pixel(const struct ImageGrid *image, double x, double y)
{
    if (x < 0)
        return NAN;
    if (y < 0)
        return NAN;
    if (x >= image->w)
        return NAN;
    if (y >= image->h)
        return NAN;

    double dx = x - floor(x);
    double dy = y - floor(y);
    if (dx != 0 || dy != 0)
        return image_grid_interpolation(image, floor(x), floor(y), dx, dy);
    else
        return image_grid_get_array(image, (int)x, (int)y);
}
