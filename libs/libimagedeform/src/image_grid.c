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

#include <stdlib.h>
#include <math.h>

#include <image_grid.h>
#include <interpolation.h>

#define SQR(x) ((x)*(x))

int image_grid_init(struct ImageGrid *image, int width, int height)
{
    image->h = height;
    image->w = width;
    image->array = calloc(width*height, sizeof(double));
    if (image->array == NULL)
        return -1;
    return 0;
}

void image_grid_finaize(struct ImageGrid *image)
{
    if (image->array != NULL)
    {
        free(image->array);
        image->array = NULL;
    }
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

/**
 * @brief Interpolate image pixels
 * 
 * @param array image
 * @param xi x of pixel
 * @param yi y of pixel
 * @param dx x position between pixels
 * @param dy y position between pixels
 * @return interpolated value
 */
static double image_grid_interpolation(const struct ImageGrid *array,
                                       int xi, int yi,
                                       double dx, double dy)
{
    double x_00 = image_grid_get_array(array, xi, yi);
    double x_10 = image_grid_get_array(array, xi+1, yi);
 
    double x_01 = image_grid_get_array(array, xi, yi+1);
    double x_11 = image_grid_get_array(array, xi+1, yi+1);
 
    return interpolation_2d_linear(x_00,  x_10,
                                   x_01,  x_11,
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
    if (dx > 1e-3 || dy > 1e-3)
        return image_grid_interpolation(image, floor(x), floor(y), dx, dy);
    else
        return image_grid_get_array(image, (int)x, (int)y);
}

void image_grid_get_area(const struct ImageGrid *img,
                         double x, double y,
                         struct ImageGrid *area)
{
    int i, j;
    int w = area->w;
    int h = area->h;
    for (i = 0; i < h; i++)
    for (j = 0; j < w; j++)
    {
        double px = x + j - (w-1)/2.0;
        double py = y + i - (h-1)/2.0;
        double val = image_grid_get_pixel(img, px, py);
        image_grid_set_pixel(area, j, i, val);
    }
}

double image_grid_correlation(const struct ImageGrid *image1,
                              const struct ImageGrid *image2)
{
    double top = 0;
    double bottom1 = 0, bottom2 = 0;

    if (image1->w != image2->w || image1->h != image2->h)
        return NAN;

    double average1 = 0, average2 = 0;
    int nump = 0;

    int i, j;
    for (i = 0; i < image1->h; i++)
    for (j = 0; j < image1->w; j++)
    {
        double pixel1 = image_grid_get_array(image1, j, i);
        double pixel2 = image_grid_get_array(image2, j, i);
        if (isnan(pixel1) || isnan(pixel2))
            continue;

        average1 += pixel1;
        average2 += pixel2;
        nump++;
    }

    if (nump == 0)
        return NAN;

    average1 /= nump;
    average2 /= nump;

    for (i = 0; i < image1->h; i++)
    for (j = 0; j < image1->w; j++)
    {
        double pixel1 = image_grid_get_array(image1, j, i);
        double pixel2 = image_grid_get_array(image2, j, i);
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
