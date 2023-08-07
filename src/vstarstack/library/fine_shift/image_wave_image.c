/*
 * Copyright (c) 2023 Vladislav Tsendrovskii
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
#include "image_wave.h"

static void image_wave_set_pixel(struct ImageWaveGrid *image, int x, int y, double val)
{
    image_wave_set_array(image, x, y, 0, val);
}

static double image_wave_get_pixel(const struct ImageWaveGrid *image, double x, double y)
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
    return image_wave_interpolation(image, floor(x), floor(y), 0, dx, dy);
}


/* Apply shift array to image */
void image_wave_shift_image(struct ImageWave *self,
                            const struct ImageWaveGrid *array,
                            const struct ImageWaveGrid *input_image,
                            struct ImageWaveGrid *output_image)
{
    int y, x;
    for (y = 0; y < output_image->h; y++)
        for (x = 0; x < output_image->w; x++)
        {
            double orig_y, orig_x;
            image_wave_shift_interpolate(self, array, x, y, &orig_x, &orig_y);
            double val = image_wave_get_pixel(input_image, orig_x, orig_y);
            image_wave_set_pixel(output_image, x, y, val);
        }
}

double image_wave_correlation(const struct ImageWaveGrid *image1,
                              const struct ImageWaveGrid *image2)
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
