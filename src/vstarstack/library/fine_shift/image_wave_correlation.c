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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image_wave.h"

#define SQR(x) ((x)*(x))

static double image_average(const struct ImageWaveGrid *image)
{
    double sum = 0;
    int i, j;
    for (i = 0; i < image->h; i++)
    for (j = 0; j < image->w; j++)
        sum += image_wave_get_array(image, j, i, 0);
    return sum / (image->w * image->h);
}

static double correlation(const struct ImageWaveGrid *image1,
                          const struct ImageWaveGrid *image2)
{
    double average1 = image_average(image1);
    double average2 = image_average(image2);

    double top = 0;
    int i, j;
    for (i = 0; i < image1->h; i++)
    for (j = 0; j < image1->w; j++)
    {
        double pixel1 = image_wave_get_array(image1, j, i, 0);
        double pixel2 = image_wave_get_array(image2, j, i, 0);
        top += (pixel1 - average1)*(pixel2 - average2);
    }

    double bottom1 = 0, bottom2 = 0;
    for (i = 0; i < image1->h; i++)
    for (j = 0; j < image1->w; j++)
    {
        double pixel1 = image_wave_get_array(image1, j, i, 0);
        bottom1 += SQR(pixel1 - average1);

        double pixel2 = image_wave_get_array(image2, j, i, 0);
        bottom2 += SQR(pixel2 - average2);
    }

    if (bottom1 == 0 || bottom2 == 0)
        return 0;
    return top / sqrt(bottom1 * bottom2);
}

static double penalty(struct ImageWave *self,
                      struct ImageWaveGrid *array,
                      const struct ImageWaveGrid *image1,
                      const struct ImageWaveGrid *image2,
                      struct ImageWaveGrid *tmp)
{
    image_wave_shift_image(self, array, image1, tmp);
    double corr = correlation(tmp, image2);

    double penalty_stretch = 0;
    int xi, yi;
    for (yi = 0; yi < self->Nh-1; yi++)
    {
        for (xi = 0; xi < self->Nw-1; xi++)
        {
            double current_x = image_wave_get_array(array, xi, yi, 0);
            double current_y = image_wave_get_array(array, xi, yi, 1);
            double right_x = image_wave_get_array(array, xi+1, yi, 0);
            double right_y = image_wave_get_array(array, xi+1, yi, 1);
            double bottom_x = image_wave_get_array(array, xi, yi+1, 0);
            double bottom_y = image_wave_get_array(array, xi, yi+1, 1);

            penalty_stretch += SQR(current_x-right_x);
            penalty_stretch += SQR(current_y-right_y);
            
            penalty_stretch += SQR(current_x-bottom_x);
            penalty_stretch += SQR(current_y-bottom_y);
        }
    }
    return penalty_stretch * self->stretch_penalty_k - corr;
}

static double partial(struct ImageWave *self,
                      int yi, int xi, int axis,
                      const struct ImageWaveGrid *image1,
                      const struct ImageWaveGrid *image2,
                      struct ImageWaveGrid *tmp)
{
    double h = 1e-9;
    memcpy(self->array_p.array, self->array.array, self->Nw*self->Nh*2*sizeof(double));
    memcpy(self->array_m.array, self->array.array, self->Nw*self->Nh*2*sizeof(double));

    double val = image_wave_get_array(&self->array, xi, yi, axis);
    image_wave_set_array(&self->array_p, xi, yi, axis, val+h);
    image_wave_set_array(&self->array_m, xi, yi, axis, val-h);

    double penlaty_p = penalty(self, &self->array_p, image1, image2, tmp);
    double penlaty_m = penalty(self, &self->array_m, image1, image2, tmp);
    return (penlaty_p-penlaty_m)/(2*h);
}

static void approximate_step(struct ImageWave *self, double dh,
                             const struct ImageWaveGrid *image1,
                             const struct ImageWaveGrid *image2,
                             struct ImageWaveGrid *tmp)
{
    int yi, xi;
    for (yi = 0; yi < self->Nh; yi++)
    {
        for (xi = 0; xi < self->Nw; xi++)
        {
            double gradient_x = partial(self, yi, xi, 0, image1, image2, tmp);
            double gradient_y = partial(self, yi, xi, 1, image1, image2, tmp);
            image_wave_set_array(&self->array_gradient, xi, yi, 0, gradient_x);
            image_wave_set_array(&self->array_gradient, xi, yi, 1, gradient_y);
        }
    }

    image_wave_move_along_gradient(self, &self->array_gradient, dh);

    image_wave_shift_image(self, &self->array, image1, tmp);
    double corr = correlation(tmp, image2);
    printf("correlation = %lf\n", corr);
}

void image_wave_approximate_by_correlation(struct ImageWave *self,
                                           double dh, size_t Nsteps,
                                           const struct ImageWaveGrid *image1,
                                           const struct ImageWaveGrid *image2,
                                           struct ImageWaveGrid *tmp)
{
    unsigned i;
    image_wave_init_shift_array(self->array.array, self->array.w, self->array.h, 0, 0);    
    for (i = 0; i < Nsteps; i++)
        approximate_step(self, dh, image1, image2, tmp);
}
