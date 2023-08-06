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

void image_wave_print_array(const struct ImageWaveGrid *array)
{
    int i, j, k;
    for (i = 0; i < array->h; i++)
    {
        for (j = 0; j < array->w; j++)
        {
            if (array->naxis > 1)
                printf("[");
            for (k = 0; k < array->naxis; k++)
                printf("%lf ", image_wave_get_array(array, j, i, k));
            if (array->naxis > 1)
                printf("] ");
        }
        printf("\n");
    }
    printf("\n");
}

/*
 * Init shift array with specified (dx, dy)
 */
void image_wave_init_shift_array(double *array, int w, int h, double dx, double dy)
{
    struct ImageWaveGrid grid = {
        .array = array,
        .w = w,
        .h = h,
        .naxis = 2,
    };
    int xi, yi;
    for (yi = 0; yi < h; yi++)
    {
        for (xi = 0; xi < w; xi++)
        {
            image_wave_set_array(&grid, xi, yi, 0, dx);
            image_wave_set_array(&grid, xi, yi, 1, dy);
        }
    }
}

int image_wave_init(struct ImageWave *self, int w, int h, double Nw, double Nh, double spk)
{
    if (h <= 0 || w <= 0 || Nw < 2 || Nh < 2)
        return -1;

    self->Nw = Nw;
    self->Nh = Nh;

    self->w = w;
    self->h = h;

    self->array.w = Nw;
    self->array.h = Nh;
    self->array.naxis = 2;

    self->array_p.w = Nw;
    self->array_p.h = Nh;
    self->array_p.naxis = 2;

    self->array_m.w = Nw;
    self->array_m.h = Nh;
    self->array_m.naxis = 2;

    self->array_gradient.w = Nw;
    self->array_gradient.h = Nh;
    self->array_gradient.naxis = 2;

    self->stretch_penalty_k = spk;

    self->sx = ((double)self->w) / (self->Nw - 1);
    self->sy = ((double)self->h) / (self->Nh - 1);

    self->array.array = calloc(self->Nw * self->Nh * 2, sizeof(double));
    if (!self->array.array)
    {
        return -1;
    }

    self->array_p.array = calloc(Nw * Nh * 2, sizeof(double));
    if (!self->array_p.array)
    {
        free(self->array.array);
        return -1;
    }

    self->array_m.array = calloc(Nw * Nh * 2, sizeof(double));
    if (!self->array_m.array)
    {
        free(self->array_p.array);
        free(self->array.array);
        return -1;
    }
    self->array_gradient.array = calloc(Nw * Nh * 2, sizeof(double));
    if (!self->array_gradient.array)
    {
        free(self->array_m.array);
        free(self->array_p.array);
        free(self->array.array);
        return -1;
    }
    return 0;
}

void image_wave_finalize(struct ImageWave *self)
{
    if (self->array_gradient.array)
    {
        free(self->array_gradient.array);
        self->array_gradient.array = NULL;
    }
    if (self->array_m.array)
    {
        free(self->array_m.array);
        self->array_m.array = NULL;
    }
    if (self->array_p.array)
    {
        free(self->array_p.array);
        self->array_p.array = NULL;
    }
    if (self->array.array)
    {
        free(self->array.array);
        self->array.array = NULL;
    }
}

void image_wave_move_along_gradient(struct ImageWave *self,
                                    struct ImageWaveGrid *gradient,
                                    double dh)
{
    int xi, yi;
    double maxv = 0;
    for (yi = 0; yi < self->Nh; yi++)
    {
        for (xi = 0; xi < self->Nw; xi++)
        {
            double gradient_x = image_wave_get_array(gradient, xi, yi, 0);
            double gradient_y = image_wave_get_array(gradient, xi, yi, 1);

            if (fabs(gradient_x) > maxv)
                maxv = fabs(gradient_x);
            if (fabs(gradient_y) > maxv)
                maxv = fabs(gradient_y);
        }
    }

    if (maxv > 1)
    {
        for (yi = 0; yi < self->Nh; yi++)
        {
            for (xi = 0; xi < self->Nw; xi++)
            {
                double gradient_x = image_wave_get_array(gradient, xi, yi, 0);
                double gradient_y = image_wave_get_array(gradient, xi, yi, 1);

                image_wave_set_array(&self->array_gradient, xi, yi, 0, gradient_x/maxv);
                image_wave_set_array(&self->array_gradient, xi, yi, 1, gradient_y/maxv);
            }
        }
    }

    for (yi = 0; yi < self->Nh; yi++)
    {
        for (xi = 0; xi < self->Nw; xi++)
        {
            double gradient_x = image_wave_get_array(gradient, xi, yi, 0);
            double gradient_y = image_wave_get_array(gradient, xi, yi, 1);
        
            double arr_x = image_wave_get_array(&self->array, xi, yi, 0);
            double arr_y = image_wave_get_array(&self->array, xi, yi, 1);

            image_wave_set_array(&self->array, xi, yi, 0, arr_x - gradient_x*dh);
            image_wave_set_array(&self->array, xi, yi, 1, arr_y - gradient_y*dh);
        }
    }
}
