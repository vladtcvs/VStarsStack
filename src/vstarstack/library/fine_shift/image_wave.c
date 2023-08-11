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
void image_wave_init_shift_array(struct ImageWaveGrid *grid, double dx, double dy)
{
    int xi, yi;
    for (yi = 0; yi < grid->h; yi++)
    {
        for (xi = 0; xi < grid->w; xi++)
        {
            image_wave_set_array(grid, xi, yi, 0, dx);
            image_wave_set_array(grid, xi, yi, 1, dy);
        }
    }
}

int image_wave_grid_init(struct ImageWaveGrid *grid, int w, int h, int naxis)
{
    if (grid->array != NULL)
        free(grid->array);

    grid->w = w;
    grid->h = h;
    grid->naxis = naxis;
    grid->array = calloc(w * h * 2, sizeof(double));
    if (grid->array == NULL)
        return -1;
    return 0;
}

void image_wave_grid_finalize(struct ImageWaveGrid *grid)
{
    if (grid->array != NULL)
    {
        free(grid->array);
        grid->array = NULL;
    }
}

int image_wave_aux_init(struct ImageWave *self)
{
    int w = self->array.w;
    int h = self->array.h;

    if (image_wave_grid_init(&self->array_p, w, h, 2))
        return -1;

    if (image_wave_grid_init(&self->array_m, w, h, 2))
    {
        image_wave_grid_finalize(&self->array_p);
        return -1;
    }

    if (image_wave_grid_init(&self->array_gradient, w, h, 2))
    {
        image_wave_grid_finalize(&self->array_p);
        image_wave_grid_finalize(&self->array_m);
        return -1;
    }

    return 0;
}

int image_wave_init(struct ImageWave *self,
                    int w, int h,
                    double Nw, double Nh,
                    double spk)
{
    memset(self, 0, sizeof(struct ImageWave));

    if (h <= 0 || w <= 0 || Nw < 2 || Nh < 2)
        return -1;

    self->Nw = Nw;
    self->Nh = Nh;

    self->w = w;
    self->h = h;

    self->stretch_penalty_k = spk;

    self->sx = ((double)self->w) / (self->Nw - 1);
    self->sy = ((double)self->h) / (self->Nh - 1);

    if (image_wave_grid_init(&self->array, self->Nw, self->Nh, 2))
        return -1;

    return 0;
}

void image_wave_finalize(struct ImageWave *self)
{
    image_wave_grid_finalize(&self->array_gradient);
    image_wave_grid_finalize(&self->array_m);
    image_wave_grid_finalize(&self->array_p);
    image_wave_grid_finalize(&self->array);
}

void image_wave_move_along_gradient(struct ImageWave *self,
                                    const struct ImageWaveGrid *gradient,
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

    if (maxv < 1)
        maxv = 1;

    for (yi = 0; yi < self->Nh; yi++)
    {
        for (xi = 0; xi < self->Nw; xi++)
        {
            double gradient_x = image_wave_get_array(gradient, xi, yi, 0);
            double gradient_y = image_wave_get_array(gradient, xi, yi, 1);
        
            double arr_x = image_wave_get_array(&self->array, xi, yi, 0);
            double arr_y = image_wave_get_array(&self->array, xi, yi, 1);

            image_wave_set_array(&self->array, xi, yi, 0, arr_x - gradient_x * dh / maxv);
            image_wave_set_array(&self->array, xi, yi, 1, arr_y - gradient_y * dh / maxv);
        }
    }
}

double image_wave_stretch_penalty(const struct ImageWaveGrid *array)
{
    double penalty_stretch = 0;
    int xi, yi;
    for (yi = 0; yi < array->h - 1; yi++)
    {
        for (xi = 0; xi < array->w - 1; xi++)
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
    return penalty_stretch;
}
