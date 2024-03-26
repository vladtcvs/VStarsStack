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

#include "image_wave_gc.h"

#define SQR(x) ((x)*(x))

int image_wave_gc_init(struct ImageWaveGlobalCorrelator *self,
                       int grid_w, int grid_h,
                       int image_w, int image_h,
                       double spk)
{
    if (grid_h <= 0 || grid_w <= 0 || image_w < 2 || image_h < 2)
        return -1;

    memset(self, 0, sizeof(struct ImageWaveGlobalCorrelator));

    self->image_w = image_w;
    self->image_h = image_h;
    self->grid_w  = grid_w;
    self->grid_h  = grid_h;
    self->stretch_penalty_k = spk;

    self->sx = ((double)self->grid_w) / (self->image_w - 1);
    self->sy = ((double)self->grid_h) / (self->image_h - 1);

    if (image_wave_grid_init(&self->array, self->grid_w, self->grid_h,
                             self->image_w, self->image_h))
    {
        return -1;
    }

    if (image_wave_grid_init(&self->array_p, self->grid_w, self->grid_h,
                             self->image_w, self->image_h))
    {
        image_wave_grid_finalize(&self->array);
        return -1;
    }

    if (image_wave_grid_init(&self->array_m, self->grid_w, self->grid_h,
                             self->image_w, self->image_h))
    {
        image_wave_grid_finalize(&self->array_p);
        image_wave_grid_finalize(&self->array);
        return -1;
    }

    if (image_wave_grid_init(&self->array_gradient, self->grid_w, self->grid_h,
                             self->image_w, self->image_h))
    {
        image_wave_grid_finalize(&self->array_m);
        image_wave_grid_finalize(&self->array_p);
        image_wave_grid_finalize(&self->array);
        return -1;
    }

    return 0;
}

void image_wave_gc_finalize(struct ImageWaveGlobalCorrelator *self)
{
    image_wave_grid_finalize(&self->array_gradient);
    image_wave_grid_finalize(&self->array_m);
    image_wave_grid_finalize(&self->array_p);
    image_wave_grid_finalize(&self->array);
}

/**
 * \brief Calculate stretch penalty for shift array
 * \param array shift array
 * \return stretch penalty
 */
static double image_wave_gc_stretch_penalty(const struct ImageWaveGrid *array)
{
    double penalty_stretch = 0;
    int xi, yi;
    for (yi = 0; yi < array->grid_w - 1; yi++)
    {
        for (xi = 0; xi < array->grid_h - 1; xi++)
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

/**
 * \brief Add gradient*dh to array to perform gradient descent
 * \param array shift array
 * \param gradient gradient of global correlator by varying array
 * \param dh step of gradient descent
 */
static void image_wave_gc_move_along_gradient(struct ImageWaveGrid *array,
                                    const struct ImageWaveGrid *gradient,
                                    double dh)
{
    int xi, yi;
    double maxv = 0;
    for (yi = 0; yi < array->grid_h; yi++)
    {
        for (xi = 0; xi < array->grid_w; xi++)
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

    for (yi = 0; yi < array->grid_h; yi++)
    {
        for (xi = 0; xi < array->grid_w; xi++)
        {
            double gradient_x = image_wave_get_array(gradient, xi, yi, 0);
            double gradient_y = image_wave_get_array(gradient, xi, yi, 1);
        
            double arr_x = image_wave_get_array(array, xi, yi, 0);
            double arr_y = image_wave_get_array(array, xi, yi, 1);

            image_wave_set_array(array, xi, yi, 0, arr_x - gradient_x * dh / maxv);
            image_wave_set_array(array, xi, yi, 1, arr_y - gradient_y * dh / maxv);
        }
    }
}
