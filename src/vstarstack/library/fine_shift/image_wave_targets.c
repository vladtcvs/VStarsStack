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

/*
 * Calculate penalty of shifts. We want to minimize it
 * Penalty contains of 2 parts:
 * 1. Penalty of points. It calculates from difference between actual
 * points shift and calculated from shift array
 * 2. Penalty of stretch. It calculates from difference between array shift values
 */
static double penalty(struct ImageWave *self,
                      struct ImageWaveGrid *array,
                      double *targets, double *points, size_t N)
{
    size_t i;
    double penalty_points = 0;
    for (i = 0; i < N; i++)
    {
        double x = points[i*2];
        double y = points[i*2+1];
        double tx = targets[i*2];
        double ty = targets[i*2+1];

        double sx, sy;
        image_wave_shift_interpolate(self, array, x, y, &sx, &sy);
        penalty_points += SQR(tx-sx) + SQR(ty-sy);
    }

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
    return penalty_points * 1 + penalty_stretch * self->stretch_penalty_k;
}

/*
 * Calculate partial derivative of penalty by shift by axis <axis> at (x,y)
 */
static double partial(struct ImageWave *self,
                      int yi, int xi, int axis,
                      double *targets, double *points, size_t N)
{
    double h = 1e-9;
    memcpy(self->array_p.array, self->array.array, self->Nw*self->Nh*2*sizeof(double));
    memcpy(self->array_m.array, self->array.array, self->Nw*self->Nh*2*sizeof(double));

    double val = image_wave_get_array(&self->array, xi, yi, axis);
    image_wave_set_array(&self->array_p, xi, yi, axis, val+h);
    image_wave_set_array(&self->array_m, xi, yi, axis, val-h);

    double penlaty_p = penalty(self, &self->array_p, targets, points, N);
    double penlaty_m = penalty(self, &self->array_m, targets, points, N);
    return (penlaty_p-penlaty_m)/(2*h);
}

/*
 * Step of gradient descent
 */
static void approximate_step(struct ImageWave *self, double dh,
                             double *targets, double *points, size_t N)
{
    int yi, xi;
    for (yi = 0; yi < self->Nh; yi++)
    {
        for (xi = 0; xi < self->Nw; xi++)
        {
            double gradient_x = partial(self, yi, xi, 0, targets, points, N);
            double gradient_y = partial(self, yi, xi, 1, targets, points, N);
            image_wave_set_array(&self->array_gradient, xi, yi, 0, gradient_x);
            image_wave_set_array(&self->array_gradient, xi, yi, 1, gradient_y);
        }
    }
    double maxv = 0;
    for (yi = 0; yi < self->Nh; yi++)
    {
        for (xi = 0; xi < self->Nw; xi++)
        {
            double gradient_x = image_wave_get_array(&self->array_gradient, xi, yi, 0);
            double gradient_y = image_wave_get_array(&self->array_gradient, xi, yi, 1);

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
                double gradient_x = image_wave_get_array(&self->array_gradient, xi, yi, 0);
                double gradient_y = image_wave_get_array(&self->array_gradient, xi, yi, 1);

                image_wave_set_array(&self->array_gradient, xi, yi, 0, gradient_x/maxv);
                image_wave_set_array(&self->array_gradient, xi, yi, 1, gradient_y/maxv);
            }
        }
    }

    for (yi = 0; yi < self->Nh; yi++)
    {
        for (xi = 0; xi < self->Nw; xi++)
        {
            double gradient_x = image_wave_get_array(&self->array_gradient, xi, yi, 0);
            double gradient_y = image_wave_get_array(&self->array_gradient, xi, yi, 1);
        
            double arr_x = image_wave_get_array(&self->array, xi, yi, 0);
            double arr_y = image_wave_get_array(&self->array, xi, yi, 1);

            image_wave_set_array(&self->array, xi, yi, 0, arr_x - gradient_x*dh);
            image_wave_set_array(&self->array, xi, yi, 1, arr_y - gradient_y*dh);
        }
    }
}

void image_wave_approximate_by_targets(struct ImageWave *self, double dh, size_t Nsteps,
                                       double *targets, double *points, size_t N)
{
    size_t i;
    if (N == 0)
        return;

    double dx = 0, dy = 0;
    for (i = 0; i < N; i++)
    {
        dx += targets[2*i] - points[2*i];
        dy += targets[2*i+1] - points[2*i+1];
    }
    dx /= N;
    dy /= N;

    image_wave_init_shift_array(self->array.array, self->array.w, self->array.h, dx, dy);    

    for (i = 0; i < Nsteps; i++)
        approximate_step(self, dh, targets, points, N);
}
