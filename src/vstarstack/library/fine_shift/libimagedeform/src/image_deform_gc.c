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

#include <image_deform_gc.h>

/**
 * \brief square of 'x'
 * \param x value
 * \return square of 'x'
 */
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

    if (image_deform_init(&self->array, self->grid_w, self->grid_h,
                             self->image_w, self->image_h))
    {
        return -1;
    }

    if (image_deform_init(&self->array_p, self->grid_w, self->grid_h,
                             self->image_w, self->image_h))
    {
        image_deform_finalize(&self->array);
        return -1;
    }

    if (image_deform_init(&self->array_m, self->grid_w, self->grid_h,
                             self->image_w, self->image_h))
    {
        image_deform_finalize(&self->array_p);
        image_deform_finalize(&self->array);
        return -1;
    }

    if (image_deform_init(&self->array_gradient, self->grid_w, self->grid_h,
                             self->image_w, self->image_h))
    {
        image_deform_finalize(&self->array_m);
        image_deform_finalize(&self->array_p);
        image_deform_finalize(&self->array);
        return -1;
    }

    return 0;
}

void image_wave_gc_finalize(struct ImageWaveGlobalCorrelator *self)
{
    image_deform_finalize(&self->array_gradient);
    image_deform_finalize(&self->array_m);
    image_deform_finalize(&self->array_p);
    image_deform_finalize(&self->array);
}

/**
 * \brief Calculate stretch penalty for shift array
 * \param array shift array
 * \return stretch penalty
 */
static double image_wave_gc_stretch_penalty(const struct ImageDeform *array)
{
    double penalty_stretch = 0;
    int xi, yi;
    for (yi = 0; yi < array->grid_w - 1; yi++)
    {
        for (xi = 0; xi < array->grid_h - 1; xi++)
        {
            double current_x = image_deform_get_array(array, xi, yi, 0);
            double current_y = image_deform_get_array(array, xi, yi, 1);
            double right_x = image_deform_get_array(array, xi+1, yi, 0);
            double right_y = image_deform_get_array(array, xi+1, yi, 1);
            double bottom_x = image_deform_get_array(array, xi, yi+1, 0);
            double bottom_y = image_deform_get_array(array, xi, yi+1, 1);

            penalty_stretch += SQR(current_x-right_x);
            penalty_stretch += SQR(current_y-right_y);

            penalty_stretch += SQR(current_x-bottom_x);
            penalty_stretch += SQR(current_y-bottom_y);
        }
    }
    return penalty_stretch;
}

/**
 * \brief Penalty of image shift grid
 * 
 * Calculate penalty of shifts. We want to minimize it
 * Penalty contains of 2 parts:
 * 1. Penalty of points. It calculates from difference between actual
 * points shift and calculated from shift array
 * 2. Penalty of stretch. It calculates from difference between array shift values
 *
 * \param self structure with parameters
 * \param grid used image shift array
 * \param expected_after_shift list of target point positions, array Nx2
 * \param points list of original point positions, array Nx2
 * \param N count of points
 * \return penalty
 */
static double image_wave_gc_penalty(struct ImageWaveGlobalCorrelator *self,
                                    struct ImageDeform *grid,
                                    double *expected_after_shift,
                                    double *points,
                                    size_t N)
{
    size_t i;
    double penalty_points = 0;
    for (i = 0; i < N; i++)
    {
        double x = points[i*2];
        double y = points[i*2+1];
        double expected_x = expected_after_shift[i*2];
        double expected_y = expected_after_shift[i*2+1];

        double shifted_x;
        double shifted_y;
        image_wave_apply(self, grid, x, y, &shifted_x, &shifted_y);
        penalty_points += SQR(expected_x-shifted_x) + SQR(expected_y-shifted_y);
    }

    double penalty_stretch = image_wave_gc_stretch_penalty(grid);
    return penalty_points * 1 + penalty_stretch * self->stretch_penalty_k;
}


/**
 * \brief Add gradient*dh to array to perform gradient descent
 * \param array shift array
 * \param gradient gradient of global correlator by varying array
 * \param dh step of gradient descent
 */
static void image_wave_gc_move_along_gradient(struct ImageDeform *array,
                                              const struct ImageDeform *gradient,
                                              double dh)
{
    int xi, yi;
    double maxv = 0;
    for (yi = 0; yi < array->grid_h; yi++)
    {
        for (xi = 0; xi < array->grid_w; xi++)
        {
            double gradient_x = image_deform_get_array(gradient, xi, yi, 0);
            double gradient_y = image_deform_get_array(gradient, xi, yi, 1);

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
            double gradient_x = image_deform_get_array(gradient, xi, yi, 0);
            double gradient_y = image_deform_get_array(gradient, xi, yi, 1);
        
            double arr_x = image_deform_get_array(array, xi, yi, 0);
            double arr_y = image_deform_get_array(array, xi, yi, 1);

            image_wave_set_array(array, xi, yi, 0, arr_x - gradient_x * dh / maxv);
            image_wave_set_array(array, xi, yi, 1, arr_y - gradient_y * dh / maxv);
        }
    }
}

/**
 * \brief Calculate partial derivative of penalty by shift by axis <axis> at (x,y)
 * \param self global correlator
 * \param xi x coordinate of movement grid point
 * \param yi y coordinate of movement grid point
 * \param axis axis of movement grid point
 * \param targets target points
 * \param expected_after_shift source point positions
 * \param N num of points
 * \return partial derivative
 */
static double image_wave_gc_partial(struct ImageWaveGlobalCorrelator *self,
                                    int yi, int xi, int axis,
                                    double *targets,
                                    double *expected_after_shift,
                                    size_t N)
{
    double h = 1e-9;
    memcpy(self->array_p.array, self->array.array, self->grid_w*self->grid_h*2*sizeof(double));
    memcpy(self->array_m.array, self->array.array, self->grid_w*self->grid_h*2*sizeof(double));

    double val = image_deform_get_array(&self->array, xi, yi, axis);
    image_wave_set_array(&self->array_p, xi, yi, axis, val+h);
    image_wave_set_array(&self->array_m, xi, yi, axis, val-h);

    double penlaty_p = penalty(self, &self->array_p, targets, expected_after_shift, N);
    double penlaty_m = penalty(self, &self->array_m, targets, expected_after_shift, N);
    return (penlaty_p-penlaty_m)/(2*h);
}

/**
 * \brief Step of gradient descent
 * \param self global correlator
 * \param dh step of gradient descent
 * \param targets target points
 * \param expected_after_shift source point positions
 * \param N num of points
 */
static void image_wave_gc_descent_step(struct ImageWaveGlobalCorrelator *self,
                                       double dh,
                                       double *targets,
                                       double *expected_after_shift,
                                       size_t N)
{
    int yi, xi;
    for (yi = 0; yi < self->grid_h; yi++)
    {
        for (xi = 0; xi < self->grid_w; xi++)
        {
            double gradient_x = image_wave_gc_partial(self, yi, xi, 0, targets, expected_after_shift, N);
            double gradient_y = image_wave_gc_partial(self, yi, xi, 1, targets, expected_after_shift, N);
            image_wave_set_array(&self->array_gradient, xi, yi, 0, gradient_x);
            image_wave_set_array(&self->array_gradient, xi, yi, 1, gradient_y);
        }
    }

    image_wave_gc_move_along_gradient(self, &self->array_gradient, dh);
}

void image_wave_gc_find(struct ImageWaveGlobalCorrelator *self, double dh, size_t Nsteps,
                        double *targets,
                        double *expected_after_shift,
                        size_t N)
{
    size_t i;
    if (N == 0)
        return;

    double dx = 0, dy = 0;
    for (i = 0; i < N; i++)
    {
        dx += targets[2*i] - expected_after_shift[2*i];
        dy += targets[2*i+1] - expected_after_shift[2*i+1];
    }
    dx /= N;
    dy /= N;

    image_wave_grid_constant_shift(&self->array, dx, dy);    

    for (i = 0; i < Nsteps; i++)
        image_wave_gc_descent_step(self, dh, targets, expected_after_shift, N);
}
