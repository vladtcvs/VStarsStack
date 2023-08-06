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

/*
 * Linear interpolation. x lies between points of f0 and f1
 */
static double linear_interpolation(double f0, double f1, double x)
{
    return f0 * (1-x) + f1 * x;
}

/*
 * Quadratic interpolation with extra area in negative direction
 */
static double quadratic_interpolation_neg(double fm1, double f0, double f1, double x)
{
    double a, b, c;
    c = f0;
    b = (f1 - fm1)/2;
    a = f1 - b - c;
    return a*x*x + b*x + c;
}

/*
 * Quadratic interpolation with extra area in positive direction
 */
static double quadratic_interpolation_pos(double f0, double f1, double f2, double x)
{
    double a, b, c;
    c = f0;
    b = (4*f1 - f2 - 3*c)/2;
    a = f1 - b - c;
    return a*x*x + b*x + c;
}

/*
 * Cubic interpolation. x lies between points of f0 and f1
 */
static double cubic_interpolation(double fm1, double f0, double f1, double f2, double x)
{
    double a, b, c, d;
    d = f0;
    b = (fm1 + f1) / 2 - d;
    a = (f2 - 2*f1 - 2*b + d) / 6;
    c = f1 - b - d - a;
    return a*x*x*x + b*x*x + c*x + d;
}

double image_wave_interpolation_1d(double fm1, double f0, double f1, double f2, double x)
{
    if (isnan(f0))
    {
        if(x < 1-1e-12)
            return NAN;
        else
            return f1;
    }
    if (isnan(f1))
    {
        if (x > 1e-12)
            return NAN;
        else
            return f0;
    }
    if (isnan(fm1) && isnan(f2))
    {
        return linear_interpolation(f0, f1, x);
    }
    if (isnan(fm1))
    {
        return quadratic_interpolation_pos(f0, f1, f2, x);
    }
    if (isnan(f2))
    {
        return quadratic_interpolation_neg(fm1, f0, f1, x);
    }
    return cubic_interpolation(fm1, f0, f1, f2, x);
}

double image_wave_interpolation_2d(double fm1m1, double f0m1, double f1m1, double f2m1,
                                   double fm10,  double f00,  double f10,  double f20,
                                   double fm11,  double f01,  double f11,  double f21,
                                   double fm12,  double f02,  double f12,  double f22,
                                   double x, double y)
{
    double fm1 = image_wave_interpolation_1d(fm1m1, fm10, fm11, fm12, y);
    double f0  = image_wave_interpolation_1d(f0m1,  f00,  f01,  f02,  y);
    double f1  = image_wave_interpolation_1d(f1m1,  f10,  f11,  f12,  y);
    double f2  = image_wave_interpolation_1d(f2m1,  f20,  f21,  f22,  y);
    return image_wave_interpolation_1d(fm1, f0, f1, f2, x);
}


/*
 * Bicubic interpolation. Use linear interpolation for x and y axes
 */
double image_wave_interpolation(const struct ImageWaveGrid *array,
                                int xi, int yi, int axis, 
                                double dx, double dy)
{
    double x_m1m1 = image_wave_get_array(array, xi-1, yi-1, axis);
    double x_0m1 = image_wave_get_array(array, xi, yi-1, axis);
    double x_1m1 = image_wave_get_array(array, xi+1, yi-1, axis);
    double x_2m1 = image_wave_get_array(array, xi+2, yi-1, axis);

    double x_m10 = image_wave_get_array(array, xi-1, yi, axis);
    double x_00 = image_wave_get_array(array, xi, yi, axis);
    double x_10 = image_wave_get_array(array, xi+1, yi, axis);
    double x_20 = image_wave_get_array(array, xi+2, yi, axis);

    double x_m11 = image_wave_get_array(array, xi-1, yi+1, axis);
    double x_01 = image_wave_get_array(array, xi, yi+1, axis);
    double x_11 = image_wave_get_array(array, xi+1, yi+1, axis);
    double x_21 = image_wave_get_array(array, xi+2, yi+1, axis);
    
    double x_m12 = image_wave_get_array(array, xi-1, yi+2, axis);
    double x_02 = image_wave_get_array(array, xi, yi+2, axis);
    double x_12 = image_wave_get_array(array, xi+1, yi+2, axis);
    double x_22 = image_wave_get_array(array, xi+2, yi+2, axis);

    return image_wave_interpolation_2d(x_m1m1, x_0m1, x_1m1, x_2m1,
                                       x_m10,  x_00,  x_10,  x_20,
                                       x_m11,  x_01,  x_11,  x_21,
                                       x_m12,  x_02,  x_12,  x_22,
                                       dx, dy);
}

/*
 * Interpolate values in shift array
 */
static void interpolation(const struct ImageWaveGrid *array,
                          int xi, int yi, double dx, double dy,
                          double *shift_x, double *shift_y)
{
    *shift_x = image_wave_interpolation(array, xi, yi, 0, dx, dy);
    *shift_y = image_wave_interpolation(array, xi, yi, 1, dx, dy);
}

void image_wave_shift_interpolate(struct ImageWave *self,
                                 struct ImageWaveGrid *array,
                                 double x, double y,
                                 double *rx, double *ry)
{
    double sx = self->sx;
    double sy = self->sy;

    int xi = floor(x/sx);
    int yi = floor(y/sy);

    double dx = x/sx - xi;
    double dy = y/sy - yi;

    double shift_x, shift_y;

    interpolation(array, xi, yi, dx, dy, &shift_x, &shift_y);

    *rx = x + shift_x;
    *ry = y + shift_y;
}
