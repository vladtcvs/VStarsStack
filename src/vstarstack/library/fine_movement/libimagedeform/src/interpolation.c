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

#include <interpolation.h>

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

double interpolation_1d_cubic(double fm1, double f0, double f1, double f2, double x)
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

double interpolation_2d_cubic(double fm1m1, double f0m1, double f1m1, double f2m1,
                              double fm10,  double f00,  double f10,  double f20,
                              double fm11,  double f01,  double f11,  double f21,
                              double fm12,  double f02,  double f12,  double f22,
                              double x, double y)
{
    double fm1 = interpolation_1d_cubic(fm1m1, fm10, fm11, fm12, y);
    double f0  = interpolation_1d_cubic(f0m1,  f00,  f01,  f02,  y);
    double f1  = interpolation_1d_cubic(f1m1,  f10,  f11,  f12,  y);
    double f2  = interpolation_1d_cubic(f2m1,  f20,  f21,  f22,  y);
    return interpolation_1d_cubic(fm1, f0, f1, f2, x);
}

double interpolation_1d_linear(double f0, double f1, double x)
{
    return linear_interpolation(f0, f1, x);
}

double interpolation_2d_linear(double f00,  double f10,
                               double f01,  double f11,
                               double x, double y)
{
    double f0 = interpolation_1d_linear(f00, f01, y);
    double f1 = interpolation_1d_linear(f10, f11, y);
    return interpolation_1d_linear(f0, f1, x);
}
