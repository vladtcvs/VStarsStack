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

#include "orthographic.h"
#include <stdio.h>

bool orthographic_projection_init(struct OrthographicProjection *self,
                                  int w, int h,
                                  double a, double b,
                                  double angle, double rot)
{
    if (h <= 0 || w <= 0 || a <= 0 || b <= 0)
        return false;

    self->a = a;
    self->b = b;
    self->angle = angle;
    self->rot = rot;
    self->w = w;
    self->h = h;
    return true;
}

bool orthographic_projection_project(void *self,
                                     double y, double x,
                                     double *lat, double *lon)
{
    const struct OrthographicProjection *proj = (const struct OrthographicProjection *)self;

    x -= proj->w/2;
    y -= proj->h/2;
    double X = x*cos(proj->angle) - y*sin(proj->angle);
    double Y = x*sin(proj->angle) + y*cos(proj->angle);

    double sin_lat = Y / (proj->b/2);
    if (fabs(sin_lat) > 1)
        return false;

    *lat = -asin(sin_lat);
    if (fabs(cos(*lat)) < 1e-6)
    {
        if (fabs(X) < 1e-4)
        {
            *lon = 0;
            return true;
        }
        return false;
    }

    double sin_lon = X / (proj->a/2) / cos(*lat);
    if (fabs(sin_lon) > 1)
        return false;

    *lon = asin(sin_lon) + proj->rot;
    return true;
}

bool orthographic_projection_reverse(void *self,
                                     double lat, double lon,
                                     double *y, double *x)
{
    const struct OrthographicProjection *proj = (const struct OrthographicProjection *)self;

    double X = proj->a/2 * sin(lon - proj->rot) * cos(lat);
    double Z = -proj->b/2 * sin(lat);
    *x = X*cos(proj->angle) + Z*sin(proj->angle) + proj->w/2;
    *y = -X*sin(proj->angle) + Z*cos(proj->angle) + proj->h/2;
    return true;
}
