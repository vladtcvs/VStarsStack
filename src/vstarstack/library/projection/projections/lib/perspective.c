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

#include <stdbool.h>
#include <math.h>

#include "perspective.h"

bool perspective_projection_init(struct PerspectiveProjection *self,
                                 double W, double H, double F,
                                 double w, double h)
{
    if (h <= 0 || w <= 0 ||
        W <= 0 || H <= 0 ||
        F <= 0)
    {
        return false;
    }

    self->W = W;
    self->H = H;
    self->F = F;
    self->w = w;
    self->h = h;

    self->kx = self->W / self->w;
    self->ky = self->H / self->h;
    return true;
}

bool perspective_projection_project(void *self,
                                    double y, double x,
                                    double *lat, double *lon)
{
    const struct PerspectiveProjection *proj = (const struct PerspectiveProjection *)self;
    double X = (proj->w / 2 - x) * proj->kx;
    double Y = (proj->h / 2 - y) * proj->ky;
    *lon = atan(X / proj->F);
    *lat = atan(Y * cos(*lon) / proj->F);
    return true;
}

bool perspective_projection_reverse(void *self,
                                    double lat, double lon,
                                    double *y, double *x)
{
    const struct PerspectiveProjection *proj = (const struct PerspectiveProjection *)self;
    double X = proj->F * tan(lon);
    double Y = proj->F * tan(lat) / cos(lon);
    *x = proj->w / 2 - X / proj->kx;
    *y = proj->h / 2 - Y / proj->ky;
    return true;
}
