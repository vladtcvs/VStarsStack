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

#include "equirectangular.h"

bool equirectangular_projection_init(struct EquirectangularProjection *self, int w, int h)
{
    if (h <= 0 || w <= 0)
        return false;

    self->w = w;
    self->h = h;
    return true;
}

bool equirectangular_projection_project(void *self,
                                        double y, double x,
                                        double *lat, double *lon)
{
    const struct EquirectangularProjection *proj = (const struct EquirectangularProjection *)self;
    *lon = (1 - 2*x/proj->w) * M_PI;
    *lat = (1 - 2*y/proj->h) * M_PI_2;
    return true;
}

bool equirectangular_projection_reverse(void *self,
                                        double lat, double lon,
                                        double *y, double *x)
{
    const struct EquirectangularProjection *proj = (const struct EquirectangularProjection *)self;
    *x = (1 - lon / M_PI)/2*proj->w;
    *y = (1 - lat / M_PI_2)/2*proj->h;
    return true;
}
