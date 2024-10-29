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

#pragma once

#include "projection_common.h"

struct EquirectangularProjection
{
    int h;          // height of image in pixels
    int w;          // width of image in pixels
};

bool equirectangular_projection_init(struct EquirectangularProjection *self, int w, int h);

bool equirectangular_projection_project(void *self,
                                        double y, double x,
                                        double *lat, double *lon);

bool equirectangular_projection_reverse(void *self,
                                        double lat, double lon,
                                        double *y, double *x);
