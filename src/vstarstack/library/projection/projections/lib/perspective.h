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

struct PerspectiveProjection
{
    double H;       // height of image in mm
    double W;       // width of image in mm
    double F;       // focal length in mm
    int h;          // height of image in pixels
    int w;          // width of image in pixels
    double kx;      // transformation from pixels to mm coefficient
    double ky;      // transformation from pixels to mm coefficient
};

bool perspective_projection_init(struct PerspectiveProjection *self,
                                 double kw, double kh, double F,
                                 double w, double h);

bool perspective_projection_project(void *self,
                                    double y, double x,
                                    double *lat, double *lon);

bool perspective_projection_reverse(void *self,
                                    double lat, double lon,
                                    double *y, double *x);
