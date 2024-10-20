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

struct OrthographicProjection
{
    int w;        // image width
    int h;        // image height
    double a;     // planet ellipse major axis
    double b;     // planet ellipse minor axis
    double angle; // planet ellipse slope
    double rot;   // planet rotation angle
};

bool orthographic_projection_init(struct OrthographicProjection *self,
                                  int w, int h,
                                  double a, double b,
                                  double angle, double rot);

bool orthographic_projection_project(void *self,
                                     double y, double x,
                                     double *lat, double *lon);

bool orthographic_projection_reverse(void *self,
                                     double lat, double lon,
                                     double *y, double *x);
