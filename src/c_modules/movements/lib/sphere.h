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

#include <unistd.h>
#include "lib/projection.h"

struct quat {
    double w, x, y, z;
};

struct SphereMovement
{
    struct quat forward; // forward quaternion
    struct quat reverse; // reverse quaternion
};

struct ProjectionDef
{
    void *projection;
    forward_project_f forward;
    reverse_project_f reverse;
};

void sphere_movement_init(struct SphereMovement *mov, struct quat q);

void sphere_movement_apply_forward(struct SphereMovement *mov,
                                   const double *posi, double *poso, size_t num,
                                   const struct ProjectionDef *in_proj,
                                   const struct ProjectionDef *out_proj);

void sphere_movement_apply_forward_lonlat(struct SphereMovement *mov,
                                          const double *posi, double *poso, size_t num);

void sphere_movement_apply_reverse(struct SphereMovement *mov,
                                   const double *posi, double *poso, size_t num,
                                   const struct ProjectionDef *in_proj,
                                   const struct ProjectionDef *out_proj);

void sphere_movement_apply_reverse_lonlat(struct SphereMovement *mov,
                                          const double *posi, double *poso, size_t num);
