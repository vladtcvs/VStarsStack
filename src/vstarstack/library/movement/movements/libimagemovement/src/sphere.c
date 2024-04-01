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
#include <sphere.h>

static struct quat quat_inv(const struct quat q)
{
    double len2 = q.w*q.w+q.x*q.x+q.y*q.y+q.z*q.z;
    struct quat rev = {
        .w = q.w/len2,
        .x = -q.x/len2,
        .y = -q.y/len2,
        .z = -q.z/len2,
    };
    return rev;
}

static struct quat quat_mul(const struct quat a, const struct quat b)
{
    struct quat q = {
        .w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        .x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        .y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        .z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    };
    return q;
}

static struct quat quat_from_vec(double x, double y, double z)
{
    struct quat q = {
        .w = 0,
        .x = x,
        .y = y,
        .z = z,
    };
    return q;
}

void sphere_movement_init(struct SphereMovement *mov, struct quat q)
{
    mov->forward = q;
    mov->reverse = quat_inv(q);
}

void sphere_movement_apply_forward(struct SphereMovement *mov,
                                   const double *posi, double *poso, size_t num,
                                   const struct ProjectionDef *in_proj,
                                   const struct ProjectionDef *out_proj)
{
    unsigned i;
    for (i = 0; i < num; i++)
    {
        double lon, lat;
        double x, y, z;
        double xi = posi[i*2];
        double yi = posi[i*2+1];
        in_proj->forward(in_proj->projection, yi, xi, &lat, &lon);
        x = cos(lon)*cos(lat);
        y = sin(lon)*cos(lat);
        z = sin(lat);
        struct quat v = quat_from_vec(x, y, z);
        struct quat vf = quat_mul(mov->forward, quat_mul(v, mov->reverse));
        x = vf.x;
        y = vf.y;
        z = vf.z;
        if (z > 1)
            z = 1;
        if (z < -1)
            z = -1;
        lon = atan2(y, x);
        lat = asin(z);
        double yo, xo;
        out_proj->reverse(out_proj->projection, lat, lon, &yo, &xo);
        poso[2*i] = xo;
        poso[2*i+1] = yo;
    }
}

void sphere_movement_apply_reverse(struct SphereMovement *mov,
                                   const double *posi, double *poso, size_t num,
                                   const struct ProjectionDef *in_proj,
                                   const struct ProjectionDef *out_proj)
{
    unsigned i;
    for (i = 0; i < num; i++)
    {
        double lon, lat;
        double x, y, z;
        double xi = posi[i*2];
        double yi = posi[i*2+1];
        out_proj->forward(out_proj->projection, yi, xi, &lat, &lon);
        x = cos(lon)*cos(lat);
        y = sin(lon)*cos(lat);
        z = sin(lat);
        struct quat v = quat_from_vec(x, y, z);
        struct quat vf = quat_mul(mov->reverse, quat_mul(v, mov->forward));
        x = vf.x;
        y = vf.y;
        z = vf.z;
        if (z > 1)
            z = 1;
        if (z < -1)
            z = -1;
        lon = atan2(y, x);
        lat = asin(z);
        double yo, xo;
        in_proj->reverse(in_proj->projection, lat, lon, &yo, &xo);
        poso[2*i] = xo;
        poso[2*i+1] = yo;
    }
}
