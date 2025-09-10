/*
 * Copyright (c) 2023-2024 Vladislav Tsendrovskii
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

#include <stdio.h>
#include <stdlib.h>
#include <image_deform.h>
#include <interpolation.h>

int image_deform_init(struct ImageDeform *deform,
                      int grid_w, int grid_h,
                      int image_w, int image_h)
{
    deform->grid_w = grid_w;
    deform->grid_h = grid_h;
    deform->image_w = image_w;
    deform->image_h = image_h;
    deform->sx = (real_t)(grid_w-1) / (image_w - 1);
    deform->sy = (real_t)(grid_h-1) / (image_h - 1);
    deform->array = calloc(deform->grid_w * deform->grid_h * 2, sizeof(real_t));
    if (deform->array == NULL)
        return -1;
    return 0;
}

void image_deform_finalize(struct ImageDeform *deform)
{
    if (deform->array != NULL)
    {
        free(deform->array);
        deform->array = NULL;
    }
}

void image_deform_constant_shift(struct ImageDeform *deform, real_t dx, real_t dy)
{
    int xi, yi;
    for (yi = 0; yi < deform->grid_h; yi++)
    {
        for (xi = 0; xi < deform->grid_w; xi++)
        {
            image_deform_set_shift(deform, xi, yi, 0, dy);
            image_deform_set_shift(deform, xi, yi, 1, dx);
        }
    }
}

/**
 * \brief Get shift at pos (x,y) for axis 1 or 2
 * \param grid Shift array
 * \param x x
 * \param y y
 * \param axis axis (0 for 'y' axis, 1 for 'x' axis)
 * \return value
 */
real_t image_deform_get_array(const struct ImageDeform *grid,
                              int x, int y, int axis)
{
    if (x >= grid->grid_w)
        return NAN;
    if (x < 0)
        return NAN;
    if (y >= grid->grid_h)
        return NAN;
    if (y < 0)
        return NAN;
    if (axis != 0 && axis != 1)
        return NAN;
    return grid->array[y * grid->grid_w * 2 + x*2 + axis];
}

void image_deform_print(const struct ImageDeform *deform, FILE *out)
{
    int i, j, k;
    for (i = 0; i < deform->grid_h; i++)
    {
        for (j = 0; j < deform->grid_w; j++)
        {
            fprintf(out, "[");
            for (k = 0; k < 2; k++)
                fprintf(out, "%lf ", image_deform_get_array(deform, j, i, k));
            fprintf(out, "]");
        }
        fprintf(out, "\n");
    }
}

real_t image_deform_get_shift(const struct ImageDeform *deform,
                              real_t x, real_t y, int axis)
{
    int xi = floor(x);
    int yi = floor(y);

    real_t dx = x - xi;
    real_t dy = y - yi;
    if (dx < 1e-3 && dy < 1e-3)
    {
        return image_deform_get_array(deform, xi, yi, axis);
    }
    else
    {
        real_t x_m1m1 = image_deform_get_array(deform, xi-1, yi-1, axis);
        real_t x_m10  = image_deform_get_array(deform, xi-1, yi, axis);
        real_t x_m11  = image_deform_get_array(deform, xi-1, yi+1, axis);
        real_t x_m12  = image_deform_get_array(deform, xi-1, yi+2, axis);

        real_t x_0m1 = image_deform_get_array(deform, xi, yi-1, axis);
        real_t x_00  = image_deform_get_array(deform, xi, yi, axis);
        real_t x_01  = image_deform_get_array(deform, xi, yi+1, axis);
        real_t x_02  = image_deform_get_array(deform, xi, yi+2, axis);

        real_t x_1m1 = image_deform_get_array(deform, xi+1, yi-1, axis);
        real_t x_10  = image_deform_get_array(deform, xi+1, yi, axis);
        real_t x_11  = image_deform_get_array(deform, xi+1, yi+1, axis);
        real_t x_12  = image_deform_get_array(deform, xi+1, yi+2, axis);

        real_t x_2m1 = image_deform_get_array(deform, xi+2, yi-1, axis);
        real_t x_20  = image_deform_get_array(deform, xi+2, yi, axis);
        real_t x_21  = image_deform_get_array(deform, xi+2, yi+1, axis);
        real_t x_22  = image_deform_get_array(deform, xi+2, yi+2, axis);

        return interpolation_2d_cubic(x_m1m1, x_0m1, x_1m1, x_2m1,
                                      x_m10,  x_00,  x_10,  x_20,
                                      x_m11,  x_01,  x_11,  x_21,
                                      x_m12,  x_02,  x_12,  x_22,
                                      dx, dy);
    }
}

void image_deform_apply_point(const struct ImageDeform *deform,
                              real_t x, real_t y,
                              real_t *srcx, real_t *srcy)
{
    real_t shift_y = image_deform_get_shift(deform, x*deform->sx, y*deform->sy, 0);
    real_t shift_x = image_deform_get_shift(deform, x*deform->sx, y*deform->sy, 1);
    if (isnan(shift_x) || isnan(shift_y))
    {
        *srcx = NAN;
        *srcy = NAN;
    }
    else
    {
        *srcx = x + shift_x/deform->sx;
        *srcy = y + shift_y/deform->sy;
    }
}

void image_deform_apply_image(const struct ImageDeform *deform,
                              const struct ImageGrid *input_image,
                              struct ImageGrid *output_image)
{
    int y, x;
    real_t kx = (real_t)input_image->w / output_image->w;
    real_t ky = (real_t)input_image->h / output_image->h;
    for (y = 0; y < output_image->h; y++)
        for (x = 0; x < output_image->w; x++)
        {
            real_t orig_y, orig_x;
            image_deform_apply_point(deform, x*kx, y*ky, &orig_x, &orig_y);

            real_t val = image_grid_get_pixel(input_image, orig_x, orig_y);
            image_grid_set_pixel(output_image, x, y, val);
        }
}

void image_deform_calculate_divergence(const struct ImageDeform *deform,
                                       struct ImageGrid *divergence)
{
    int y, x;
    real_t kx = (real_t)deform->image_w / divergence->w;
    real_t ky = (real_t)deform->image_h / divergence->h;
    // Calculate density from original coordinates
    for (y = 0; y < divergence->h; y++)
    {
        real_t dy1, dy2;
        if (y == 0)
        {
            dy1 = 0;
            dy2 = 1;
        }
        else if (y == divergence->h-1)
        {
            dy1 = -1;
            dy2 = 0;
        }
        else
        {
            dy1 = -1;
            dy2 = 1;
        }

        for (x = 0; x < divergence->w; x++)
        {
            real_t dx1, dx2;
            if (x == 0)
            {
                dx1 = 0;
                dx2 = 1;
            }
            else if (x == divergence->w-1)
            {
                dx1 = -1;
                dx2 = 0;
            }
            else
            {
                dx1 = -1;
                dx2 = 1;
            }

            real_t vx1 = image_deform_get_shift(deform, (x+dx1)*kx*deform->sx, y*ky*deform->sy, 1);
            real_t vx2 = image_deform_get_shift(deform, (x+dx2)*kx*deform->sx, y*ky*deform->sy, 1);
            real_t ddx = (vx2 - vx1) / (dx2-dx1);

            real_t vy1 = image_deform_get_shift(deform, x*kx*deform->sx, (y+dy1)*ky*deform->sy, 0);
            real_t vy2 = image_deform_get_shift(deform, x*kx*deform->sx, (y+dy2)*ky*deform->sy, 0);
            real_t ddy = (vy2 - vy1) / (dy2-dy1);

            real_t div = ddx + ddy;

            image_grid_set_pixel(divergence, x, y, div);
        }
    }
}
