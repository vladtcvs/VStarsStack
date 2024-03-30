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

#pragma once

#include <math.h>
#include <stddef.h>

#include <image_grid.h>

/**
 * \brief Image movements grid
 */
struct ImageDeform
{
    int image_w;    ///< Image width
    int image_h;    ///< Image height
    int grid_w;     ///< Grid width
    int grid_h;     ///< Grid height
    double sx;      ///< Stretch between grid and image
    double sy;      ///< Stretch between grid and image
    double *array;  ///< Movement data, grid_h x grid_w x 2
};

/**
 * \brief Init image deform
 * \param grid   Image deform structure
 * \param grid_w Grid width
 * \param grid_h Grid height
 * \param image_w Image width
 * \param image_height Image height
 */
int image_deform_init(struct ImageDeform *grid,
                      int grid_w, int grid_h,
                      int image_w, int image_h);

/**
 * \brief Deallocate image deform
 * \param grid image deform object
 */
void image_deform_finalize(struct ImageDeform *grid);

/**
 * \brief Print image deform to stdout
 * \param grid image deform object
 */
void image_deform_print(const struct ImageDeform *grid);

/**
 * \brief Set image deform with constant (dx, dy)
 * \param grid Already allocated image deform object
 * \param dx dx
 * \param dy dy
 */
void image_deform_constant_shift(struct ImageDeform *grid, double dx, double dy);

/**
 * \brief Set shift at pos (x,y) for axis 1 or 2
 * \param grid Shift array
 * \param x x
 * \param y y
 * \param axis axis (0 for 'y' axis, 1 for 'x' axis)
 * \param val value
 */
static inline void image_deform_set_array(struct ImageDeform *grid,
                                         int x, int y, int axis,
                                         double val)
{
    grid->array[y * grid->grid_w * 2 + x*2 + axis] = val;
}

/**
 * \brief Get shift at pos (x,y) for axis 1 or 2
 * \param grid Shift array
 * \param x x
 * \param y y
 * \param axis axis (0 for 'y' axis, 1 for 'x' axis)
 * \return value
 */
static inline double image_deform_get_array(const struct ImageDeform *grid,
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
    return grid->array[y * grid->grid_w * 2 + x*2 + axis];
}

/**
 * \brief Apply image deformation to point
 * \param[in] deform deformation structure
 * \param[in] x target x position
 * \param[in] y target y position
 * \param[in] subpixels upscaling coefficient of image during deform
 * \param[out] srcx source x position
 * \param[out] srcy soutce y position
 */
void image_deform_apply_point(struct ImageDeform *deform,
                              double x, double y,
                              double *srcx, double *srcy);

/**
 * \brief Apply image deformation
 * \param deform deformation structure
 * \param input_image original image
 * \param output_image resulting image
 * \param subpixels upscaling coefficient of image during deform
 */
void image_deform_apply_image(struct ImageDeform *deform,
                              const struct ImageGrid *input_image,
                              struct ImageGrid *output_image,
                              int subpixels);
