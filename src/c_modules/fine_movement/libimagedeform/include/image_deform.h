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

#include <stdio.h>
#include <math.h>

#include <image_grid.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Image movements grid
 */
struct ImageDeform
{
    int image_w;    ///< Image width
    int image_h;    ///< Image height
    int grid_w;     ///< Grid width
    int grid_h;     ///< Grid height
    real_t sx;      ///< Stretch between grid and image
    real_t sy;      ///< Stretch between grid and image
    real_t *array;  ///< Movement data, grid_h * grid_w * (dy, dx)
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
void image_deform_print(const struct ImageDeform *grid, FILE *out);

/**
 * \brief Set image deform with constant (dx, dy)
 * \param grid Already allocated image deform object
 * \param dx dx
 * \param dy dy
 */
void image_deform_constant_shift(struct ImageDeform *grid, real_t dx, real_t dy);

/**
 * \brief Set shift at pos (x,y) for axis y (axis=0) or x (axis=1)
 * \param deform Shift array
 * \param x x in grid coordinates
 * \param y y in grid coordinates
 * \param axis axis. 0 for 'y' axis, 1 for 'x' axis
 * \param val value
 */
static inline void image_deform_set_shift(struct ImageDeform *deform,
                                          int x, int y, int axis,
                                          real_t val)
{
    if (x < 0 || y < 0 || x >= deform->grid_w || x >= deform->grid_w || axis < 0 || axis > 1)
        return;
    deform->array[y * deform->grid_w * 2 + x*2 + axis] = val;
}

/**
 * \brief Fill image deform by shift data
 */
static inline void image_deform_set_shifts(struct ImageDeform *deform, const real_t *data)
{
    memcpy(deform->array, data, deform->grid_h*deform->grid_w*2*sizeof(real_t));
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
                              int x, int y, int axis);

/**
 * @brief Get shift at pos (x,y)
 *
 * @param deform Shift array
 * @param x x in grid coordinates
 * @param y y in grid coordinates
 * @param axis axis. 0 for 'y' axis, 1 for 'x' axis
 * @return shift by specified axis
 */
real_t image_deform_get_shift(const struct ImageDeform *deform,
                              real_t x, real_t y, int axis);

/**
 * \brief Apply image deformation to point
 * \param[in] deform deformation structure
 * \param[in] x x in image coordinates
 * \param[in] y y in image coordinates
 * \param[out] srcx source x positionx in image coordinates
 * \param[out] srcy soutce y positionx in image coordinates
 */
void image_deform_apply_point(const struct ImageDeform *deform,
                              real_t x, real_t y,
                              real_t *srcx, real_t *srcy);

/**
 * \brief Apply image deformation
 * \param deform deformation structure
 * \param input_image original image
 * \param output_image resulting image
 */
void image_deform_apply_image(const struct ImageDeform *deform,
                              const struct ImageGrid *input_image,
                              struct ImageGrid *output_image);

/**
 * \brief Generate image deform density
 * \param deform deformation sructure
 * \param divergence output image for divergence
 */
void image_deform_calculate_divergence(const struct ImageDeform *deform,
                                       struct ImageGrid *divergence);

#ifdef __cplusplus
}
#endif
