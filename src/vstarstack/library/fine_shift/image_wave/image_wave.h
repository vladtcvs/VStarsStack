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

#include "image_grid.h"

/**
 * \brief Global correlation between 2 images
 * \param image1 first image
 * \param image2 second image
 * \return correlator
 */
double image_wave_gc_correlation(const struct ImageGrid *image1,
                                 const struct ImageGrid *image2);


/**
 * \brief Image movements grid
 */
struct ImageWaveGrid
{
    int image_w;    ///< Image width
    int image_h;    ///< Image height
    int grid_w;     ///< Grid width
    int grid_h;     ///< Grid height
    double *array;  ///< Movement data, grid_h x grid_w x 2
};

/**
 * \brief Init image wave
 * \param grid   Image wave structure
 * \param grid_w Grid width
 * \param grid_h Grid height
 * \param image_w Image width
 * \param image_height
 */
int image_wave_grid_init(struct ImageWaveGrid *grid,
                         int grid_w, int grid_h,
                         int image_w, int image_h);

/**
 * \brief Deallocate image wave grid
 * \param grid image wave grid object
 */
void image_wave_grid_finalize(struct ImageWaveGrid *grid);

/**
 * \brief Print image wave grid to stdout
 * \param grid image wave grid object
 */
void image_wave_grid_print(const struct ImageWaveGrid *grid);

/**
 * \brief Set image wave grid with constant (dx, dy)
 * \param grid Already allocated image wave grid object
 * \param dx dx
 * \param dy dy
 */
void image_wave_grid_constant_shift(struct ImageWaveGrid *grid, double dx, double dy);

/**
 * \brief Set shift at pos (x,y) for axis 1 or 2
 * \param grid Shift array
 * \param x x
 * \param y y
 * \param axis axis (0 for 'y' axis, 1 for 'x' axis)
 */
static inline void image_wave_set_array(struct ImageWaveGrid *grid,
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
 */
static inline double image_wave_get_array(const struct ImageWaveGrid *grid,
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
