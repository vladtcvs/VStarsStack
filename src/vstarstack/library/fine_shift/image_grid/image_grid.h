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

/**
 * \brief Image data
 */
struct ImageGrid
{
    int w;         ///< Image width
    int h;         ///< Image height
    double *array; ///< Image pixel data
};

/**
 * \brief Set pixel at pos (x,y)
 * \param image Image array
 * \param x x
 * \param y y
 * \param val pixel value
 */
static inline void image_grid_set_pixel(struct ImageGrid *image,
                                        int x, int y, double val)
{
    image->array[y * image->w + x] = val;
}

/**
 * \brief Get pixel at pos (x,y)
 * \param image Image array
 * \param x x
 * \param y y
 * \return pixel value
 */
double image_grid_get_pixel(const struct ImageGrid *image,
                            double x, double y);

/**
 * \brief Global correlation between 2 images
 * \param image1 first image
 * \param image2 second image
 * \return correlator
 */
double image_grid_correlation(const struct ImageGrid *image1,
                              const struct ImageGrid *image2);
