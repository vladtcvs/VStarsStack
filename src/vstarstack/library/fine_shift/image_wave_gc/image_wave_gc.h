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

#include "../image_wave/image_wave.h"

/**
 * \brief Helper structure to find movement grid by global correlation
 */
struct ImageWaveGlobalCorrelator
{
    struct ImageWaveGrid array;          ///< Movements grid
    struct ImageWaveGrid array_p;        ///< Helper field for gradient calculation
    struct ImageWaveGrid array_m;        ///< Helper field for gradient calculation
    struct ImageWaveGrid array_gradient; ///< Global correlation gradient

    int image_w;                ///< Image width 
    int image_h;                ///< Image height
    int grid_w;                 ///< Grid width
    int grid_h;                 ///< Grid height
    double sx;                  ///< Stretch between grid and image
    double sy;                  ///< Stretch between grid and image
    double stretch_penalty_k;   ///< Penalty for image stretching
};

/**
 * \brief Init ImageWaveGlobalCorrelator
 * \param self structure pointer
 * \param grid_w grid width
 * \param grid_h grid height
 * \param image_w image width
 * \param image_h image heigth
 * \param spk penalty for image stretching
 */
int  image_wave_gc_init(struct ImageWaveGlobalCorrelator *self,
                        int grid_w, int grid_h,
                        int image_w, int image_h,
                        double spk);

/**
 * \brief Deallocate content of ImageWaveGlobalCorrelation
 * \param self structure pointer
 */
void image_wave_gc_finalize(struct ImageWaveGlobalCorrelator *self);
