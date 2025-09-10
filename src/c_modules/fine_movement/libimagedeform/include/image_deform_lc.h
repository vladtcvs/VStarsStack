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

#include <image_deform.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Helper structure to find movement grid by local correlation
 */
struct ImageDeformLocalCorrelator
{
    struct ImageDeform array;   ///< Movements grid
    int image_w;                ///< Image width
    int image_h;                ///< Image height
    int grid_w;                 ///< Grid width
    int grid_h;                 ///< Grid height
    int pixels;
};

/**
 * \brief Init ImageDeformLocalCorrelator
 * \param self structure pointer
 * \param image_w image width
 * \param image_h image heigth
 * \param pixels image pixels per grid step
 * \return 0 for success, -1 for fail
 */
int  image_deform_lc_init(struct ImageDeformLocalCorrelator *self,
                          int image_w, int image_h, int pixels);

/**
 * \brief Deallocate content of ImageDeformLocalCorrelation
 * \param self structure pointer
 */
void image_deform_lc_finalize(struct ImageDeformLocalCorrelator *self);

/**
 * \brief Find constant correlator
 */
void image_deform_lc_find_constant(struct ImageDeformLocalCorrelator *self,
                                   const struct ImageGrid *img,
                                   const struct ImageDeform *pre_align,
                                   const struct ImageGrid *ref_img,
                                   const struct ImageDeform *ref_pre_align,
                                   real_t maximal_shift,
                                   int subpixels);

/**
 * \brief Find correlator
 */
void image_deform_lc_find(struct ImageDeformLocalCorrelator *self,
                          const struct ImageGrid *img,
                          const struct ImageDeform *pre_align,
                          const struct ImageGrid *ref_img,
                          const struct ImageDeform *ref_pre_align,
                          int radius,
                          real_t maximal_shift,
                          int subpixels);

#ifdef __cplusplus
}
#endif
