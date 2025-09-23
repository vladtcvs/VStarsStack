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
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define USE_OPENCL

#ifdef USE_OPENCL
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
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
    bool use_opencl;            ///< Use OpenCL

#ifdef USE_OPENCL
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel_constant_shift;    // compute kernel
    cl_kernel kernel_grid_shift;        // compute kernel

    cl_mem img_buf;
    cl_mem ref_img_buf;
    cl_mem correlations_buf_const;
    float *correlations_const;

    cl_mem correlations_buf_grid;


    int maximal_shift;
    int subpixels;
#endif
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
                          int image_w, int image_h, int pixels,
                          const char *kernel_source);

/**
 * \brief Deallocate content of ImageDeformLocalCorrelation
 * \param self structure pointer
 */
void image_deform_lc_finalize(struct ImageDeformLocalCorrelator *self);

/**
 * \brief Find constant correlator
 */
int image_deform_lc_find_constant(struct ImageDeformLocalCorrelator *self,
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

void image_deform_lc_get_area(const struct ImageGrid *img,
                              const struct ImageDeform *pre_align,
                              struct ImageGrid *area,
                              real_t x, real_t y);

#define UNUSED(x) ((void)(x))

#ifdef __cplusplus
}
#endif
