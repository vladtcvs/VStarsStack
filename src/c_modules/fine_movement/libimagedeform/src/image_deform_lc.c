/*
 * Copyright (c) 2022-2025 Vladislav Tsendrovskii
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

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include <image_deform_lc.h>

int  image_deform_lc_init(struct ImageDeformLocalCorrelator *self,
                          int image_w, int image_h, int pixels,
                          const char *kernel_source)
{
    self->image_w = image_w;
    self->image_h = image_h;
    self->pixels = pixels;
    self->grid_w = ceil((real_t)image_w/pixels);
    self->grid_h = ceil((real_t)image_h/pixels);
    if (image_deform_init(&self->array, self->grid_w, self->grid_h, self->image_w, self->image_h) != 0)
        return -1;
    self->use_opencl = false;
#ifdef USE_OPENCL
    if (kernel_source)
    {
        self->use_opencl = true;
        int clerr = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_DEFAULT, 1, &self->device_id, NULL);
        if (clerr != CL_SUCCESS)
        {
            printf("Error: Failed to create an OpenCL device group!\n");
            return -1;
        }

        self->context = clCreateContext(0, 1, &self->device_id, NULL, NULL, &clerr);
        if (!self->context)
        {
            printf("Error: Failed to create a compute context!\n");
            return -1;
        }

        self->commands = clCreateCommandQueue(self->context, self->device_id, 0, &clerr);
        if (!self->commands)
        {
            printf("Error: Failed to create a command commands!\n");
            clReleaseContext(self->context);
            return -1;
        }

        self->program = clCreateProgramWithSource(self->context, 1, (const char **)&kernel_source, NULL, &clerr);
        if (!self->program)
        {
            printf("Error: Failed to create compute program!\n");
            clReleaseCommandQueue(self->commands);
            clReleaseContext(self->context);
            return -1;
        }

        clerr = clBuildProgram(self->program, 0, NULL, NULL, NULL, NULL);
        if (clerr != CL_SUCCESS)
        {
            size_t len;
            char buffer[2048];

            printf("Error: Failed to build program executable!\n");
            clGetProgramBuildInfo(self->program, self->device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
            printf("%s\n", buffer);

            clReleaseProgram(self->program);
            clReleaseCommandQueue(self->commands);
            clReleaseContext(self->context);
            return -1;
        }

        self->kernel_constant_shift = clCreateKernel(self->program, "image_deform_lc_constant", &clerr);
        if (!self->kernel_constant_shift || clerr != CL_SUCCESS)
        {
            printf("Error: Failed to create compute kernel \"image_deform_lc_constant\"!\n");
            clReleaseProgram(self->program);
            clReleaseCommandQueue(self->commands);
            clReleaseContext(self->context);
            return -1;
        }

        self->kernel_grid_shift = clCreateKernel(self->program, "image_deform_lc_grid", &clerr);
        if (!self->kernel_grid_shift || clerr != CL_SUCCESS)
        {
            printf("Error: Failed to create compute kernel \"image_deform_lc_grid\"!\n");
            clReleaseKernel(self->kernel_constant_shift);
            clReleaseProgram(self->program);
            clReleaseCommandQueue(self->commands);
            clReleaseContext(self->context);
            return -1;
        }
    }

#endif // USE_OPENCL
    return 0;
}

void image_deform_lc_finalize(struct ImageDeformLocalCorrelator *self)
{
#ifdef USE_OPENCL
    if (self->use_opencl) {
        clReleaseKernel(self->kernel_grid_shift);
        clReleaseKernel(self->kernel_constant_shift);
        clReleaseProgram(self->program);
        clReleaseCommandQueue(self->commands);
        clReleaseContext(self->context);
        if (self->img_buf)
            clReleaseMemObject(self->img_buf);
        if (self->ref_img_buf)
            clReleaseMemObject(self->ref_img_buf);
        if (self->correlations_buf_const)
            clReleaseMemObject(self->correlations_buf_const);
        if (self->correlations_const)
            free(self->correlations_const);
        self->img_buf = NULL;
        self->ref_img_buf = NULL;
        self->correlations_buf_const = NULL;
        self->correlations_const = NULL;
    }
#endif // USE_OPENCL

    image_deform_finalize(&self->array);
}

void image_deform_lc_get_area(const struct ImageGrid *img,
                              const struct ImageDeform *pre_align,
                              struct ImageGrid *area,
                              real_t x, real_t y)
{
    real_t pre_aligned_x;
    real_t pre_aligned_y;
    if (pre_align == NULL)
    {
        pre_aligned_x = x;
        pre_aligned_y = y;
    }
    else
    {
        image_deform_apply_point(pre_align, x, y, &pre_aligned_x, &pre_aligned_y);
    }

    image_grid_get_area(img, pre_aligned_x, pre_aligned_y, area);
}
