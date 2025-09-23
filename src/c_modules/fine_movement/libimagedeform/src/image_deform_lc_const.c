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

#include <image_deform_lc.h>

#ifdef USE_OPENCL
static int image_deform_lc_find_constant_ocl(struct ImageDeformLocalCorrelator *self,
                                              const struct ImageGrid *img,
                                              const struct ImageDeform *pre_align,
                                              const struct ImageGrid *ref_img,
                                              const struct ImageDeform *ref_pre_align,
                                              int maximal_shift,
                                              int subpixels)
{
    
    size_t local;

    if (img->h != self->image_h || img->w != self->image_w) {
        if (self->img_buf)
            clReleaseMemObject(self->img_buf);
        if (self->ref_img_buf)
            clReleaseMemObject(self->ref_img_buf);

        self->img_buf = NULL;
        self->ref_img_buf = NULL;
        self->image_w = img->w;
        self->image_h = img->h;
    }

    if (maximal_shift != self->maximal_shift || subpixels != self->subpixels) {
        if (self->correlations_buf_const)
            clReleaseMemObject(self->correlations_buf_const);
        if (self->correlations_const)
            free(self->correlations_const);

        self->correlations_buf_const = NULL;
        self->correlations_const = NULL;
        self->maximal_shift = maximal_shift;
        self->subpixels = subpixels;
    }

    if (self->img_buf == NULL)
        self->img_buf = clCreateBuffer(self->context,  CL_MEM_READ_ONLY,  sizeof(real_t) * self->image_w * self->image_h, NULL, NULL);

    if (self->ref_img_buf == NULL)
        self->ref_img_buf = clCreateBuffer(self->context,  CL_MEM_READ_ONLY,  sizeof(real_t) * self->image_w * self->image_h, NULL, NULL);

    size_t size_corr = (2*maximal_shift*subpixels + 1)*(2*maximal_shift*subpixels + 1);
    if (self->correlations_buf_const == NULL)
        self->correlations_buf_const = clCreateBuffer(self->context, CL_MEM_WRITE_ONLY, sizeof(float) * size_corr, NULL, NULL);
    if (self->correlations_const == NULL)
        self->correlations_const = malloc(size_corr * sizeof(float));

    if (!self->img_buf || !self->ref_img_buf || !self->correlations_buf_const)
    {
        printf("Error: Failed to allocate device memory!\n");
        return -1;
    }

    int err = clEnqueueWriteBuffer(self->commands, self->img_buf, CL_TRUE, 0, sizeof(real_t) * img->w * img->h, img->array, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        return -1;
    }

    err = clEnqueueWriteBuffer(self->commands, self->ref_img_buf, CL_TRUE, 0, sizeof(real_t) * ref_img->w * ref_img->h, ref_img->array, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        return -1;
    }

    err = 0;
    err |= clSetKernelArg(self->kernel_constant_shift, 0, sizeof(unsigned), &img->h);
    err |= clSetKernelArg(self->kernel_constant_shift, 1, sizeof(unsigned), &img->w);
    err |= clSetKernelArg(self->kernel_constant_shift, 2, sizeof(cl_mem), &self->img_buf);
    err |= clSetKernelArg(self->kernel_constant_shift, 3, sizeof(cl_mem), &self->ref_img_buf);
    err |= clSetKernelArg(self->kernel_constant_shift, 4, sizeof(cl_mem), &self->correlations_buf_const);
    err |= clSetKernelArg(self->kernel_constant_shift, 5, sizeof(int), &maximal_shift);
    err |= clSetKernelArg(self->kernel_constant_shift, 6, sizeof(int), &subpixels);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        return -1;
    }

    err = clGetKernelWorkGroupInfo(self->kernel_constant_shift, self->device_id,
                                    CL_KERNEL_WORK_GROUP_SIZE,
                                    sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        return -1;
    }

    size_t dims[1] = {(2*maximal_shift*subpixels + 1) *
                      (2*maximal_shift*subpixels + 1)};
    err = clEnqueueNDRangeKernel(self->commands, self->kernel_constant_shift, 1, NULL, dims, NULL, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel! Error %i\n", err);
        return -1;
    }

    clFinish(self->commands);

    err = clEnqueueReadBuffer(self->commands, self->correlations_buf_const, CL_TRUE, 0, sizeof(float) * size_corr,
                              self->correlations_const, 0, NULL, NULL );  
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        return -1;
    }

    int y, x, maxx, maxy;
    float maxcorr = -2;
    for (y = 0; y < 2*maximal_shift*subpixels + 1; y++)
    for (x = 0; x < 2*maximal_shift*subpixels + 1; x++)
    {
        if (self->correlations_const[y*(2*maximal_shift*subpixels + 1)+x] > maxcorr) {
            maxx = x;
            maxy = y;
            maxcorr = self->correlations_const[y*(2*maximal_shift*subpixels + 1)+x];
        }
    }

    float shift_y = ((float)maxy - maximal_shift*subpixels) / subpixels;
    float shift_x = ((float)maxx - maximal_shift*subpixels) / subpixels;

    int i, j;
    for (i = 0; i < self->grid_h; i++)
        for (j = 0; j < self->grid_w; j++)
        {
            image_deform_set_shift(&self->array, j, i, 0, shift_y * self->array.sy);
            image_deform_set_shift(&self->array, j, i, 1, shift_x * self->array.sx);
        }


    return 0;
}
#endif

static int image_deform_lc_find_constant_cpu(struct ImageDeformLocalCorrelator *self,
                                              const struct ImageGrid *img,
                                              const struct ImageDeform *pre_align,
                                              const struct ImageGrid *ref_img,
                                              const struct ImageDeform *ref_pre_align,
                                              real_t maximal_shift,
                                              int subpixels)
{
    int i, j;
    real_t iter_x, iter_y;
    real_t best_x, best_y;
    real_t best_corr;

    struct ImageGrid global_area;
    image_grid_init(&global_area, img->w, img->h);
    best_x = best_y = 0;
    image_deform_lc_get_area(img, pre_align, &global_area, img->w / 2.0, img->h / 2.0);
    best_corr = image_grid_correlation(&global_area, ref_img);
    for (iter_y = -maximal_shift; iter_y <= maximal_shift; iter_y += 1.0 / subpixels)
        for (iter_x = -maximal_shift; iter_x <= maximal_shift; iter_x += 1.0 / subpixels)
        {
            image_deform_lc_get_area(img, pre_align, &global_area, iter_x + img->w / 2.0, iter_y + img->h / 2.0);
            real_t corr = image_grid_correlation(&global_area, ref_img);
            if (corr > best_corr)
            {
                best_corr = corr;
                best_x = iter_x;
                best_y = iter_y;
            }
        }
    for (i = 0; i < self->grid_h; i++)
        for (j = 0; j < self->grid_w; j++)
        {
            image_deform_set_shift(&self->array, j, i, 0, best_y * self->array.sy);
            image_deform_set_shift(&self->array, j, i, 1, best_x * self->array.sx);
        }
    image_grid_finalize(&global_area);

    UNUSED(ref_pre_align);
    return 0;
}

int image_deform_lc_find_constant(struct ImageDeformLocalCorrelator *self,
                                  const struct ImageGrid *img,
                                  const struct ImageDeform *pre_align,
                                  const struct ImageGrid *ref_img,
                                  const struct ImageDeform *ref_pre_align,
                                  real_t maximal_shift,
                                  int subpixels)
{
#ifdef USE_OPENCL
    if (self->use_opencl)
        return image_deform_lc_find_constant_ocl(self, img, pre_align, ref_img, ref_pre_align, maximal_shift, subpixels);
    else
        return image_deform_lc_find_constant_cpu(self, img, pre_align, ref_img, ref_pre_align, maximal_shift, subpixels);
#else
    return image_deform_lc_find_constant_cpu(self, img, pre_align, ref_img, ref_pre_align, maximal_shift, subpixels);
#endif
}
