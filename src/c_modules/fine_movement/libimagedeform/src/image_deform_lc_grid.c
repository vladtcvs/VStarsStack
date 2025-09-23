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
static int image_deform_lc_find_ocl(struct ImageDeformLocalCorrelator *self,
                                    const struct ImageGrid *img,
                                    const struct ImageDeform *pre_align,
                                    const struct ImageGrid *ref_img,
                                    const struct ImageDeform *ref_pre_align,
                                    unsigned int radius,
                                    unsigned int maximal_shift,
                                    unsigned int subpixels)
{
    size_t local;
    int center = maximal_shift * subpixels;

    if (img->h != self->image_h || img->w != self->image_w)
    {
        if (self->img_buf)
            clReleaseMemObject(self->img_buf);
        if (self->ref_img_buf)
            clReleaseMemObject(self->ref_img_buf);

        self->img_buf = NULL;
        self->ref_img_buf = NULL;
        self->image_w = img->w;
        self->image_h = img->h;
    }

    if (maximal_shift != self->maximal_shift || subpixels != self->subpixels)
    {
        if (self->correlations_buf_const)
            clReleaseMemObject(self->correlations_buf_grid);
        if (self->correlations_grid)
            free(self->correlations_grid);

        self->correlations_buf_grid = NULL;
        self->correlations_grid = NULL;
        self->maximal_shift = maximal_shift;
        self->subpixels = subpixels;
    }

    if (self->img_buf == NULL)
        self->img_buf = clCreateBuffer(self->context, CL_MEM_READ_ONLY, sizeof(real_t) * self->image_w * self->image_h, NULL, NULL);

    if (self->ref_img_buf == NULL)
        self->ref_img_buf = clCreateBuffer(self->context, CL_MEM_READ_ONLY, sizeof(real_t) * self->image_w * self->image_h, NULL, NULL);

    size_t size_corr =  self->grid_h *
                        self->grid_w *
                        (2 * center + 1) *
                        (2 * center + 1);

    if (self->correlations_buf_grid == NULL)
        self->correlations_buf_grid = clCreateBuffer(self->context, CL_MEM_WRITE_ONLY, sizeof(float) * size_corr, NULL, NULL);
    if (self->correlations_grid == NULL)
        self->correlations_grid = malloc(size_corr * sizeof(float));

    if (!self->img_buf || !self->ref_img_buf || !self->correlations_buf_grid || !self->correlations_grid)
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
    int argindex = 0;
    int has_pre_align = (pre_align != NULL);
    int has_ref_pre_align = (ref_pre_align != NULL);

    err |= clSetKernelArg(self->kernel_grid_shift, argindex++, sizeof(unsigned), &img->h);
    err |= clSetKernelArg(self->kernel_grid_shift, argindex++, sizeof(unsigned), &img->w);
    err |= clSetKernelArg(self->kernel_grid_shift, argindex++, sizeof(cl_mem), &self->img_buf);
    err |= clSetKernelArg(self->kernel_grid_shift, argindex++, sizeof(cl_mem), &self->ref_img_buf);

    err |= clSetKernelArg(self->kernel_grid_shift, argindex++, sizeof(unsigned), &self->grid_h);
    err |= clSetKernelArg(self->kernel_grid_shift, argindex++, sizeof(unsigned), &self->grid_w);

    err |= clSetKernelArg(self->kernel_grid_shift, argindex++, sizeof(cl_mem), &self->pre_align_buf);
    err |= clSetKernelArg(self->kernel_grid_shift, argindex++, sizeof(int), &has_pre_align);

    err |= clSetKernelArg(self->kernel_grid_shift, argindex++, sizeof(cl_mem), &self->ref_pre_align_buf);
    err |= clSetKernelArg(self->kernel_grid_shift, argindex++, sizeof(int), &has_ref_pre_align);

    err |= clSetKernelArg(self->kernel_grid_shift, argindex++, sizeof(cl_mem), &self->correlations_buf_grid);

    err |= clSetKernelArg(self->kernel_grid_shift, argindex++, sizeof(int), &maximal_shift);
    err |= clSetKernelArg(self->kernel_grid_shift, argindex++, sizeof(int), &subpixels);
    err |= clSetKernelArg(self->kernel_grid_shift, argindex++, sizeof(unsigned), &radius);
    
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        return -1;
    }



    err = clGetKernelWorkGroupInfo(self->kernel_grid_shift, self->device_id,
                                   CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        return -1;
    }

    size_t dims[1] = {self->grid_h *
                      self->grid_w *
                      (2 * center + 1) *
                      (2 * center + 1)
                    };

    err = clEnqueueNDRangeKernel(self->commands, self->kernel_grid_shift, 1, NULL, dims, NULL, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel! Error %i\n", err);
        return -1;
    }

    clFinish(self->commands);

    err = clEnqueueReadBuffer(self->commands, self->correlations_buf_grid, CL_TRUE, 0, sizeof(float) * size_corr,
                              self->correlations_grid, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        return -1;
    }

    unsigned int i, j, y, x;
    float maxcorr = -2;
    int hasnan = 0;

    for (j = 0; j < self->grid_h; j++) {
        for (i = 0; i < self->grid_w; i++)
        {
            
            int index0 = center +
                         (2*center+1) * center +
                         (2*center+1) * (2*center+1) * j +
                         (2*center+1) * (2*center+1) * self->grid_w * i;
            maxcorr = self->correlations_grid[index0];
            int numbest = 1;
            float maxxs = center;
            float maxys = center;

            for (y = 0; y < 2 * center + 1; y++)
                for (x = 0; x < 2 * center + 1; x++)
                {
                    int index = x +
                        (2*center+1) * y +
                        (2*center+1) * (2*center+1) * j +
                        (2*center+1) * (2*center+1) * self->grid_w * i;
                    float corr = self->correlations_grid[index];
                    if (corr > maxcorr + 1e-6f) {
                        maxxs = x;
                        maxys = y;
                        maxcorr = self->correlations_grid[index];
                        numbest = 1;
                    } else if (fabsf(corr - maxcorr) < 1e-6f) {
                        maxxs += x;
                        maxys += y;
                        numbest++;
                    }
                }

            float shift_y = (maxys/numbest - maximal_shift * subpixels) / subpixels;
            float shift_x = (maxxs/numbest - maximal_shift * subpixels) / subpixels;

            image_deform_set_shift(&self->array, j, i, 0, shift_y * self->array.sy);
            image_deform_set_shift(&self->array, j, i, 1, shift_x * self->array.sx);
        }
    }

    return 0;
}
#endif

static void image_deform_lc_find_local(const struct ImageGrid *img,
                                       const struct ImageDeform *pre_align,
                                       const struct ImageGrid *ref_img,
                                       const struct ImageDeform *ref_pre_align,
                                       int x,
                                       int y,
                                       struct ImageGrid *area,
                                       struct ImageGrid *ref_area,
                                       real_t maximal_shift,
                                       int subpixels,
                                       real_t mean_shift_x,
                                       real_t mean_shift_y,
                                       real_t *shift_x,
                                       real_t *shift_y)
{
    real_t iter_x, iter_y;
    real_t best_x = 0;
    real_t best_y = 0;
    int num_best = 1;
    image_deform_lc_get_area(img, pre_align, area, x, y);
    image_deform_lc_get_area(ref_img, ref_pre_align, ref_area, x, y);
    real_t best_corr = image_grid_correlation(area, ref_area);
    real_t best_dist = fabs(mean_shift_x) + fabs(mean_shift_y);

    for (iter_y = -maximal_shift; iter_y <= maximal_shift; iter_y += 1.0 / subpixels)
        for (iter_x = -maximal_shift; iter_x <= maximal_shift; iter_x += 1.0 / subpixels)
        {
            if (iter_x == 0 && iter_y == 0)
            {
                continue;
            }
            image_deform_lc_get_area(img, pre_align, area, x + iter_x, y + iter_y);
            image_deform_lc_get_area(ref_img, ref_pre_align, ref_area, x, y);
            real_t corr1 = image_grid_correlation(area, ref_area);
            image_deform_lc_get_area(img, pre_align, area, x, y);
            image_deform_lc_get_area(ref_img, ref_pre_align, ref_area, x - iter_x, y - iter_y);
            real_t corr2 = image_grid_correlation(area, ref_area);
            real_t corr = (corr1 + corr2) / 2;
            if (corr > best_corr)
            {
                best_corr = corr;
                best_x = iter_x;
                best_y = iter_y;
                num_best = 1;
                best_dist = fabs(iter_x - mean_shift_x) + fabs(iter_y - mean_shift_y);
            }
            else if (corr == best_corr)
            {
                real_t dist = fabs(iter_x - mean_shift_x) + fabs(iter_y - mean_shift_y);
                if (dist < best_dist)
                {
                    best_x = iter_x;
                    best_y = iter_y;
                    num_best = 1;
                    best_dist = dist;
                }
                else if (dist == best_dist)
                {
                    num_best++;
                }
            }
        }
    if (num_best == 1)
    {
        *shift_x = best_x;
        *shift_y = best_y;
    }
    else
    {
        *shift_x = NAN;
        *shift_y = NAN;
    }
}

static int image_deform_lc_find_cpu(struct ImageDeformLocalCorrelator *self,
                                    const struct ImageGrid *img,
                                    const struct ImageDeform *pre_align,
                                    const struct ImageGrid *ref_img,
                                    const struct ImageDeform *ref_pre_align,
                                    int radius,
                                    real_t maximal_shift,
                                    int subpixels)
{
    unsigned int i, j;

    // Find deviations from mean shift
    int w = radius * 2 + 1;
    int h = w;
    bool hasnan = false;
    struct ImageGrid area;
    struct ImageGrid ref_area;
    image_grid_init(&area, w, h);
    image_grid_init(&ref_area, w, h);

    for (i = 0; i < self->grid_h; i++)
        for (j = 0; j < self->grid_w; j++)
        {
            // Find how we should modify img to fit to ref_img
            unsigned int x = j * self->pixels;
            unsigned int y = i * self->pixels;

            real_t best_x, best_y;
            image_deform_lc_find_local(img, pre_align, ref_img, ref_pre_align,
                                       x, y, &area, &ref_area,
                                       maximal_shift, subpixels,
                                       0, 0,
                                       &best_x, &best_y);

            if (isnan(best_x) || isnan(best_y))
                hasnan = true;
            image_deform_set_shift(&self->array, j, i, 0, best_y * self->array.sy);
            image_deform_set_shift(&self->array, j, i, 1, best_x * self->array.sx);
        }

    if (hasnan)
    {
        for (i = 0; i < self->grid_h; i++)
            for (j = 0; j < self->grid_w; j++)
            {
                real_t sy = image_deform_get_shift(&self->array, j, i, 0);
                real_t sx = image_deform_get_shift(&self->array, j, i, 1);
                if (!isnan(sy) && !isnan(sx))
                    continue;
                int ii, jj;
                real_t shx = 0, shy = 0;
                int cnt = 0;
                for (ii = -1; ii <= 1; ii++)
                    for (jj = -1; jj <= 1; jj++)
                    {
                        real_t vy = image_deform_get_array(&self->array, j + jj, i + ii, 0);
                        real_t vx = image_deform_get_array(&self->array, j + jj, i + ii, 1);
                        if (isnan(vx) || isnan(vy))
                            continue;
                        shx += vx / self->array.sx;
                        shy += vy / self->array.sy;
                        cnt++;
                    }

                if (cnt == 0)
                    continue;
                shx /= cnt;
                shy /= cnt;

                int x = j * self->pixels;
                int y = i * self->pixels;
                real_t best_x, best_y;
                image_deform_lc_find_local(img, pre_align, ref_img, ref_pre_align,
                                           x, y, &area, &ref_area,
                                           maximal_shift, subpixels,
                                           shx, shy,
                                           &best_x, &best_y);
                image_deform_set_shift(&self->array, j, i, 0, best_y * self->array.sy);
                image_deform_set_shift(&self->array, j, i, 1, best_x * self->array.sx);
            }
    }

    image_grid_finalize(&area);
    image_grid_finalize(&ref_area);
    return 0;
}

int image_deform_lc_find(struct ImageDeformLocalCorrelator *self,
                         const struct ImageGrid *img,
                         const struct ImageDeform *pre_align,
                         const struct ImageGrid *ref_img,
                         const struct ImageDeform *ref_pre_align,
                         int radius,
                         real_t maximal_shift,
                         int subpixels)
{
#ifdef USE_OPENCL
    if (self->use_opencl)
        return image_deform_lc_find_ocl(self, img, pre_align, ref_img, ref_pre_align, radius, maximal_shift, subpixels);
    else
        return image_deform_lc_find_cpu(self, img, pre_align, ref_img, ref_pre_align, radius, maximal_shift, subpixels);
#else
    return image_deform_lc_find_cpu(self, img, pre_align, ref_img, ref_pre_align, radius, maximal_shift, subpixels);
#endif
}
