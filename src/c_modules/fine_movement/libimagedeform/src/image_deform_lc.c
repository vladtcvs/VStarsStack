/*
 * Copyright (c) 2022-2024 Vladislav Tsendrovskii
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
                          int image_w, int image_h, int pixels)
{
    self->image_w = image_w;
    self->image_h = image_h;
    self->pixels = pixels;
    self->grid_w = ceil((double)image_w/pixels);
    self->grid_h = ceil((double)image_h/pixels);
    if (image_deform_init(&self->array, self->grid_w, self->grid_h, self->image_w, self->image_h) != 0)
        return -1;
    return 0;
}

void image_deform_lc_finalize(struct ImageDeformLocalCorrelator *self)
{
    image_deform_finalize(&self->array);
}

static void image_deform_lc_get_area(const struct ImageGrid *img,
                                     const struct ImageDeform *pre_align,
                                     struct ImageGrid *area,
                                     double x, double y)
{
    double pre_aligned_x;
    double pre_aligned_y;
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

void image_deform_lc_find_constant(struct ImageDeformLocalCorrelator *self,
                                   const struct ImageGrid *img,
                                   const struct ImageDeform *pre_align,
                                   const struct ImageGrid *ref_img,
                                   const struct ImageDeform *ref_pre_align,
                                   double maximal_shift,
                                   int subpixels)
{
    int i, j;
    double iter_x, iter_y;
    double best_x, best_y;
    double best_corr;

    // TODO: use them
    (void)ref_pre_align;
    (void)pre_align;

    struct ImageGrid global_area;
    image_grid_init(&global_area, img->w, img->h);
    best_x = best_y = 0;
    image_deform_lc_get_area(img, pre_align, &global_area, img->w/2.0, img->h/2.0);
    best_corr = image_grid_correlation(&global_area, ref_img);
    for (iter_y = -maximal_shift; iter_y <= maximal_shift; iter_y += 1.0 / subpixels)
    for (iter_x = -maximal_shift; iter_x <= maximal_shift; iter_x += 1.0 / subpixels)
    {
        image_deform_lc_get_area(img, pre_align, &global_area, iter_x + img->w/2.0, iter_y + img->h/2.0);
        double corr = image_grid_correlation(&global_area, ref_img);
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
        image_deform_set_shift(&self->array, j, i, 0, best_y*self->array.sy);
        image_deform_set_shift(&self->array, j, i, 1, best_x*self->array.sx);
    }
    image_grid_finalize(&global_area);
}

static void image_deform_lc_find_local(const struct ImageGrid *img,
                                       const struct ImageDeform *pre_align,
                                       const struct ImageGrid *ref_img,
                                       const struct ImageDeform *ref_pre_align,
                                       int x,
                                       int y,
                                       struct ImageGrid *area,
                                       struct ImageGrid *ref_area,
                                       double maximal_shift,
                                       int subpixels,
                                       double mean_shift_x,
                                       double mean_shift_y,
                                       double *shift_x,
                                       double *shift_y)
{
    double iter_x, iter_y;
    double best_x = 0;
    double best_y = 0;
    int num_best = 1;
    image_deform_lc_get_area(img, pre_align, area, x, y);
    image_deform_lc_get_area(ref_img, ref_pre_align, ref_area, x, y);
    double best_corr = image_grid_correlation(area, ref_area);
    double best_dist = fabs(mean_shift_x) + fabs(mean_shift_y);

    for (iter_y = -maximal_shift; iter_y <= maximal_shift; iter_y += 1.0 / subpixels)
    for (iter_x = -maximal_shift; iter_x <= maximal_shift; iter_x += 1.0 / subpixels)
    {
        if (iter_x == 0 && iter_y == 0)
        {
            continue;
        }
        image_deform_lc_get_area(img, pre_align, area, x+iter_x, y+iter_y);
        image_deform_lc_get_area(ref_img, pre_align, ref_area, x, y);
        double corr1 = image_grid_correlation(area, ref_area);
        image_deform_lc_get_area(img, pre_align, area, x, y);
        image_deform_lc_get_area(ref_img, pre_align, ref_area, x-iter_x, y-iter_y);
        double corr2 = image_grid_correlation(area, ref_area);
        double corr = (corr1 + corr2)/2;
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
            double dist = fabs(iter_x - mean_shift_x) + fabs(iter_y - mean_shift_y);
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

void image_deform_lc_find(struct ImageDeformLocalCorrelator *self,
                          const struct ImageGrid *img,
                          const struct ImageDeform *pre_align,
                          const struct ImageGrid *ref_img,
                          const struct ImageDeform *ref_pre_align,
                          int radius,
                          double maximal_shift,
                          int subpixels)
{
    int i, j;

    // Find deviations from mean shift
    int w = radius*2+1;
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
        int x = j * self->pixels;
        int y = i * self->pixels;

        double best_x, best_y;
        image_deform_lc_find_local(img, pre_align, ref_img, ref_pre_align,
                                   x, y, &area, &ref_area,
                                   maximal_shift, subpixels,
                                   0, 0,
                                   &best_x, &best_y);

        if (isnan(best_x) || isnan(best_y))
            hasnan = true;
        image_deform_set_shift(&self->array, j, i, 0, best_y*self->array.sy);
        image_deform_set_shift(&self->array, j, i, 1, best_x*self->array.sx);
    }

    if (hasnan)
    {
        for (i = 0; i < self->grid_h; i++)
        for (j = 0; j < self->grid_w; j++)
        {
            double sy = image_deform_get_shift(&self->array, j, i, 0);
            double sx = image_deform_get_shift(&self->array, j, i, 1);
            if (!isnan(sy) && !isnan(sx))
                continue;
            int ii, jj;
            double shx = 0, shy = 0;
            int cnt = 0;
            for (ii = -1; ii <= 1; ii++)
            for (jj = -1; jj <= 1; jj++)
            {
                double vy = image_deform_get_array(&self->array, j+jj, i+ii, 0);
                double vx = image_deform_get_array(&self->array, j+jj, i+ii, 1);
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
            double best_x, best_y;
            image_deform_lc_find_local(img, pre_align, ref_img, ref_pre_align,
                                       x, y, &area, &ref_area,
                                       maximal_shift, subpixels,
                                       shx, shy,
                                       &best_x, &best_y);
            image_deform_set_shift(&self->array, j, i, 0, best_y*self->array.sy);
            image_deform_set_shift(&self->array, j, i, 1, best_x*self->array.sx);
        }
    }

    image_grid_finalize(&area);
    image_grid_finalize(&ref_area);
}
