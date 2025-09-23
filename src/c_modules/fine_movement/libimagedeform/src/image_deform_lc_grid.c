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
        image_deform_lc_get_area(img, pre_align, area, x+iter_x, y+iter_y);
        image_deform_lc_get_area(ref_img, ref_pre_align, ref_area, x, y);
        real_t corr1 = image_grid_correlation(area, ref_area);
        image_deform_lc_get_area(img, pre_align, area, x, y);
        image_deform_lc_get_area(ref_img, ref_pre_align, ref_area, x-iter_x, y-iter_y);
        real_t corr2 = image_grid_correlation(area, ref_area);
        real_t corr = (corr1 + corr2)/2;
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

void image_deform_lc_find(struct ImageDeformLocalCorrelator *self,
                          const struct ImageGrid *img,
                          const struct ImageDeform *pre_align,
                          const struct ImageGrid *ref_img,
                          const struct ImageDeform *ref_pre_align,
                          int radius,
                          real_t maximal_shift,
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

        real_t best_x, best_y;
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
                real_t vy = image_deform_get_array(&self->array, j+jj, i+ii, 0);
                real_t vx = image_deform_get_array(&self->array, j+jj, i+ii, 1);
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
            image_deform_set_shift(&self->array, j, i, 0, best_y*self->array.sy);
            image_deform_set_shift(&self->array, j, i, 1, best_x*self->array.sx);
        }
    }

    image_grid_finalize(&area);
    image_grid_finalize(&ref_area);
}
