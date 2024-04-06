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

#include <image_deform_lc.h>

int  image_deform_lc_init(struct ImageDeformLocalCorrelator *self,
                          int image_w, int image_h, int pixels)
{
    self->image_w = image_w;
    self->image_h = image_h;
    self->pixels = pixels;
    self->grid_w = ceil((double)image_w/pixels);
    self->grid_h = ceil((double)image_h/pixels);
    printf("image: %i:%i grid: %i:%i\n", self->image_w, self->image_h, self->grid_w, self->grid_h);
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
    int w = radius*2+1;
    int h = w;
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

        // Init with no shift
        double best_x = x, best_y = y;
        image_deform_lc_get_area(img, pre_align, &area, x, y);
        image_deform_lc_get_area(ref_img, ref_pre_align, &ref_area, x, y);
        double best_corr = image_grid_correlation(&area, &ref_area);

        // Find shift where best correlation between area in img and ref_img
        double iter_x, iter_y;
        for (iter_y = y - maximal_shift; iter_y <= y + maximal_shift; iter_y += 1.0 / subpixels)
        for (iter_x = x - maximal_shift; iter_x <= x + maximal_shift; iter_x += 1.0 / subpixels)
        {
            if (iter_x == x && iter_y == y)
            {
                // We have already calculated it
                continue;
            }
            image_deform_lc_get_area(img, pre_align, &area, iter_x, iter_y);
            double corr = image_grid_correlation(&area, &ref_area);
            if (corr > best_corr)
            {
                best_corr = corr;
                best_x = iter_x;
                best_y = iter_y;
            }
        }
        image_deform_set_shift(&self->array, j, i, 0, best_y - y);
        image_deform_set_shift(&self->array, j, i, 1, best_x - x);
    }

    image_grid_finalize(&area);
    image_grid_finalize(&ref_area);
}
