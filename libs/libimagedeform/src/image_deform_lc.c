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
    if (image_deform_init(&self->array, self->grid_w, self->grid_h, self->image_w, self->image_h) != 0)
        return -1;
    return 0;
}

void image_deform_lc_finalize(struct ImageDeformLocalCorrelator *self)
{
    image_deform_finalize(&self->array);
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
    struct ImageDeform ref_area;

    image_grid_init(&area, w, h);
    image_grid_init(&ref_area, w, h);

    for (i = 0; i < self->grid_h; i++)
    for (j = 0; j < self->grid_w; j++)
    {
        int x = j * self->pixels;
        int y = i * self->pixels;

        double best_orig_x, best_orig_y;
        if (pre_align == NULL)
        {
            best_orig_x = x;
            best_orig_y = y;
        }
        else
        {
            image_deform_apply_point(pre_align, x, y, &best_orig_x, &best_orig_y);
        }

        image_grid_get_area(img, best_orig_x, best_orig_y, &area);

        double orig_y, orig_x;
        if (ref_pre_align == NULL)
        {
            orig_i = i;
            orig_j = j;
        }
        else
        {
            image_wave_shift_interpolate(ref_pre_align, &ref_pre_align->array,
                                         j, i, &orig_j, &orig_i);
        }
        get_area(ref_img, orig_j, orig_i, &ref_area);
        double best_corr = image_wave_correlation(&area, &ref_area);

        double x, y;
        for (y = i - maximal_shift; y <= i + maximal_shift; y += 1.0 / subpixels)
        for (x = j - maximal_shift; x <= j + maximal_shift; x += 1.0 / subpixels)
        {
            double orig_x, orig_y;
            if (pre_align == NULL)
            {
                orig_x = x;
                orig_y = y;
            }
            else
            {
                image_wave_shift_interpolate(pre_align, &pre_align->array,
                                             x, y, &orig_x, &orig_y);
            }
            get_area(img, orig_x, orig_y, &area);
            double corr = image_wave_correlation(&area, &ref_area);
            if (corr > best_corr)
            {
                best_corr = corr;
                best_orig_x = orig_x;
                best_orig_y = orig_y;
            }
        }
        image_wave_set_array(&self->array, j, i, 0, best_orig_x - j);
        image_wave_set_array(&self->array, j, i, 1, best_orig_y - i);
    }

    image_deform_finalize(&area);
    image_deform_finalize(&ref_area);
}
