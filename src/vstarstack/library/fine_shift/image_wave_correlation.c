/*
 * Copyright (c) 2022 Vladislav Tsendrovskii
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


#include "image_wave.h"


void image_wave_approximate_with_images(struct ImageWave *self,
                                        const struct ImageDeform *img,
                                        const struct ImageWave *pre_align,
                                        const struct ImageDeform *ref_img,
                                        const struct ImageWave *ref_pre_align,
                                        int radius,
                                        double maximal_shift,
                                        int subpixels)
{
    int i, j;
    int w = radius*2+1;
    int h = w;
    struct ImageDeform area = {
        .naxis = 1,
        .w = w,
        .h = w,
        .array = calloc(w*h, sizeof(double)),
    };
    struct ImageDeform ref_area = {
        .naxis = 1,
        .w = w,
        .h = w,
        .array = calloc(w*h, sizeof(double)),
    };

    for (i = 0; i < self->Nh; i++)
    for (j = 0; j < self->Nw; j++)
    {
        double best_orig_x, best_orig_y;
        if (pre_align == NULL)
        {
            best_orig_x = j;
            best_orig_y = i;
        }
        else
        {
            image_wave_shift_interpolate(pre_align, &pre_align->array,
                                         j, i, &best_orig_x, &best_orig_y);
        }

        get_area(img, best_orig_x, best_orig_y, &area);

        double orig_i, orig_j;
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

    free(ref_area.array);
    free(area.array);
}
