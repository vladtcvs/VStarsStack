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

static void get_area(const struct ImageWaveGrid *img,
                     double x, double y,
                     struct ImageWaveGrid *area)
{
    int i, j;
    int w = area->w;
    int h = area->h;
    for (i = 0; i < h; i++)
    for (j = 0; j < w; j++)
    {
        double px = x + j - (w-1)/2.0;
        double py = y + i - (h-1)/2.0;
        double val = image_wave_get_array(img, px, py, 0);
        image_wave_set_array(area, j, i, 0, val);
    }
}

void image_wave_approximate_with_images(struct ImageWave *self,
                                        const struct ImageWaveGrid *img,
                                        const struct ImageWaveGrid *ref_img,
                                        int radius,
                                        double maximal_shift,
                                        int subpixels)
{
    int i, j;
    int w = radius*2+1;
    int h = w;
    struct ImageWaveGrid area = {
        .naxis = 1,
        .w = w,
        .h = w,
        .array = calloc(w*h, sizeof(double)),
    };
    struct ImageWaveGrid ref_area = {
        .naxis = 1,
        .w = w,
        .h = w,
        .array = calloc(w*h, sizeof(double)),
    };

    for (i = 0; i < self->Nh; i++)
    for (j = 0; j < self->Nw; j++)
    {
        double x, y;
        double best_x = j, best_y = i;
        get_area(img, best_x, best_y, &area);
        get_area(ref_img, j, i, &ref_area);
        double best_corr = image_wave_correlation(&area, &ref_area);

        for (y = i - maximal_shift; y <= i + maximal_shift; y += 1.0 / subpixels)
        for (x = j - maximal_shift; x <= j + maximal_shift; x += 1.0 / subpixels)
        {
            get_area(img, x, y, &area);
            double corr = image_wave_correlation(&area, &ref_area);
            if (corr > best_corr)
            {
                best_corr = corr;
                best_x = x;
                best_y = y;
            }
        }
        image_wave_set_array(&self->array, j, i, 0, best_x - j);
        image_wave_set_array(&self->array, j, i, 1, best_y - i);
    }

    free(ref_area.array);
    free(area.array);
}
