/*
 * Copyright (c) 2023 Vladislav Tsendrovskii
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

#include <stdio.h>
#include <math.h>
#include "image_wave.h"

#define SQR(x) ((x)*(x))


/* Apply shift array to image */
void image_wave_shift_image(struct ImageWave *self,
                            const struct ImageWaveGrid *array,
                            const struct ImageWaveGrid *input_image,
                            struct ImageWaveGrid *output_image,
                            int subpixels)
{
    int y, x;
    for (y = 0; y < output_image->h; y++)
        for (x = 0; x < output_image->w; x++)
        {
            double orig_y, orig_x;
            image_wave_shift_interpolate(self, array,
                                         (double)x/subpixels, (double)y/subpixels,
                                         &orig_x, &orig_y);

            double val = image_wave_get_pixel(input_image, orig_x, orig_y);
            image_wave_set_pixel(output_image, x, y, val);
        }
}

