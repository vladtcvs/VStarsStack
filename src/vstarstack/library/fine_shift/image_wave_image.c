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

#include <math.h>
#include "image_wave.h"

static void image_wave_set_pixel(struct ImageWaveGrid *image, int x, int y, double val)
{
    image_wave_set_array(image, x, y, 0, val);
}

static double image_wave_get_pixel(const struct ImageWaveGrid *image, int x, int y)
{
    if (x < 0)
        return NAN;
    if (y < 0)
        return NAN;
    if (x >= image->w - 1)
        return NAN;
    if (y >= image->h - 1)
        return NAN;
    
    double dx = x - (int)x;
    double dy = y - (int)y;
    return image_wave_interpolation(image, (int)x, (int)y, 0, dx, dy);
}


/* Apply shift array to image */
void image_wave_shift_image(struct ImageWave *self,
                            const struct ImageWaveGrid *input_image,
                            struct ImageWaveGrid *output_image)
{
    int y, x;
    for (y = 0; y < output_image->h; y++)
        for (x = 0; x < output_image->w; x++)
        {
            double oy, ox;
            image_wave_shift_interpolate(self, &self->array, x, y, &ox, &oy);
            double val = image_wave_get_pixel(input_image, ox, oy);
            if (isnan(val))
                val = 0;
            image_wave_set_pixel(output_image, x, y, val);
        }
}
