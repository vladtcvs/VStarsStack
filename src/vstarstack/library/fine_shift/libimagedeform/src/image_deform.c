/*
 * Copyright (c) 2023-2024 Vladislav Tsendrovskii
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

#include <image_deform.h>
#include <interpolation.h>

#define SQR(x) ((x)*(x))

int image_deform_init(struct ImageDeform *deform,
                      int grid_w, int grid_h,
                      int image_w, int image_h)
{
    if (deform->array != NULL)
        free(deform->array);

    deform->grid_w = grid_w;
    deform->grid_h = grid_h;
    deform->image_w = image_w;
    deform->image_h = image_h;
    deform->array = calloc(deform->grid_w * deform->grid_h * 2, sizeof(double));
    if (deform->array == NULL)
        return -1;
    return 0;
}

void image_deform_finalize(struct ImageDeform *deform)
{
    if (deform->array != NULL)
    {
        free(deform->array);
        deform->array = NULL;
    }
}

void image_deform_print(const struct ImageDeform *deform)
{
    int i, j, k;
    for (i = 0; i < deform->grid_h; i++)
    {
        for (j = 0; j < deform->grid_w; j++)
        {
            printf("[");
            for (k = 0; k < 2; k++)
                printf("%lf ", image_deform_get_array(deform, j, i, k));
            printf("]");
        }
        printf("\n");
    }
    printf("\n");
}

void image_deform_constant_shift(struct ImageDeform *deform, double dx, double dy)
{
    int xi, yi;
    for (yi = 0; yi < deform->grid_h; yi++)
    {
        for (xi = 0; xi < deform->grid_w; xi++)
        {
            image_wave_set_array(deform, xi, yi, 0, dx);
            image_wave_set_array(deform, xi, yi, 1, dy);
        }
    }
}

void image_deform_get_shift(struct ImageDeform *deform,
                            double x, double y,
                            double *shift_x, double *shift_y)
{
    double sx = deform->sx;
    double sy = deform->sy;

    int xi = floor(x/deform->sx);
    int yi = floor(y/deform->sy);

    double dx = x/deform->sx - xi;
    double dy = y/deform->sy - yi;
    if (fabs(dx) < 1e-3 && fabs(dy) < 1e-3)
    {
        *shift_y = image_deform_get_array(deform, xi, yi, 0);
        *shift_x = image_deform_get_array(deform, xi, yi, 1);
    }
    else
    {
        double shifts[2];
        for (int i = 0; i < 2; i++)
        {
            double fm1m1 = image_deform_get_array(deform, xi-1, yi-1, i);
            double fm10  = image_deform_get_array(deform, xi-1, yi-1, i);
            double fm11  = image_deform_get_array(deform, xi-1, yi-1, i);
            double fm12  = image_deform_get_array(deform, xi-1, yi-1, i);
        }
    }
}

void image_deform_apply_point(struct ImageDeform *deform,
                              double x, double y,
                              double *srcx, double *srcy)
{

    double shift_x, shift_y;
    image_deform_get_shift(deform, x, y, &shift_x, &shift_y);
    *srcx = x + shift_x;
    *srcy = y + shift_y;
}

void image_deform_apply_image(struct ImageDeform *deform,
                              const struct ImageGrid *input_image,
                              struct ImageGrid *output_image,
                              int subpixels)
{
    int y, x;
    for (y = 0; y < output_image->h; y++)
        for (x = 0; x < output_image->w; x++)
        {
            double orig_y, orig_x;
            image_wave_shift_interpolate(deform,
                                         (double)x/subpixels, (double)y/subpixels,
                                         &orig_x, &orig_y);

            double val = image_wave_get_pixel(input_image, orig_x, orig_y);
            image_wave_set_pixel(output_image, x, y, val);
        }
}
