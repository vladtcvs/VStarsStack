/*
 * Copyright (c) 2025 Vladislav Tsendrovskii
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

float _get_pixel(unsigned int image_h, unsigned int image_w, __global const float *image, int yi, int xi)
{
    if (xi < 0 || xi >= image_w || yi < 0 || yi >= image_h)
        return NAN;
    return image[yi*image_w + xi];
}

float linear_interp(float a0, float a1, float x)
{
    if (isnan(a0))
        return a1;
    if (isnan(a1))
        return a0;
    return a0 * (1-x) + a1 * x;
}

float bilinear_interp(float a00, float a01, float a10, float a11, float x, float y)
{
    float a0 = linear_interp(a00, a01, x);
    float a1 = linear_interp(a10, a11, x);
    return linear_interp(a0, a1, y);
}

float get_pixel(unsigned int image_h, unsigned int image_w, __global const float *image, float y, float x)
{
    int xi = floor(x);
    int yi = floor(y);

    float dx = x - xi;
    float dy = y - yi;

    float v00 = _get_pixel(image_h, image_w, image, yi, xi);
    float v01 = _get_pixel(image_h, image_w, image, yi, xi+1);
    float v10 = _get_pixel(image_h, image_w, image, yi+1, xi);
    float v11 = _get_pixel(image_h, image_w, image, yi+1, xi+1);
    return bilinear_interp(v00, v01, v10, v11, dx, dy);
}

float find_correlation_for_area(unsigned int image_h, unsigned int image_w,
                                __global const float *image, __global const float *ref_image,

                                unsigned int shift_grid_h, unsigned int shift_grid_w,
                                int shift_grid_row, int shift_grid_col,

                                float shift_y, float shift_x,
                                int correlation_r)
{
    float ref_image_x = (float)(image_w-1) * shift_grid_col / (shift_grid_w-1);
    float ref_image_y = (float)(image_h-1) * shift_grid_row / (shift_grid_h-1);

    float image_x = ref_image_x + shift_x;
    float image_y = ref_image_y + shift_y;

    int xi, yi;

    float average = 0, ref_average = 0;
    int nump = 0;

    for (yi = -correlation_r; yi <= correlation_r; yi += 1)
    for (xi = -correlation_r; xi <= correlation_r; xi += 1)
    {
        float y = image_y + yi;
        float x = image_x + xi;

        float ref_y = ref_image_y + yi;
        float ref_x = ref_image_x + xi;

        float pixel = get_pixel(image_h, image_w, image, y, x);
        float ref_pixel = get_pixel(image_h, image_w, ref_image, ref_y, ref_x);

        if (isnan(pixel) || isnan(ref_pixel))
            continue;
        average += pixel;
        ref_average += ref_pixel;
    }
    if (nump == 0)
        return NAN;

    average /= nump;
    ref_average /= nump;

    float top = 0, bottom1 = 0, bottom2 = 0;

    for (yi = -correlation_r; yi <= correlation_r; yi++)
    for (xi = -correlation_r; xi <= correlation_r; xi++)
    {
        float y = image_y + yi;
        float x = image_x + xi;

        float ref_y = ref_image_y + yi;
        float ref_x = ref_image_x + xi;

        float pixel = get_pixel(image_h, image_w, image, y, x);
        float ref_pixel = get_pixel(image_h, image_w, ref_image, ref_y, ref_x);

        if (isnan(pixel) || isnan(ref_pixel))
            continue;

        top += (pixel - average)*(ref_pixel - ref_average);
        bottom1 += (pixel - average)*(pixel - average);
        bottom2 += (ref_pixel - ref_average)*(ref_pixel - ref_average);
    }

    if (bottom1 == 0 || bottom2 == 0)
    {
        if (bottom1 == 0 && bottom2 == 0)
            return 1;
        return 0;
    }
    float corr = top / sqrt(bottom1 * bottom2);
    if (corr > 1-1e-6f)
        return 1;
    if (corr < -1+1e-6f)
        return -1;
    return corr;
} 

// image            - float [image_h * image_w]
// ref_image        - float [image_h * image_w]
//
// correlation      - float [shift_grid_h * shift_grid_w * (2*maximal_shift*division+1) * (2*maximal_shift*division+1)]
__kernel void image_deform_lc_grid( unsigned int image_h, unsigned int image_w,
                                    __global const float *image,
                                    __global const float *ref_image,

                                    float constant_shift_y, float constant_shift_x,

                                    unsigned int shift_grid_h, unsigned int shift_grid_w,
                                    __global float *correlation,

                                    int maximal_shift,
                                    int division,
                                    unsigned int correlation_r)
{
    int shift_grid_row = get_global_id(0);
    int shift_grid_col = get_global_id(1);

    // varies  -maximal_shift*division .. maximal_shift*division
    int correlation_shift_y = get_global_id(2) - maximal_shift*division; 
    int correlation_shift_x = get_global_id(3) - maximal_shift*division;

    int index = correlation_shift_x +
                (2*maximal_shift*division+1) * correlation_shift_y +
                (2*maximal_shift*division+1) * (2*maximal_shift*division+1) * shift_grid_col +
                (2*maximal_shift*division+1) * (2*maximal_shift*division+1) * shift_grid_w * shift_grid_row;

    // varies -maximal_shift .. maximal_shift with step 1.0/division
    float shift_y = correlation_shift_y / division;
    float shift_x = correlation_shift_x / division;

    correlation[index] = find_correlation_for_area(image_h, image_w,
                                                   image, ref_image,

                                                   shift_grid_h, shift_grid_w,
                                                   shift_grid_row, shift_grid_col,

                                                   shift_y + constant_shift_y, shift_x + constant_shift_x,

                                                   correlation_r);
}

float find_correlation_for_image(unsigned int image_h, unsigned int image_w,
                                 __global const float *image, __global const float *ref_image,

                                 float shift_y, float shift_x)
{
    int xi, yi;

    float average = 0, ref_average = 0;
    int nump = 0;

    for (yi = 0; yi <= image_h; yi += 1)
    for (xi = 0; xi <= image_w; xi += 1)
    {
        float y = yi + shift_y;
        float x = xi + shift_x;

        float ref_y = yi;
        float ref_x = xi;

        float pixel = get_pixel(image_h, image_w, image, y, x);
        float ref_pixel = get_pixel(image_h, image_w, ref_image, ref_y, ref_x);

        if (isnan(pixel) || isnan(ref_pixel))
            continue;
        average += pixel;
        ref_average += ref_pixel;
        nump++;
    }
    if (nump == 0)
        return NAN;

    average /= nump;
    ref_average /= nump;

    float top = 0, bottom1 = 0, bottom2 = 0;

    for (yi = 0; yi <= image_h; yi++)
    for (xi = 0; xi <= image_w; xi++)
    {
        float y = yi + shift_y;
        float x = xi + shift_x;

        float ref_y = yi;
        float ref_x = xi;

        float pixel = get_pixel(image_h, image_w, image, y, x);
        float ref_pixel = get_pixel(image_h, image_w, ref_image, ref_y, ref_x);

        if (isnan(pixel) || isnan(ref_pixel))
            continue;

        top += (pixel - average)*(ref_pixel - ref_average);
        bottom1 += (pixel - average)*(pixel - average);
        bottom2 += (ref_pixel - ref_average)*(ref_pixel - ref_average);
    }

    if (bottom1 == 0 || bottom2 == 0)
    {
        if (bottom1 == 0 && bottom2 == 0)
            return 1;
        return 0;
    }
    float corr = top / sqrt(bottom1 * bottom2);
    if (corr > 1-1e-6f)
        return 1;
    if (corr < -1+1e-6f)
        return -1;
    return corr;
}

// image            - float [image_h * image_w]
// ref_image        - float [image_h * image_w]
//
// correlation      - float [(2*maximal_shift*division+1) * (2*maximal_shift*division+1)]
__kernel void image_deform_lc_constant(unsigned int image_h, unsigned int image_w,
                                       __global const float *image,
                                       __global const float *ref_image,

                                       __global float *correlation,
                                       int maximal_shift,
                                       int division)
{
    int shift_y_id = get_global_id(0);
    int shift_x_id = get_global_id(1);

    int index = (2*maximal_shift*division + 1)*shift_y_id + shift_x_id;

    float shift_y = (float)(shift_y_id - maximal_shift*division) / division;
    float shift_x = (float)(shift_x_id - maximal_shift*division) / division;

    float corr = find_correlation_for_image(image_h, image_w, image, ref_image, shift_y, shift_x);
    correlation[index] = corr;
}
