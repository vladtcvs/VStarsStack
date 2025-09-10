float get_pixel(unsigned int image_h, unsigned int image_w, const float *image, float y, float x)
{
    int xi = floor(x + 0.5f);
    int yi = floor(y + 0.5f);

    if (xi < 0 || xi >= image_w || yi < 0 || yi >= image_h)
        return NAN;
    
    float dx = x - xi;
    float dy = y - yi;

    return image[yi*image_w + xi];
}

float find_correlation(unsigned int image_h, unsigned int image_w,
                       const float *image, const float *ref_image,

                       unsigned int shift_grid_h, unsigned int shift_grid_w,
                       int shift_grid_row, int shift_grid_col

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

        top += (pixel1 - average1)*(pixel2 - average2);
        bottom1 += (pixel1 - average1)*(pixel1 - average1);
        bottom2 += (pixel2 - average2)*(pixel2 - average2);
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
    return corr;
}

// image            - float [image_h * image_w]
// ref_image        - float [image_h * image_w]
//
// correlation      - float [shift_grid_h * shift_grid_w * (2*maximal_shift*division+1) * (2*maximal_shift*division+1)]
__kernel void image_deform_lc_find( unsigned int image_h, unsigned int image_w,
                                    __global const float *image,
                                    __global const float *ref_image,

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
                (2*maximal_shift*division+1) * (2*maximal_shift*division+1) * shift_grid_col_col +
                (2*maximal_shift*division+1) * (2*maximal_shift*division+1) * shift_grid_w * shift_grid_row;

    // varies -maximal_shift .. maximal_shift with step 1.0/division
    float shift_y = correlation_shift_y / division;
    float shift_x = correlation_shift_x / division;

    correlation[index] = find_correlation(image_h, image_w,
                                          image, ref_image,

                                          shift_grid_h, shift_grid_w,
                                          shift_grid_row, shift_grid_col,
                                          shift_y, shift_x

                                          correlation_r);
}
