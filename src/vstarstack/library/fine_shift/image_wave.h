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

#pragma once

#include <math.h>
#include <stddef.h>

struct ImageWaveGrid
{
    int w;
    int h;
    int naxis;
    double *array;
};

struct ImageWave
{
    struct ImageWaveGrid array;
    struct ImageWaveGrid array_p;
    struct ImageWaveGrid array_m;
    struct ImageWaveGrid array_gradient;

    int Nw;
    int Nh;
    int w;       // Grid image width
    int h;       // Grid image height
    double sx;
    double sy;
    double stretch_penalty_k;
};

/* Initialization methods */
int  image_wave_init(struct ImageWave *self, int w, int h,
                     double Nw, double Nh, double spk);
void image_wave_finalize(struct ImageWave *self);

void image_wave_init_shift_array(struct ImageWaveGrid *grid, double dx, double dy);
void image_wave_print_array(const struct ImageWaveGrid *array);


/* Common math methods */
void image_wave_move_along_gradient(struct ImageWave *self,
                                    const struct ImageWaveGrid *gradient,
                                    double dh);

double image_wave_stretch_penalty(const struct ImageWaveGrid *array);

double image_wave_correlation(const struct ImageWaveGrid *image1,
                              const struct ImageWaveGrid *image2);

/*
 * Set shift array at (x,y)
 */
static inline void image_wave_set_array(struct ImageWaveGrid *grid,
                                        int x, int y, int axis,
                                        double val)
{
    grid->array[y*(grid->w * grid->naxis) + x*grid->naxis + axis] = val;
}

/*
 * Get shift array at (x,y)
 */
static inline double image_wave_get_array(const struct ImageWaveGrid *grid,
                                          int x, int y, int axis)
{
    if (x >= grid->w)
        return NAN;
    if (x < 0)
        return NAN;
    if (y >= grid->h)
        return NAN;
    if (y < 0)
        return NAN;
    return grid->array[y*(grid->w * grid->naxis) + x*grid->naxis + axis];
}

/* Interpolation methods */
double image_wave_interpolation_1d(double fm1, double f0, double f1, double f2, double x);

double image_wave_interpolation_2d(double fm1m1, double f0m1, double f1m1, double f2m1,
                        double fm10,  double f00,  double f10,  double f20,
                        double fm11,  double f01,  double f11,  double f21,
                        double fm12,  double f02,  double f12,  double f22,
                        double x, double y);

void image_wave_shift_interpolate(struct ImageWave *self,
                                  const struct ImageWaveGrid *array,
                                  double x, double y,
                                  double *rx, double *ry);

double image_wave_interpolation(const struct ImageWaveGrid *array,
                                int xi, int yi, int axis, 
                                double dx, double dy);


/* Image related methods */

void image_wave_shift_image(struct ImageWave *self,
                            const struct ImageWaveGrid *array,
                            const struct ImageWaveGrid *input_image,
                            struct ImageWaveGrid *output_image);


/* Approximation by targets methods */
void image_wave_approximate_by_targets(struct ImageWave *self, double dh, size_t Nsteps,
                                       double *targets, double *points, size_t N);

void image_wave_approximate_with_images(struct ImageWave *self,
                                        const struct ImageWaveGrid *img,
                                        const struct ImageWaveGrid *ref_img,
                                        int radius,
                                        double maximal_shift,
                                        int subpixels);
