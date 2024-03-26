
/* Interpolation methods */
double image_wave_interpolation_1d(double fm1, double f0, double f1, double f2, double x);

double image_wave_interpolation_2d(double fm1m1, double f0m1, double f1m1, double f2m1,
                        double fm10,  double f00,  double f10,  double f20,
                        double fm11,  double f01,  double f11,  double f21,
                        double fm12,  double f02,  double f12,  double f22,
                        double x, double y);

void image_wave_shift_interpolate(const struct ImageWave *self,
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
                            struct ImageWaveGrid *output_image,
                            int subpixels);


/* Approximation by targets methods */
void image_wave_approximate_by_targets(struct ImageWave *self, double dh, size_t Nsteps,
                                       double *targets, double *points, size_t N);

void image_wave_approximate_with_images(struct ImageWave *self,
                                        const struct ImageWaveGrid *img,
                                        const struct ImageWave *pre_align,
                                        const struct ImageWaveGrid *ref_img,
                                        const struct ImageWave *ref_pre_align,
                                        int radius,
                                        double maximal_shift,
                                        int subpixels);
