
/* Interpolation methods */
double image_grid_interpolation_1d(double fm1, double f0, double f1, double f2, double x);

double image_grid_interpolation_2d(double fm1m1, double f0m1, double f1m1, double f2m1,
                        double fm10,  double f00,  double f10,  double f20,
                        double fm11,  double f01,  double f11,  double f21,
                        double fm12,  double f02,  double f12,  double f22,
                        double x, double y);

void image_wave_shift_interpolate(const struct ImageWave *self,
                                  const struct ImageDeform *array,
                                  double x, double y,
                                  double *rx, double *ry);

double image_wave_interpolation(const struct ImageDeform *array,
                                int xi, int yi, int axis, 
                                double dx, double dy);


/* Image related methods */

void image_wave_shift_image(struct ImageWave *self,
                            const struct ImageDeform *array,
                            const struct ImageDeform *input_image,
                            struct ImageDeform *output_image,
                            int subpixels);


/* Approximation by targets methods */
void image_wave_approximate_by_targets(struct ImageWave *self, double dh, size_t Nsteps,
                                       double *targets, double *points, size_t N);

void image_wave_approximate_with_images(struct ImageWave *self,
                                        const struct ImageDeform *img,
                                        const struct ImageWave *pre_align,
                                        const struct ImageDeform *ref_img,
                                        const struct ImageWave *ref_pre_align,
                                        int radius,
                                        double maximal_shift,
                                        int subpixels);
