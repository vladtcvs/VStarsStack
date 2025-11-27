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

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

/// @brief function for apriori esimation p({f_i}), i=0....N-1
typedef double (*apriori_f)(const double *f, int N, const void *param);

/** @brief parameters for integration
 * Estimation approach:
 *  1. iterate {f_i} in some region of R+^N
 *  2. for each {f_i} find p(f|F), it requieres integration {f'_i} over R+^N, but we cannot integrate over ifinite region,
 *      so we integrate over some area in R+^N
 *  3. select {f_i} with maximal p(f|F)
 */
struct bayes_posterior_ctx_s
{
    int N;                  ///< number of dimensions of vector {f_i}, i=0..N-1
    int *index_integration; ///< vector for indexing current {f'_i} during integration
    int *index_estimation;  ///< vector for indexing current {f_i} during searching maximal p
    int *index_max;         ///< maximal index values for each index
    double *f_integration;  ///< vector for {f'_i} during integration
    double *f_estimation;   ///< vector for keeping {f_i} with maximal p(f|F) during estimation
};

/**
 * @brief Estimate posterior probability p({p_i}|{F_a}). Here 'i' for components indexing, 'a' for frame indexing
 * @param ctx - context
 * @param num_frames - number of measurements
 * @param F - frames {F_a}, a=0..num_frames-1
 * @param f - {f_i} for which we find p({f_i}|{F_a})
 * @param d - constant coefficients {d_i}, for dark, sky
 * @param v - spatial coefficient {v_i}, for flat
 * @param K - contribution coefficients {K_i,a}
 * @param apriori - apriori p({f_i})
 * @param apriori_params - aux params for apriori function
 * @param limits_low - lower limit for integration
 * @param limits_high - higher limits for integration
 * @param df - integration step
 * @return p({p_i} | {F_a})
 */
double bayes_posterior(struct bayes_posterior_ctx_s *ctx,
                       int num_frames,
                       const uint64_t *F,
                       const double *f,
                       const double *d,
                       const double *v,
                       const double *K,
                       const apriori_f apriori,
                       const void *apriori_params,
                       const double *limits_low,
                       const double *limits_high,
                       double dl);

/**
 * @brief Find estimation E[{f_i}] by integral over all {f_i} of p({f_i}|{F_a}) * {f_i}
 * @param ctx - context
 * @param num_frames - number of measurements
 * @param F - frames {F_a}, a=0..num_frames-1
 * @param d - constant coefficients {d_i}, for dark, sky
 * @param v - spatial coefficient {v_i}, for flat
 * @param K - contribution coefficients {K_i,a}
 * @param apriori - apriori p({f_i})
 * @param apriori_params - aux params for apriori function
 * @param limits_low - lower limit for integration
 * @param limits_high - higher limits for integration
 * @param dl - integration step
 * @param clip - only use points where posterior > max posterior * clip
 * @param f - estimated value
 */
void bayes_estimate(struct bayes_posterior_ctx_s *ctx,
                    int num_frames,
                    const uint64_t *F,
                    const double *d,
                    const double *v,
                    const double *K,
                    const apriori_f apriori,
                    const void *apriori_params,
                    const double *limits_low,
                    const double *limits_high,
                    double dl,
                    double clip,
                    double *f);

/**
 * @brief Find such {f_i} where p({f_i} | {F_a}) is maximal
 * @param ctx - context
 * @param num_frames - number of measurements
 * @param F - frames {F_a}, a=0..num_frames-1
 * @param d - constant coefficients {d_i}, for dark, sky
 * @param v - spatial coefficient {v_i}, for flat
 * @param K - contribution coefficients {K_i,a}
 * @param apriori - apriori p({f_i})
 * @param apriori_params - aux params for apriori function
 * @param limits_low - lower limit for integration
 * @param limits_high - higher limits for integration
 * @param dl - integration step
 * @param f - estimated value
 */
void bayes_maxp(struct bayes_posterior_ctx_s *ctx,
                int num_frames,
                const uint64_t *F,
                const double *d,
                const double *v,
                const double *K,
                const apriori_f apriori,
                const void *apriori_params,
                const double *limits_low,
                const double *limits_high,
                double dl,
                double *f);

void bayes_posterior_free(struct bayes_posterior_ctx_s *ctx);
bool bayes_posterior_init(struct bayes_posterior_ctx_s *ctx, int num_dim);

#ifdef __cplusplus
}
#endif
