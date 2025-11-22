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

#include <stdbool.h>
#include <stdint.h>

typedef double (*apriori_f)(const double *f, int num_dim, const void *param);

struct bayes_posterior_ctx_s
{
    int num_dim;
    int *index_integration;
    int *index_estimation;
    int *index_max;
    double *f_integration;
    double *f_estimation;
};

double bayes_posterior(struct bayes_posterior_ctx_s *ctx,
                       int num_frames,
                       const uint64_t *F,
                       const double *f,
                       const double *lambdas_d,
                       const double *lambdas_v,
                       const apriori_f apriori,
                       const void *apriori_params,
                       const double *limits_low,
                       const double *limits_high,
                       double dl);

void bayes_estimate(struct bayes_posterior_ctx_s *ctx,
                    int num_frames,
                    const uint64_t *F,
                    const double *lambdas_d,
                    const double *lambdas_v,
                    const apriori_f apriori,
                    const void *apriori_params,
                    const double *limits_low,
                    const double *limits_high,
                    double dl,
                    double clip,
                    double *f);

void bayes_maxp(struct bayes_posterior_ctx_s *ctx,
                int num_frames,
                const uint64_t *F,
                const double *lambdas_d,
                const double *lambdas_v,
                const apriori_f apriori,
                const void *apriori_params,
                const double *limits_low,
                const double *limits_high,
                double dl,
                double *f);

void bayes_posterior_free(struct bayes_posterior_ctx_s *ctx);
bool bayes_posterior_init(struct bayes_posterior_ctx_s *ctx, int num_dim);
