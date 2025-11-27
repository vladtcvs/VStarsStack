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

#include "bayes.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

static void init_index(int num_dim, int *index)
{
    memset(index, 0, sizeof(int) * num_dim);
}

/**
 * @brief go to next integration or iteration position
 */
static bool next_index(int num_dim, const int *index_max, int *index)
{
    int i;
    index[0]++;
    for (i = 0; i < num_dim - 1; i++)
    {
        if (index[i] >= index_max[i])
        {
            index[i] = 0;
            index[i + 1]++;
        }
        else
        {
            return true;
        }
    }
    if (index[num_dim - 1] >= index_max[num_dim - 1])
        return false;
    return true;
}

/**
 * @brief convert index in interation space to {f_i}
 */
static void index_2_f(int num_dim, const int *index, const double *low, double dl, double *f)
{
    int i;
    for (i = 0; i < num_dim; i++)
        f[i] = low[i] + dl * index[i];
}

/**
 * @brief Calculate Sum of all Lambda({f_i}) over all frames (see formulas 21, 22)
 */
static double SumLambda(int N,
                     int num_frames,
                     const uint64_t *F,
                     const double *f,
                     const double *d,
                     const double *v,
                     const double *K)
{
    double sum = 0;
    int i, k;
    for (i = 0; i < num_frames; i++)
    {
        double item = d[i];
        for (k = 0; k < N; k++)
            item += v[i] * K[i * N + k] * f[k];
        sum += F[i] * log(item) - item;
    }
    return sum;
}

/**
 * @brief Calculate posterior probability item which to be integrated (see formula 22)
 */
static double posterior_item(int N,
                             int num_frames,
                             const uint64_t *F,
                             const double *f,
                             const double *f_integration,
                             const double *d,
                             const double *v,
                             const double *K)
{
    double Lambdas_f_posterior = SumLambda(N, num_frames, F, f, d, v, K);
    double Lambdas_f_integration = SumLambda(N, num_frames, F, f_integration, d, v, K);
    return exp(Lambdas_f_integration - Lambdas_f_posterior);
}

/**
 * @brief Calculate posterior probability p({f_i}|{F_a})  (see formula 22)
 */
static double _bayes_posterior(struct bayes_posterior_ctx_s *ctx,
                               int num_frames,
                               const uint64_t *F,
                               const double *f,
                               const double *d,
                               const double *v,
                               const double *K,
                               const apriori_f apriori,
                               const void *apriori_params,
                               const double *limits_low,
                               const int *index_max,
                               double dl)
{
    double apriori_f = apriori(f, ctx->N, apriori_params);
    if (apriori_f > -1e-12 && apriori_f < 1e-12)
        return 0;
    double s = 0;
    double dln = pow(dl, ctx->N);
    init_index(ctx->N, ctx->index_integration);
    do
    {
        index_2_f(ctx->N, ctx->index_integration, limits_low, dl, ctx->f_integration);
        double apriori_f_integration = apriori(ctx->f_integration, ctx->N, apriori_params);
        if (apriori_f_integration > -1e-12 && apriori_f_integration < 1e-12)
            continue;
        double item = posterior_item(ctx->N, num_frames, F, f, ctx->f_integration, d, v, K) * apriori_f_integration;
        s += item * dln;
    } while (next_index(ctx->N, index_max, ctx->index_integration));
    if (s < 1e-14) {
        // We are inegrating over wrong area where all probabilies are too low
        return 0;
    }
    double posterior = apriori_f / s;
    if (posterior > 1)
        return 1;
    if (posterior < 0)
        return 0;
    return posterior;
}

/**
 * @brief find maximal value for each component of iteration index
 */
static void bayes_index_max(int N,
                            const double *limits_low,
                            const double *limits_high,
                            double dl,
                            int *index_max)
{
    int i;
    for (i = 0; i < N; i++) {
        index_max[i] = ceil((limits_high[i] - limits_low[i]) / dl);
    }
}

/**
 * @brief Calculate posterior probability p({f_i}|{F_a})  (see formula 22)
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
                       double dl)
{
    bayes_index_max(ctx->N, limits_low, limits_high, dl, ctx->index_max);
    return _bayes_posterior(ctx, num_frames, F, f, d, v, K, apriori, apriori_params, limits_low, ctx->index_max, dl);
}

/**
 * @brief Find such {f_i} where p({f_i} | {F_a}) is maximal
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
                double *f)
{
    double maxp = 0;
    bayes_index_max(ctx->N, limits_low, limits_high, dl, ctx->index_max);
    init_index(ctx->N, ctx->index_estimation);
    do
    {
        index_2_f(ctx->N, ctx->index_estimation, limits_low, dl, ctx->f_estimation);
        double p = _bayes_posterior(ctx,
                                    num_frames, F,
                                    ctx->f_estimation,
                                    d, v, K,
                                    apriori, apriori_params,
                                    limits_low, ctx->index_max, dl);
        if (p > maxp)
        {
            memcpy(f, ctx->f_estimation, sizeof(double) * ctx->N);
            maxp = p;
        }
    } while (next_index(ctx->N, ctx->index_max, ctx->index_estimation));
}

/**
 * @brief Find estimation E[{f_i}] by integral over all {f_i} of p({f_i}|{F_a}) * {f_i}
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
                    double *f)
{
    double maxp = 0;

    bayes_index_max(ctx->N, limits_low, limits_high, dl, ctx->index_max);
    /* If clip > 0 estimate take into account only points near to maximal position,
       where posterior > maximal * clip
     */
    if (clip > 0)
    {
        init_index(ctx->N, ctx->index_estimation);
        do
        {
            index_2_f(ctx->N, ctx->index_estimation, limits_low, dl, ctx->f_estimation);
            double p = _bayes_posterior(ctx,
                                        num_frames, F,
                                        ctx->f_estimation,
                                        d, v, K,
                                        apriori, apriori_params,
                                        limits_low, ctx->index_max, dl);
            if (p > maxp)
                maxp = p;
        } while (next_index(ctx->N, ctx->index_max, ctx->index_estimation));
    }

    double dln = pow(dl, ctx->N);
    int i;
    double sump = 0;
    memset(f, 0, ctx->N * sizeof(double));
    init_index(ctx->N, ctx->index_estimation);
    do
    {
        index_2_f(ctx->N, ctx->index_estimation, limits_low, dl, ctx->f_estimation);
        double p = _bayes_posterior(ctx,
                                    num_frames, F,
                                    ctx->f_estimation,
                                    d, v, K,
                                    apriori, apriori_params,
                                    limits_low, ctx->index_max, dl);
        if (p > maxp * clip)
        {
            for (i = 0; i < ctx->N; i++)
            {
                f[i] += ctx->f_estimation[i] * p * dln;
                sump += p * dln;
            }
        }
    } while (next_index(ctx->N, ctx->index_max, ctx->index_estimation));

    for (i = 0; i < ctx->N; i++)
    {
        f[i] /= sump;
    }
}

void bayes_posterior_free(struct bayes_posterior_ctx_s *ctx)
{
    if (ctx->f_integration != NULL)
    {
        free(ctx->f_integration);
        ctx->f_integration = NULL;
    }
    if (ctx->f_estimation != NULL)
    {
        free(ctx->f_estimation);
        ctx->f_estimation = NULL;
    }
    if (ctx->index_estimation != NULL)
    {
        free(ctx->index_estimation);
        ctx->index_estimation = NULL;
    }
    if (ctx->index_integration != NULL)
    {
        free(ctx->index_integration);
        ctx->index_integration = NULL;
    }
    if (ctx->index_max != NULL)
    {
        free(ctx->index_max);
        ctx->index_max = NULL;
    }
}

bool bayes_posterior_init(struct bayes_posterior_ctx_s *ctx, int num_dim)
{
    ctx->N = num_dim;
    ctx->f_integration = calloc(num_dim, sizeof(double));
    ctx->f_estimation = calloc(num_dim, sizeof(double));
    ctx->index_max = calloc(num_dim, sizeof(int));
    ctx->index_estimation = calloc(num_dim, sizeof(int));
    ctx->index_integration = calloc(num_dim, sizeof(int));
    return true;
}
