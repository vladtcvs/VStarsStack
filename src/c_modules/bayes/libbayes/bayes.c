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

static void index_2_f(int num_dim, const int *index, const double *low, double dl, double *f)
{
    int i;
    for (i = 0; i < num_dim; i++)
        f[i] = low[i] + dl * index[i];
}

static double Lambda(int num_dim,
                     int num_frames,
                     const uint64_t *F,
                     const double *f,
                     const double *lambdas_d,
                     const double *lambdas_v)
{
    double sum = 0;
    int i, k;
    for (i = 0; i < num_frames; i++)
    {
        double item = lambdas_d[i];
        for (k = 0; k < num_dim; k++)
            item += lambdas_v[i * num_dim + k] * f[k];
        sum += F[i] * log(item) - item;
    }
    return sum;
}

static double posterior_item(int num_dim,
                             int num_frames,
                             const uint64_t *F,
                             const double *f,
                             const double *f_integration,
                             const double *lambdas_d,
                             const double *lambdas_v)
{
    double Lambdas_f_posterior = Lambda(num_dim, num_frames, F, f, lambdas_d, lambdas_v);
    double Lambdas_f_integration = Lambda(num_dim, num_frames, F, f_integration, lambdas_d, lambdas_v);
    return exp(Lambdas_f_integration - Lambdas_f_posterior);
}

static double _bayes_posterior(struct bayes_posterior_ctx_s *ctx,
                               int num_frames,
                               const uint64_t *F,
                               const double *f,
                               const double *lambdas_d,
                               const double *lambdas_v,
                               const apriori_f apriori,
                               const void *apriori_params,
                               const double *limits_low,
                               const int *index_max,
                               double dl)
{
    double apriori_f = apriori(f, ctx->num_dim, apriori_params);
    if (apriori_f > -1e-12 && apriori_f < 1e-12)
        return 0;
    double s = 0;
    double dln = pow(dl, ctx->num_dim);
    init_index(ctx->num_dim, ctx->index_integration);
    do
    {
        index_2_f(ctx->num_dim, ctx->index_integration, limits_low, dl, ctx->f_integration);
        double apriori_f_integration = apriori(ctx->f_integration, ctx->num_dim, apriori_params);
        if (apriori_f_integration > -1e-12 && apriori_f_integration < 1e-12)
            continue;
        double item = posterior_item(ctx->num_dim, num_frames, F, f, ctx->f_integration, lambdas_d, lambdas_v) * apriori_f_integration;
        s += item * dln;
    } while (next_index(ctx->num_dim, index_max, ctx->index_integration));
    if (s < 1e-14)
        return -1;
    return apriori_f / s;
}

static void bayes_index_max(int num_dim,
                            const double *limits_low,
                            const double *limits_high,
                            double dl,
                            int *index_max)
{
    int i;
    for (i = 0; i < num_dim; i++)
    {
        index_max[i] = ceilf((limits_high[i] - limits_low[i]) / dl);
    }
}

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
                       double dl)
{
    bayes_index_max(ctx->num_dim, limits_low, limits_high, dl, ctx->index_max);
    return _bayes_posterior(ctx, num_frames, F, f, lambdas_d, lambdas_v, apriori, apriori_params, limits_low, ctx->index_max, dl);
}

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
                double *f)
{
    double maxp = 0;
    bayes_index_max(ctx->num_dim, limits_low, limits_high, dl, ctx->index_max);
    init_index(ctx->num_dim, ctx->index_estimation);
    do
    {
        index_2_f(ctx->num_dim, ctx->index_estimation, limits_low, dl, ctx->f_estimation);
        double p = _bayes_posterior(ctx,
                                    num_frames, F,
                                    ctx->f_estimation,
                                    lambdas_d, lambdas_v,
                                    apriori, apriori_params,
                                    limits_low, ctx->index_max, dl);
        if (p > maxp)
        {
            memcpy(f, ctx->f_estimation, sizeof(double) * ctx->num_dim);
            maxp = p;
        }
    } while (next_index(ctx->num_dim, ctx->index_max, ctx->index_estimation));
}

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
                    double *f)
{
    double maxp = 0;

    bayes_index_max(ctx->num_dim, limits_low, limits_high, dl, ctx->index_max);
    if (clip > 0)
    {
        init_index(ctx->num_dim, ctx->index_estimation);
        do
        {
            index_2_f(ctx->num_dim, ctx->index_estimation, limits_low, dl, ctx->f_estimation);
            double p = _bayes_posterior(ctx,
                                        num_frames, F,
                                        ctx->f_estimation,
                                        lambdas_d, lambdas_v,
                                        apriori, apriori_params,
                                        limits_low, ctx->index_max, dl);
            if (p > maxp)
                maxp = p;
        } while (next_index(ctx->num_dim, ctx->index_max, ctx->index_estimation));
    }

    double dln = pow(dl, ctx->num_dim);
    int i;
    double sump = 0;
    memset(f, 0, ctx->num_dim * sizeof(double));
    init_index(ctx->num_dim, ctx->index_estimation);
    do
    {
        index_2_f(ctx->num_dim, ctx->index_estimation, limits_low, dl, ctx->f_estimation);
        double p = _bayes_posterior(ctx,
                                    num_frames, F,
                                    ctx->f_estimation,
                                    lambdas_d, lambdas_v,
                                    apriori, apriori_params,
                                    limits_low, ctx->index_max, dl);
        if (p > maxp * clip)
        {
            for (i = 0; i < ctx->num_dim; i++)
            {
                f[i] += ctx->f_estimation[i] * p * dln;
                sump += p * dln;
            }
        }
    } while (next_index(ctx->num_dim, ctx->index_max, ctx->index_estimation));

    for (i = 0; i < ctx->num_dim; i++)
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
    bayes_posterior_free(ctx);
    ctx->num_dim = num_dim;
    ctx->f_integration = calloc(num_dim, sizeof(double));
    ctx->f_estimation = calloc(num_dim, sizeof(double));
    ctx->index_max = calloc(num_dim, sizeof(int));
    ctx->index_estimation = calloc(num_dim, sizeof(int));
    ctx->index_integration = calloc(num_dim, sizeof(int));
    return true;
}
