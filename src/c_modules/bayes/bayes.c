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

static void index_2_f(int num_dim, const int *index, const float *low, float dl, float *f)
{
    int i;
    for (i = 0; i < num_dim; i++)
        f[i] = low[i] + dl * index[i];
}

static float Lambda(int num_dim,
                    int num_frames,
                    const int *F,
                    const float *f,
                    const float *lambdas_d,
                    const float **lambdas_v)
{
    float sum = 0;
    int i, k;
    for (i = 0; i < num_frames; i++)
    {
        float item = lambdas_d[i];
        for (k = 0; k < num_dim; k++)
            item += lambdas_v[i][k] * f[k];
        sum += F[i] * logf(item) - item;
    }
    return sum;
}

static float posterior_item(int num_dim,
                            int num_frames,
                            const int *F,
                            const float *f,
                            const float *f_integration,
                            const float *lambdas_d,
                            const float **lambdas_v)
{
    float Lambdas_f_posterior = Lambda(num_dim, num_frames, F, f, lambdas_d, lambdas_v);
    float Lambdas_f_integration = Lambda(num_dim, num_frames, F, f_integration, lambdas_d, lambdas_v);
    return expf(Lambdas_f_integration - Lambdas_f_posterior);
}

float bayes_posterior(struct bayes_posterior_ctx_s *ctx,
                      int num_frames,
                      const int *F,
                      const float *f,
                      const float *lambdas_d,
                      const float **lambdas_v,
                      const apriori_f apriori,
                      const void *apriori_params,
                      const float *limits_low,
                      const int *index_max,
                      float dl)
{
    float apriori_f = apriori(f, ctx->num_dim, apriori_params);
    if (apriori_f > -1e-12 && apriori_f < 1e-12)
        return 0;
    float s = 0;
    float dln = powf(dl, ctx->num_dim);
    init_index(ctx->num_dim, ctx->index_integration);
    do
    {
        index_2_f(ctx->num_dim, ctx->index_integration, limits_low, dl, ctx->f_integration);
        float apriori_f_integration = apriori(ctx->f_integration, ctx->num_dim, apriori_params);
        if (apriori_f_integration > -1e-12 && apriori_f_integration < 1e-12)
            continue;
        float item = posterior_item(ctx->num_dim, num_frames, F, f, ctx->f_integration, lambdas_d, lambdas_v) * apriori_f_integration;
        s += item * dln;
    } while (next_index(ctx->num_dim, index_max, ctx->index_integration));
    if (s < 1e-14)
        return -1;
    return apriori_f / s;
}

void bayes_index_max(int num_dim,
                     const float *limits_low,
                     const float *limits_high,
                     float dl,
                     int *index_max)
{
    int i;
    for (i = 0; i < num_dim; i++)
    {
        index_max[i] = ceilf((limits_high[i] - limits_low[i]) / dl);
    }
}

void bayes_maxp(struct bayes_posterior_ctx_s *ctx,
                int num_frames,
                const int *F,
                const float *lambdas_d,
                const float **lambdas_v,
                const apriori_f apriori,
                const void *apriori_params,
                const float *limits_low,
                const float *limits_high,
                float dl,
                float *f)
{
    float maxp = 0;
    bayes_index_max(ctx->num_dim, limits_low, limits_high, dl, ctx->index_max);
    init_index(ctx->num_dim, ctx->index_estimation);
    do
    {
        index_2_f(ctx->num_dim, ctx->index_estimation, limits_low, dl, ctx->f_estimation);
        float p = bayes_posterior(ctx,
                                  num_frames, F,
                                  ctx->f_estimation,
                                  lambdas_d, lambdas_v,
                                  apriori, apriori_params,
                                  limits_low, ctx->index_max, dl);
        if (p > maxp)
        {
            memcpy(f, ctx->f_estimation, sizeof(float) * ctx->num_dim);
            maxp = p;
        }
    } while (next_index(ctx->num_dim, ctx->index_max, ctx->index_estimation));
}

void bayes_estimate(struct bayes_posterior_ctx_s *ctx,
                    int num_frames,
                    const int *F,
                    const float *lambdas_d,
                    const float **lambdas_v,
                    const apriori_f apriori,
                    const void *apriori_params,
                    const float *limits_low,
                    const float *limits_high,
                    float dl,
                    float clip,
                    float *f)
{
    float maxp = 0;

    bayes_index_max(ctx->num_dim, limits_low, limits_high, dl, ctx->index_max);
    if (clip > 0)
    {
        init_index(ctx->num_dim, ctx->index_estimation);
        do
        {
            index_2_f(ctx->num_dim, ctx->index_estimation, limits_low, dl, ctx->f_estimation);
            float p = bayes_posterior(ctx,
                                      num_frames, F,
                                      ctx->f_estimation,
                                      lambdas_d, lambdas_v,
                                      apriori, apriori_params,
                                      limits_low, ctx->index_max, dl);
            if (p > maxp)
                maxp = p;
        } while (next_index(ctx->num_dim, ctx->index_max, ctx->index_estimation));
    }

    float dln = powf(dl, ctx->num_dim);
    int i;
    float sump = 0;
    init_index(ctx->num_dim, ctx->index_estimation);
    do
    {
        index_2_f(ctx->num_dim, ctx->index_estimation, limits_low, dl, ctx->f_estimation);
        float p = bayes_posterior(ctx,
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
    ctx->f_integration = calloc(num_dim, sizeof(float));
    ctx->f_estimation = calloc(num_dim, sizeof(float));
    ctx->index_max = calloc(num_dim, sizeof(int));
    ctx->index_estimation = calloc(num_dim, sizeof(int));
    ctx->index_integration = calloc(num_dim, sizeof(int));
    return true;
}
