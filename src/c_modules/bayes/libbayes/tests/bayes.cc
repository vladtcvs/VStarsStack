#include <gtest/gtest.h>
#include <bayes.h>

extern "C" double 

TEST(dim_1, aposteriori) {
    struct bayes_posterior_ctx_s ctx;
    bayes_posterior_init(&ctx, 1);

    uint64_t F[] = {1, 1, 1, 1, 1};  // Samples
    double f[1] = {1};          // target for which find aposteriori
    double d[1] = {0};          // darks
    double v[1] = {1};          // flats
    double Ks[1] = {1};         // K
    double lows[1] = {0};
    double highs[1] = {10};
    double dl = 0.1;

    auto apriori = [](const double *f, int N, const void *param) -> double
    {
        if (f[0] < 0)
            return 0;
        if (f[0] > 10)
            return 0;
        return 1.0/10;
    };

    double posterior = bayes_posterior(&ctx, sizeof(F)/sizeof(F[0]), F, f, d, v, Ks, apriori, NULL, lows, highs, dl);

    bayes_posterior_free(&ctx);
}
