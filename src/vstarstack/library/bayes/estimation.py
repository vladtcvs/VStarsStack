"""Bayes estimation methods"""
#
# Copyright (c) 2025 Vladislav Tsendrovskii
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

from typing import Callable, Iterator, Tuple
import numpy as np
import vstarstack.library.bayes
import vstarstack.library.bayes.bayes

def _generate_crds(H : int, W : int) -> Iterator[Tuple[int, int]]:
    for y in range(H):
        for x in range(W):
            yield y,x

def estimate(samples : np.ndarray,
             Ks : np.ndarray,
             background : np.ndarray,
             flat : np.ndarray,
             max_signal : np.ndarray,
             estimator : vstarstack.library.bayes.bayes.BayesEstimator,
             apriori_fun_params : any = None,
             clip : float = 0) -> np.ndarray:
    """
    Estimate value with Bayes theorem
    
    samples    - (nsamples, h, w) - 
    Ks         - (nsamples, ndim) - 
    background - (nsamples, h, w) - 
    flat       - (nsamples, h, w) - 
    max_signal - (ndim)           -

    estimator          - 
    apriori_fun_params - 
    clip               - 
    """

    # lambda(f1, f2) = background + flat * (Ks_1 * f1 + Ks_2 * f2)

    assert len(max_signal.shape) == 1
    ndim = max_signal.shape[0]
    max_signal = max_signal.astype(np.double)
    min_signal = np.zeros(max_signal.shape, dtype=np.double)

    assert len(samples.shape) == 3
    nsamples = samples.shape[0]
    h = samples.shape[1]
    w = samples.shape[2]
    samples = samples.astype(np.uint)

    assert len(background.shape) == 3
    assert background.shape[0] == nsamples
    assert background.shape[1] == h
    assert background.shape[2] == w
    background = background.astype(np.double)

    assert len(flat.shape) == 3
    assert flat.shape[0] == nsamples
    assert flat.shape[1] == h
    assert flat.shape[2] == w
    background = background.astype(np.double)

    assert len(Ks.shape) == 2
    assert Ks.shape[0] == nsamples
    assert Ks.shape[1] == ndim
    flat = flat.astype(np.double)

    npixels = w*h
    samples = np.reshape(samples, (nsamples, npixels))
    background = np.reshape(background, (nsamples, npixels))
    flat = np.reshape(flat, (nsamples, npixels))

    result = np.ndarray((npixels, ndim))
    for i in range(npixels):
        f = estimator.estimate(samples[:,i],
                               background[:,i],
                               flat[:,i],
                               Ks,
                               apriori_fun_params,
                               limits_low=min_signal,
                               limits_high=max_signal,
                               clip=clip)
        result[i,:] = f
    result = np.reshape(result, (h, w, ndim))
    return result

def estimate_with_dark_flat(samples : np.ndarray,
                            dark : np.ndarray,
                            flat : np.ndarray,
                            max_signal : float,
                            integration_dl : float,
                            apriori_fun : any,
                            apriori_fun_params : any = None,
                            clip : float = 0) -> np.ndarray:
    assert len(samples.shape) == 3
    nsamples = samples.shape[0]

    # lambda(f) = dark + flat * 1 * f

    if len(dark.shape) == 2:
        dks = np.zeros((nsamples, dark.shape[0], dark.shape[0]))
        for i in range(nsamples):
            dks[i,:,:] = dark
    elif len(dark.shape) == 3:
        dks = dark
    else:
        return None

    if len(flat.shape) == 2:
        flts = np.zeros((nsamples, flat.shape[0], flat.shape[0]))
        for i in range(nsamples):
            flts[i,:,:] = flat
    elif len(flat.shape) == 3:
        flts = flat
    else:
        return None

    Ks = np.ones((nsamples, 1))

    _max_signal = np.array([max_signal], dtype=np.double)

    estimator = vstarstack.library.bayes.bayes.BayesEstimator(apriori=apriori_fun, dl=integration_dl, ndim=1)
    return estimate(samples, Ks, dks, flts, _max_signal, estimator, apriori_fun_params, clip)

def estimate_with_dark_flat_sky(samples : np.ndarray,
                                dark : np.ndarray,
                                flat : np.ndarray,
                                sky : np.ndarray,
                                max_signal : float,
                                integration_dl : float,
                                apriori_fun : any,
                                apriori_fun_params : any = None,
                                clip : float = 0) -> np.ndarray:
    assert len(samples.shape) == 3
    nsamples = samples.shape[0]

    # lambda(f) = (dark + flat * sky) + flat * 1 * f

    if len(darks.shape) == 2:
        darks = np.zeros((nsamples, dark.shape[0], dark.shape[0]))
        for i in range(nsamples):
            darks[i,:,:] = dark
    elif len(darks.shape) == 3:
        pass
    else:
        return None

    if len(flat.shape) == 2:
        flts = np.zeros((nsamples, flat.shape[0], flat.shape[0]))
        for i in range(nsamples):
            flts[i,:,:] = flat
    elif len(flat.shape) == 3:
        flts = flat
    else:
        return None

    if len(sky.shape) == 2:
        skies = np.zeros((nsamples, sky.shape[0], sky.shape[0]))
        for i in range(nsamples):
            skies[i,:,:] = sky
    elif len(sky.shape) == 3:
        skies = sky
    else:
        return None

    bgs = darks + skies * flts

    Ks = np.ones((nsamples, 1))

    _max_signal = np.array([max_signal], dtype=np.double)

    estimator = vstarstack.library.bayes.bayes.BayesEstimator(apriori=apriori_fun, dl=integration_dl, ndim=1)
    return estimate(samples, Ks, bgs, flts, _max_signal, estimator, apriori_fun_params, clip)

def estimate_with_dark_flat_sky_continuum(samples_narrow : np.ndarray,
                                          dark_narrow : np.ndarray,
                                          flat_narrow : np.ndarray,
                                          sky_narrow : np.ndarray,
                                          max_signal_emission : float,
                                          samples_wide : np.ndarray,
                                          dark_wide : np.ndarray,
                                          flat_wide : np.ndarray,
                                          sky_wide : np.ndarray,
                                          max_signal_continuum : float,
                                          wide_narrow_k : float,
                                          integration_dl : float,
                                          apriori_fun : str | Callable[[np.ndarray], float],
                                          apriori_fun_params : any = None,
                                          clip : float = 0) -> np.ndarray:
    assert len(samples_wide.shape) == 3
    assert len(samples_narrow.shape) == 3
    nsamples_wide = samples_wide.shape[0]
    nsamples_narrow = samples_narrow.shape[0]

    # lambda_narrow(f) = (dark_narrow + flat_narrow * sky_narrow) + flat_narrow * (1 * f_continuum + 1 * f_emission)
    # lambda_wide(f) = (dark_wide + flat_wide * sky_wide) + flat_wide * (k * f_continuum + 1 * f_emission)

    # f = [f_continuum f_emission]

    # prepare parameters for images with wide filter
    if len(dark_wide.shape) == 2:
        darks_wide = np.zeros((nsamples_wide, dark_wide.shape[0], dark_wide.shape[0]))
        for i in range(nsamples_wide):
            darks_wide[i,:,:] = dark_wide
    elif len(dark_wide.shape) == 3:
        darks_wide = dark_wide
    else:
        return None

    if len(flat_wide.shape) == 2:
        flats_wide = np.zeros((nsamples_wide, flat_wide.shape[0], flat_wide.shape[0]))
        for i in range(nsamples_wide):
            flats_wide[i,:,:] = flat_wide
    elif len(flat_wide.shape) == 3:
        flats_wide = flat_wide
    else:
        return None

    if len(sky_wide.shape) == 2:
        skies_wide = np.zeros((nsamples_wide, sky_wide.shape[0], sky_wide.shape[0]))
        for i in range(nsamples_wide):
            skies_wide[i,:,:] = sky_wide
    elif len(sky_wide.shape) == 3:
        skies_wide = sky_wide
    else:
        return None

    bgs_wide = darks_wide + skies_wide * flats_wide

    Ks_wide = np.ones((nsamples_wide, 2))

    # prepare parameters for images with narrow filter
    if len(dark_narrow.shape) == 2:
        darks_narrow = np.zeros((nsamples_narrow, dark_narrow.shape[0], dark_narrow.shape[0]))
        for i in range(nsamples_narrow):
            darks_narrow[i,:,:] = dark_narrow
    elif len(dark_narrow.shape) == 3:
        darks_narrow = dark_narrow
    else:
        return None

    if len(flat_narrow.shape) == 2:
        flats_narrow = np.zeros((nsamples_narrow, flat_narrow.shape[0], flat_narrow.shape[0]))
        for i in range(nsamples_narrow):
            flats_narrow[i,:,:] = flat_narrow
    elif len(flat_narrow.shape) == 3:
        flats_narrow = flat_narrow
    else:
        return None

    if len(sky_narrow.shape) == 2:
        skies_narrow = np.zeros((nsamples_narrow, sky_narrow.shape[0], sky_narrow.shape[0]))
        for i in range(nsamples_narrow):
            skies_narrow[i,:,:] = sky_narrow
    elif len(sky_narrow.shape) == 3:
        skies_narrow = sky_narrow
    else:
        return None

    bgs_narrow = darks_narrow + skies_narrow * flats_narrow

    Ks_narrow = np.zeros((nsamples_narrow, 2))
    Ks_narrow[:,0] = wide_narrow_k
    Ks_narrow[:,1] = 1

    # concat all samples into single array
    samples    = np.concat([samples_wide, samples_narrow], axis=0)
    bgs        = np.concat([bgs_wide, bgs_narrow], axis=0)
    flats      = np.concat([flats_wide, flats_narrow], axis=0)
    Ks         = np.concat([Ks_wide, Ks_narrow], axis=0)
    max_signal = np.array([max_signal_emission, max_signal_continuum])
    estimator  = vstarstack.library.bayes.bayes.BayesEstimator(apriori=apriori_fun, dl=integration_dl, ndim=2)
    return estimate(samples, Ks, bgs, flats, max_signal, estimator, apriori_fun_params, clip)
