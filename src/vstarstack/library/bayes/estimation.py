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

from typing import Callable, Generator
import numpy as np
import vstarstack.library.bayes
import vstarstack.library.bayes.bayes

def _generate_crds(H : int, W : int) -> Generator[int, int]:
    for y in range(H):
        for x in range(W):
            yield y,x

def estimate(samples : np.ndarray,
             backgrounds : np.ndarray,
             nus : np.ndarray,
             max_signal : np.ndarray,
             estimator : vstarstack.library.bayes.bayes.BayesEstimator,
             apriori_fun_params : any = None,
             clip : float = 0) -> np.ndarray:

    # lambda(f1, f2) = background + nu_1 * f1 + nu_2 * f2

    assert len(max_signal.shape) == 1
    ndim = max_signal.shape[0]
    max_signal = max_signal.astype(np.double)
    min_signal = np.zeros(max_signal.shape, dtype=np.double)

    assert len(samples.shape) == 3
    nsamples = samples.shape[0]
    h = samples.shape[1]
    w = samples.shape[2]
    samples = samples.astype(np.uint)

    assert len(backgrounds.shape) == 3
    assert backgrounds.shape[0] == nsamples
    assert backgrounds.shape[1] == h
    assert backgrounds.shape[2] == w
    backgrounds = backgrounds.astype(np.double)

    assert len(nus.shape) == 4
    assert nus.shape[0] == nsamples
    assert nus.shape[1] == h
    assert nus.shape[2] == w
    assert nus.shape[3] == ndim
    nus = nus.astype(np.double)

    result = np.ndarray((h, w, ndim))

    for y,x in _generate_crds(h, w):
        f = estimator.estimate(samples[:,y,x],
                               backgrounds[:,y,x],
                               nus[:,y,x,:],
                               apriori_fun_params,
                               limits_low=min_signal,
                               limits_high=max_signal,
                               clip=clip)
        result[y,x] = f
    return result

def estimate_with_dark_flat(samples : np.ndarray,
                            darks : np.ndarray,
                            flats : np.ndarray,
                            max_signal : float,
                            integration_dl : float,
                            apriori_fun : any,
                            apriori_fun_params : any = None,
                            clip : float = 0) -> np.ndarray:
    assert len(samples.shape) == 3
    nsamples = samples.shape[0]

    if len(darks.shape) == 2:
        bgs = np.zeros((nsamples, darks.shape[0], darks.shape[0]))
        for i in range(nsamples):
            bgs[i,:,:] = darks
    elif len(darks.shape) == 3:
        bgs = darks
    else:
        return None

    if len(flats.shape) == 2:
        nus = np.zeros((nsamples, flats.shape[0], flats.shape[0], 1))
        for i in range(nsamples):
            nus[i,:,:,0] = flats
    elif len(flats.shape) == 3:
        nus = flats.reshape(nsamples, flats.shape[1], flats.shape[2], 1)
    else:
        return None

    _max_signal = np.array([max_signal], dtype=np.double)

    estimator = vstarstack.library.bayes.bayes.BayesEstimator(apriori=apriori_fun, dl=integration_dl, ndim=1)
    return estimate(samples, bgs, nus, _max_signal, estimator, apriori_fun_params, clip)

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

    if len(darks.shape) == 2:
        darks = np.zeros((nsamples, dark.shape[0], dark.shape[0]))
        for i in range(nsamples):
            darks[i,:,:] = dark
    elif len(darks.shape) == 3:
        bgs = darks
    else:
        return None

    if len(flat.shape) == 2:
        nus = np.zeros((nsamples, flat.shape[0], flat.shape[0], 1))
        for i in range(nsamples):
            nus[i,:,:,0] = flat
    elif len(flat.shape) == 3:
        nus = flat.reshape(nsamples, flat.shape[1], flat.shape[2], 1)
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

    bgs = darks + skies

    _max_signal = np.array([max_signal], dtype=np.double)

    estimator = vstarstack.library.bayes.bayes.BayesEstimator(apriori=apriori_fun, dl=integration_dl, ndim=1)
    return estimate(samples, bgs, nus, _max_signal, estimator, apriori_fun_params, clip)

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

    # f = (f_n f_c)

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
        nus_wide = np.zeros((nsamples_wide, flat_wide.shape[0], flat_wide.shape[0], 2))
        for i in range(nsamples_wide):
            nus_wide[i,:,:,0] = flat_wide
            nus_wide[i,:,:,1] = flat_wide * wide_narrow_k
    elif len(flat_wide.shape) == 3:
        nus_wide = np.zeros((nsamples_wide, flat_wide.shape[0], flat_wide.shape[0], 2))
        for i in range(nsamples_wide):
            nus_wide[i,:,:,0] = flat_wide[i,:,:]
            nus_wide[i,:,:,1] = flat_wide[i,:,:] * wide_narrow_k
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
    
    bgs_wide = darks_wide + skies_wide * nus_wide[:,:,:,0] # 0 because nu for sky doesn't include wide_narrow_k

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
        nus_narrow = np.zeros((nsamples_narrow, flat_narrow.shape[0], flat_narrow.shape[0], 2))
        for i in range(nsamples_narrow):
            nus_narrow[i,:,:,0] = flat_narrow
            nus_narrow[i,:,:,1] = flat_narrow
    elif len(flat_narrow.shape) == 3:
        nus_narrow = np.zeros((nsamples_narrow, flat_narrow.shape[0], flat_narrow.shape[0], 2))
        for i in range(nsamples_narrow):
            nus_narrow[i,:,:,0] = flat_narrow[i,:,:]
            nus_narrow[i,:,:,1] = flat_narrow[i,:,:]
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

    bgs_narrow = darks_narrow + skies_narrow * nus_narrow[:,:,:,0] # 0 because nu for sky doesn't include wide_narrow_k

    # concat all samples into single array
    samples    = np.concat([samples_wide, samples_narrow], axis=0)
    bgs        = np.concat([bgs_wide, bgs_narrow], axis=0)
    nus        = np.concat([nus_wide, nus_narrow], axis=0)
    max_signal = np.array([max_signal_emission, max_signal_continuum])
    estimator  = vstarstack.library.bayes.bayes.BayesEstimator(apriori=apriori_fun, dl=integration_dl, ndim=2)
    return estimate(samples, bgs, nus, max_signal, estimator, apriori_fun_params, clip)
