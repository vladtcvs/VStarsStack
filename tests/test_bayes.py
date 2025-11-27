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

import numpy as np
from vstarstack.library.bayes.bayes import BayesEstimator
from vstarstack.library.bayes.estimation import *

def test_MAP_single_value():
    f_signal_max = 10
    dl = 0.1
    dark = 5
    signal = 3
    v = 1
    K = 1
    samples = np.array([dark + signal * v * K], dtype=np.uint64)

    low = np.array([0], dtype=np.double)
    high = np.array([f_signal_max], dtype=np.double)
    estimator = BayesEstimator(apriori = "uniform", dl=dl, ndim=1)
    lambdas_K = np.array([[1]], dtype=np.double)
    lambdas_v = np.array([1], dtype=np.double)
    lambdas_d = np.array([5], dtype=np.double)

    f = estimator.MAP(F=samples,
                      lambdas_d=lambdas_d,
                      lambdas_v=lambdas_v,
                      lambdas_K=lambdas_K,
                      apriori_params=None,
                      limits_low=low,
                      limits_high=high)
    assert f == signal

def test_MAP_1():
    background = np.zeros((5,5))
    flat = np.ones((5,5))
    true_signal = np.ones((5,5))*4

    nsamples = 10
    samples = np.zeros((nsamples, 5, 5))
    backgrounds = np.zeros((nsamples, 5, 5))
    flats = np.zeros((nsamples, 5, 5))
    Ks = np.zeros((nsamples, 1))
    for i in range(nsamples):
        samples[i,:,:] = true_signal * flat + background
        backgrounds[i,:,:] = background
        flats[i,:,:] = flat
        Ks[i,0] = 1

    max_signal = np.array([10])
    clip = 0.8
    dl = 0.05
    apriori_params = None
    estimator = BayesEstimator(apriori = "uniform", dl=dl, ndim=1)

    estimated = estimate(samples, Ks, backgrounds, flats, max_signal, estimator, apriori_params, clip, "MAP")
    assert len(estimated.shape) == 3
    assert estimated.shape[0] == true_signal.shape[0]
    assert estimated.shape[1] == true_signal.shape[1]
    assert estimated.shape[2] == 1
    estimated = estimated[:,:,0]
    assert (estimated == true_signal).all()

def test_dark_flat_1():
    dark = np.zeros((5,5))
    flat = np.ones((5,5))
    true_signal = np.ones((5,5))*4

    nsamples = 10
    samples = np.zeros((nsamples, 5, 5))
    for i in range(nsamples):
        samples[i,:,:] = true_signal * flat + dark

    max_signal = 10
    clip = 0.8
    dl = 0.05

    apriori_params = None

    estimated = estimate_with_dark_flat(samples,
                                        dark,
                                        flat,
                                        max_signal,
                                        dl,
                                        "uniform",
                                        None,
                                        clip,
                                        "MAP")

    assert len(estimated.shape) == 3
    assert estimated.shape[0] == true_signal.shape[0]
    assert estimated.shape[1] == true_signal.shape[1]
    assert estimated.shape[2] == 1
    estimated = estimated[:,:,0]
    assert (estimated == true_signal).all()

def test_dark_flat_sky_1():
    dark = np.zeros((5,5))
    sky = np.ones((5,5))
    flat = np.ones((5,5))
    true_signal = np.ones((5,5))*4

    nsamples = 10
    samples = np.zeros((nsamples, 5, 5))
    for i in range(nsamples):
        samples[i,:,:] = true_signal * flat + sky * flat + dark

    max_signal = 10
    clip = 0.8
    dl = 0.05

    apriori_params = None

    estimated = estimate_with_dark_flat_sky(samples,
                                            dark,
                                            flat,
                                            sky,
                                            max_signal,
                                            dl,
                                            "uniform",
                                            None,
                                            clip,
                                            "MAP")

    assert len(estimated.shape) == 3
    assert estimated.shape[0] == true_signal.shape[0]
    assert estimated.shape[1] == true_signal.shape[1]
    assert estimated.shape[2] == 1
    estimated = estimated[:,:,0]
    assert (estimated == true_signal).all()

def test_dark_flat_sky_continuum_1():
    h = 1
    w = 1
    dark = np.zeros((h,w))
    sky = np.ones((h,w))
    flat = np.ones((h,w))
    true_signal_emission = np.ones((h,w))*1
    true_signal_continuum = np.ones((h,w))*0.5
    K = 2

    nsamples_wide = 10
    samples_wide = np.zeros((nsamples_wide, h, w))
    for i in range(nsamples_wide):
        samples_wide[i,:,:] = (true_signal_continuum * K + true_signal_emission) * flat + sky * flat + dark

    nsamples_narrow = 10
    samples_narrow = np.zeros((nsamples_narrow, h, w))
    for i in range(nsamples_narrow):
        samples_narrow[i,:,:] = (true_signal_continuum + true_signal_emission) * flat + sky * flat + dark

    max_signal = 4
    clip = 0.8
    dl = 0.02

    apriori_params = None

    estimated = estimate_with_dark_flat_sky_continuum(samples_narrow,
                                                      dark,
                                                      flat,
                                                      sky,
                                                      max_signal,
                                                      samples_wide,
                                                      dark,
                                                      flat,
                                                      sky,
                                                      max_signal,
                                                      K,
                                                      dl,
                                                      "uniform",
                                                      apriori_params,
                                                      clip,
                                                      "MAP")

    assert len(estimated.shape) == 3
    assert estimated.shape[0] == true_signal_emission.shape[0]
    assert estimated.shape[1] == true_signal_emission.shape[1]
    assert estimated.shape[2] == 2
    estimated_continuum = estimated[:,:,0]
    estimated_emission = estimated[:,:,1]
    print(estimated_continuum)
    print(estimated_emission)
    assert (estimated_continuum == true_signal_continuum).all()
    assert (estimated_emission == true_signal_emission).all()
