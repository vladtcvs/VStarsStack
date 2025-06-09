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

from typing import Callable, Tuple
import numpy as np

def Lambda(F : np.ndarray,
           f : np.ndarray,
           lambdas_d : np.ndarray,
           lambdas_v : np.ndarray) -> float:
    lambdas   = lambdas_d + np.matmul(lambdas_v, f)
    Lambdas   = np.sum(F * np.log(lambdas) - lambdas)
    return Lambdas

def _posterior_item(F : np.ndarray,
                   f : np.ndarray,
                   f_integration : np.ndarray,
                   lambdas_d : np.ndarray,
                   lambdas_v : np.ndarray) -> float:
    """
        Arguments:
            F - samples {F_i}
            f - vector of signal f_k
            lambda_d, lambda_v - parameters of lambda_i(f) = d_i + v_ik * f_k
        Return:
            exp(L(f') - L(f)) where L(f) = sum_i F_i*ln(lambda_i(f)) - lambda_i(f)
    """
    Lambdas_f_posterior   = Lambda(F, f, lambdas_d, lambdas_v)
    Lambdas_f_integration = Lambda(F, f_integration, lambdas_d, lambdas_v)
    return np.exp(Lambdas_f_integration - Lambdas_f_posterior)

def _get_indexes(limits_low : np.ndarray, limits_high : np.ndarray, dl : float):
    num_indexes = np.ceil((limits_high - limits_low)/dl).astype(np.uint)
    num_indexes = np.clip(num_indexes, 1, None)
    dims = limits_low.shape[0]
    index = np.zeros((dims))
    while True:
        yield index
        index[0] += 1
        for i in range(dims-1):
            if index[i] == num_indexes[i]:
                index[i] = 0
                index[i+1] += 1
            else:
                break
        if index[-1] == num_indexes[-1]:
            break

def find_posterior(F : np.ndarray,
                   f : np.ndarray,
                   lambdas_d : np.ndarray,
                   lambdas_v : np.ndarray,
                   apriori : Callable[[np.ndarray], float],
                   limits_low : np.ndarray,
                   limits_high : np.ndarray,
                   dl : float) -> float:
    """
        Find p(f | {F_i})
        Arguments:
            F - samples {F_i}
            f - vector of signal f_k
            lambda_d, lambda_v - parameters of lambda_i(f) = d_i + v_ik * f_k
            apriori - apriori probabilities p(f_k)
            limits_low, limits_high - interval of integration of f'_k
            dl - step of integration
        Return:
            p(f | {F_i})
    """
    assert len(F.shape) == 1
    assert lambdas_v.shape[1] == f.shape[0]
    assert lambdas_d.shape[0] == lambdas_v.shape[0]
    assert lambdas_v.shape[0] == F.shape[0]
    assert limits_low.shape[0] == f.shape[0]
    assert limits_high.shape[0] == f.shape[0]

    apriori_f_posterior = apriori(f)
    if apriori_f_posterior == 0:
        return 0
    s = 0
    ndim = limits_low.shape[0]
    for indexes in _get_indexes(limits_low, limits_high, dl):
        f_integration : np.ndarray = limits_low + indexes * dl
        apriori_f_integration = apriori(f_integration)
        if abs(apriori_f_integration) < 1e-12:
            continue
        item = _posterior_item(F, f, f_integration, lambdas_d, lambdas_v) * apriori_f_integration / apriori_f_posterior
        s += item * dl**ndim
    return 1/s

def bayes_maxp(F : np.ndarray,
               lambdas_d : np.ndarray,
               lambdas_v : np.ndarray,
               apriori : Callable[[np.ndarray], float],
               limits_low : np.ndarray,
               limits_high : np.ndarray,
               dl : float) -> np.ndarray:
    f_maxp = np.zeros(lambdas_d.shape)
    maxp = 0
    for indexes in _get_indexes(limits_low, limits_high, dl):
        f = limits_low + indexes * dl
        p = find_posterior(F, f, lambdas_d, lambdas_v, apriori, limits_low, limits_high, dl)
        if p > maxp:
            f_maxp = f
            maxp = p
    return f_maxp

def bayes_estimation(F : np.ndarray,
                     lambdas_d : np.ndarray,
                     lambdas_v : np.ndarray,
                     apriori : Callable[[np.ndarray], float],
                     limits_low : np.ndarray,
                     limits_high : np.ndarray,
                     dl : float,
                     clip : float = 0) -> np.ndarray:
    maxp = 0
    probs = []
    for indexes in _get_indexes(limits_low, limits_high, dl):
        f = limits_low + indexes * dl
        p = find_posterior(F, f, lambdas_d, lambdas_v, apriori, limits_low, limits_high, dl)
        probs.append((p,f))
        if p > maxp:
            maxp = p

    ndim = limits_low.shape[0]
    probs = [ item for item in probs if item[0] > maxp*clip ]
    val = 0
    sump = np.zeros((ndim,))
    for p, f in probs:
        val += f * p * dl**ndim
        sump += p * dl**ndim
    return val/sump
