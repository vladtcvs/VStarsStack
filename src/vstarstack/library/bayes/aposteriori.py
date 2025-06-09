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

from typing import Callable
import numpy as np

def posterior_item(F : np.ndarray,
                   f_posteriori : np.ndarray,
                   f_integration : np.ndarray,
                   lambdas_d : np.ndarray,
                   lambdas_v : np.ndarray) -> float:

    lambdas_f_posterior   = lambdas_d + np.matmul(lambdas_v, f_posteriori)
    Lambdas_f_posterior   = np.sum(F * np.log(lambdas_f_posterior)   - lambdas_f_posterior)

    lambdas_f_integration = lambdas_d + np.matmul(lambdas_v, f_integration)    
    Lambdas_f_integration = np.sum(F * np.log(lambdas_f_integration) - lambdas_f_integration)

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
        if index[-1] == num_indexes[-1]:
            break

def find_posterior(F : np.ndarray,
                   f_posteriori : np.ndarray,
                   lambdas_d : np.ndarray,
                   lambdas_v : np.ndarray,
                   apriori : Callable[[np.ndarray], float],
                   limits_low : np.ndarray,
                   limits_high : np.ndarray,
                   dl : float) -> float:
    """
        Find p(f_posteriori | {F_i})
    """
    assert lambdas_v.shape[1] == f_posteriori.shape[0]
    assert lambdas_d.shape[0] == lambdas_v.shape[0]
    assert lambdas_v.shape[0] == F.shape[0]
    assert limits_low.shape[0] == f_posteriori.shape[0]
    assert limits_high.shape[0] == f_posteriori.shape[0]

    apriori_f_posterior = apriori(f_posteriori)
    if apriori_f_posterior == 0:
        return 0
    s = 0
    for indexes in _get_indexes(limits_low, limits_high, dl):
        f_integration : np.ndarray = limits_low + indexes * dl
        apriori_f_integration = apriori(f_integration)
        if abs(apriori_f_integration) < 1e-12:
            continue
        item = posterior_item(F, f_posteriori, f_integration, lambdas_d, lambdas_v) * apriori_f_integration / apriori_f_posterior
        s += item * dl**len(indexes)
    return 1/s
