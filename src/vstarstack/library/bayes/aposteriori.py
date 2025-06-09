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

import numpy as np

def _posterior_item(F : np.ndarray,
                   f_posteriori : np.ndarray,
                   f_integration : np.ndarray,
                   lambdas_d : np.ndarray,
                   lambdas_v : np.ndarray,
                   apriori_f_posterior : float,
                   apriori_f_integration : float) -> float:

    lambdas_f_posterior   = lambdas_d + np.matmul(lambdas_v, f_posteriori)
    Lambdas_f_posterior   = np.sum(F * np.log(lambdas_f_posterior)   - lambdas_f_posterior)

    lambdas_f_integration = lambdas_d + np.matmul(lambdas_v, f_integration)    
    Lambdas_f_integration = np.sum(F * np.log(lambdas_f_integration) - lambdas_f_integration)

    return np.exp(Lambdas_f_integration - Lambdas_f_posterior) * apriori_f_integration / apriori_f_posterior

def _find_apriori(apriory_table : np.ndarray, indexes : np.ndarray)->float:
    """Apriori probability p(f_intergation)"""
    return apriory_table[tuple(indexes)]

def _posterior_item(F : np.ndarray,
                    f_posteriori : np.ndarray,
                    lambdas_d : np.ndarray,
                    lambdas_v : np.ndarray,
                    apriori : np.ndarray,
                    apriori_f_posterior : float,
                    indexes_integration : np.ndarray,
                    limits_low : np.ndarray,
                    dl : float) -> float:
    f_integration : np.ndarray = limits_low + indexes_integration * dl
    apriori_f_integration : float = _find_apriori(apriori, indexes_integration)
    if apriori_f_integration == 0:
        return 0
    return _posterior_item(F, f_posteriori, f_integration, lambdas_d, lambdas_v, apriori_f_posterior, apriori_f_integration)

def _get_indexes(limits_low : np.ndarray, num_indexes : np.ndarray, dl : float):
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
                   indexes_posteriori : np.ndarray,
                   lambdas_d : np.ndarray,
                   lambdas_v : np.ndarray,
                   apriori : np.ndarray,
                   limits_low : np.ndarray,
                   num_indexes : np.ndarray,
                   dl : float) -> float:
    """
        Find p(f_posteriori | {F_i})
    """
    assert lambdas_v.shape[1] == indexes_posteriori.shape[0]
    assert lambdas_d.shape[0] == lambdas_v.shape[0]
    assert lambdas_v.shape[0] == F.shape[0]
    assert limits_low.shape[0] == indexes_posteriori.shape[0]
    assert num_indexes.shape[0] == indexes_posteriori.shape[0]

    f_posteriori : np.ndarray = limits_low + indexes_posteriori * dl

    apriori_f_posterior : float = _find_apriori(apriori, f_posteriori)
    if apriori_f_posterior == 0:
        return 0
    s = 0
    for indexes in _get_indexes(limits_low, num_indexes, dl):
        s += _posterior_item(F, f_posteriori, lambdas_d, lambdas_v, apriori, apriori_f_posterior, indexes, limits_low, dl) * dl**len(indexes)
    return 1/s
