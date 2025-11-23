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
import vstarstack.library.bayes
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
