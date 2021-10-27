import pytest
import numpy as np
import WignerFunctionMeasurement as WFM
from WignerFunctionMeasurement import fit

data_1 = tuple(np.genfromtxt('../test_data/r1op bsb delay scan after sbc_processed'))
data_2 = tuple(np.genfromtxt('../test_data/r1op rsb delay scan after sbc 5r1op_processed'))





class TestClass_fit_sum_multi_sine_offset:

    @pytest.mark.parametrize("x, y, yerr", [ data_1] )
    @pytest.mark.parametrize("weights", [   [1, 0, 0]  , [0 ,1, 0], [0, 0, 1] ] )
    @pytest.mark.parametrize('Omega_0',[ 0.04, 0.05, 0.06])
    @pytest.mark.parametrize('gamma', [1e-4])
    def test_is_fit_results_correct_bsb(self,x,y,yerr, weights, Omega_0, gamma ):

        res = fit.fit_sum_multi_sine_offset(x, y, yerr, weights, Omega_0, gamma, offset=0.0,
                              rsb=False, gamma_fixed=False,
                              customized_bound_population=None, debug=False)
        assert res['weight fit'][0] > 0.95
