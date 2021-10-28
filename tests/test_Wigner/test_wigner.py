from shutil import Error
import pytest
import WignerFunctionMeasurement as WFM

fname_1 = '../test_data/r1op bsb delay scan after sbc_processed'

fname_nonexist = 'nonexist'
class Test_Wigner():
    @pytest.mark.parametrize("fname",[fname_1])
    def test_debugPrint(self,fname):
        assert WFM.print_debug() == 'debug'


class Test_SideBandMeasurement():
    @pytest.mark.parametrize("fname",[fname_nonexist])
    def test_is_nonfoundfile_raised(self,fname):        
        try:
            WFM.SideBandMeasurement(fname)
            assert False
        except OSError:
            assert True

    
    @pytest.mark.parametrize("fname",[fname_1])
    def test_is_nonerror_file_exist(self, fname):
        try:
            WFM.SideBandMeasurement(fname)
            assert True
        except OSError:
            assert False

    @pytest.mark.parametrize("fname",[fname_1])
    @pytest.mark.parametrize("parity",[1])
    def test_parity_calculate_correct(self,fname, parity):
        sbm = WFM.SideBandMeasurement(fname)
        res = sbm.eval_parity()
        assert res>.9


    