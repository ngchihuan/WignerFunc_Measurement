from shutil import Error
import pytest
import WignerFunctionMeasurement as WFM

fname_1 = '../test_data/r1op bsb delay scan after sbc_processed'
fpath_1 = '../test_data'
fname_nonexist = 'nonexist'

# unprocessed data saved by the iongui program used in Dzmitry's lab.
fname_raw_1 = '../test_data/r1op bsb delay scan after sbc Xcohex 50us phase 0'

class Test_Wigner():
    @pytest.mark.parametrize("fname",[fname_1])
    def test_debugPrint(self,fname):
        assert WFM.print_debug() == 'debug'

    @pytest.mark.parametrize("fpath",[fname_nonexist,1])
    def test_is_wrong_path_caught(self,fpath):
        try:
            sb1 = WFM.WignerFunc_Measurement(fpath)
            sb1.set_path(fpath)
            assert False
        except:
            assert True

    @pytest.mark.parametrize("fpath",[fpath_1,'.'])
    def test_noerror_if_right_path_isgiven(self,fpath):
        try:
            sb1 = WFM.WignerFunc_Measurement(fpath)
            sb1.set_path(fpath)
            assert True
        except:
            assert False
    @pytest.mark.parametrize("fpath",[fpath_1])
    def test_list_allfiles_noerr(self,fpath):
        try:
            sb1 = WFM.WignerFunc_Measurement(fpath)
            sb1.list_all_files()
            assert True
        except:
            assert False

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

    @pytest.mark.parametrize("fname", [fname_raw_1])
    def test_is_rawdata_isread_correct(self,fname):
        try:
            sb1 = WFM.SideBandMeasurement(fname,raw = True)
            assert bool(sb1.xy)
        except:
            assert False


    @pytest.mark.parametrize("fname",[fname_1])
    @pytest.mark.parametrize("parity",[1])
    def test_parity_calculate_correct(self,fname, parity):
        sbm = WFM.SideBandMeasurement(fname)
        res = sbm.eval_parity()
        assert res>0.9

    @pytest.mark.parametrize("fname",[fname_1])
    @pytest.mark.parametrize("weight",[ {"a":1}, 2, ["a", "b"]])
    def test_setweight_bad_input_given(self, fname, weight):
        try:
            sbm = WFM.SideBandMeasurement(fname)
            sbm.set_weight(weight)
            assert False
        except TypeError:
            assert True
        except ValueError:
            assert True
