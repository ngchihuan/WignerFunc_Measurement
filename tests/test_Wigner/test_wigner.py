import pytest
import WignerFunctionMeasurement as WFM

class Test_Wigner():
    def test_debugPrint(self):
        assert WFM.print_debug() == 'debug2'