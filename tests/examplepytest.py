import pytest

@pytest.fixture(scope="session")
def empty_array():
    return []

@pytest.fixture
def array_add(empty_array):
    empty_array.append(1)
    return empty_array

@pytest.fixture
def array_add_2(empty_array):
    empty_array.append(2)
    return empty_array


class TestClass1:
    def test1(self,array_add):
        assert array_add[0]==1
    
    def test2(self,array_add_2):
        assert array_add_2[0]==2
