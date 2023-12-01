from ADLER.ADLERdata.AdlerData import DATA

import pytest


def test_number_of_files():
    print(DATA.res_dir.absolutePath())
    assert len(DATA._files) == 3
