import os
import tempfile

import pytest

from src.ADLERdata.AdlerData import DATA
from src.ADLERcalc.RixsMeasurement import RixsMeasurement

file_wd = os.path.dirname(os.path.realpath(__file__))

file1 = os.path.join(file_wd, "data", "VI3grazing-65K_R0010.sif")
file2 = os.path.join(file_wd, "data", "VI3grazing-65K_R0011.sif")

def test_load_one_measurement():
    rixs_instance1 = RixsMeasurement([file1])
    assert abs(rixs_instance1.energy - 516.0) < 1e-10

def test_load_and_merge_two():
    rixs_instance = RixsMeasurement([file1, file2])
    assert len(rixs_instance.data_files) == 2
    assert len(rixs_instance.log_files) == 2
    assert len(rixs_instance.header_files) == 2

def test_merge_and_write_out():
    temp_name = tempfile.mktemp()
    rixs_instance = RixsMeasurement([file1, file2])
    rixs_instance.write_extended_ADLER(temp_name)
    assert os.path.exists(temp_name)
    assert os.path.isfile(temp_name)
    os.remove(temp_name)
