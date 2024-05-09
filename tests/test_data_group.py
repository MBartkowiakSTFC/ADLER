
from src.ADLERcalc.DataHandling import DataEntry, DataGroup

import pytest


def test_datagroup_noname():
    group1 = DataGroup()
    group2 = DataGroup()
    assert group1.label == "UnnamedGroup1"
    assert group2.label == "UnnamedGroup2"

def test_datagroup_wrongkey():
    group = DataGroup()
    item = group['NoSuchKey']
    assert item.label == ""

def test_datagroup_rightkey():
    group = DataGroup(label='Stinker')
    group['RealKey'] = DataEntry('23 ', 'number')
    item = group['RealKey']
    assert item.label == "number"
    assert item.string == "23.0"

def test_datagroup_init_elements():
    group = DataGroup(label='Stinker', elements = [(23, 30), ('semi', 'skimmed')])
    item = group[23]
    assert item == 30
