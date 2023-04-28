
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from src.DataHandling import DataEntry, DataGroup

import pytest


def test_dataentry_conversion():
    temp = DataEntry('23 ', 'number')
    stringform = yaml.dump(temp)
    print(stringform)
    next = yaml.load(stringform, Loader=Loader)
    assert temp == next

def test_datagroup_conversion():
    start = DataGroup(label= "dummy")
    start['unit1'] = DataEntry('23 ', 'number')
    stringform = yaml.dump(start)
    print(stringform)
    next = yaml.load(stringform, Loader=Loader)
    assert start == next
