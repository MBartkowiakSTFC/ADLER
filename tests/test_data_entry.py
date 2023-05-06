
from src.DataHandling import DataEntry

import pytest


def test_dataentry_creation():
    item = DataEntry("25.0", 'madeup')
    assert item.string == "25.0"
    assert item.data == 25.0

def test_dataentry_addition():
    item1 = DataEntry("25.0", 'madeup')
    item2 = DataEntry("33.0", 'madeup')
    item = item1 + item2
    assert len(item.data) == 2
    assert len(item.string.split()) == 2
