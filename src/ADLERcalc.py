
#    This file is part of ADLER.
#
#    ADLER is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# 
# Copyright (C) Maciej Bartkowiak, 2019-2023

__doc__ = """
The part of the ADLER code responsible for
the handling of the files and processing the data.
"""

import math
import numpy as np
import os
import time
import sys
import gzip
import h5py
from os.path import expanduser
import copy
from collections import defaultdict
from numba import jit, prange
from scipy.sparse import csc_array

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

has_voigt = True
try:
    from scipy.special import voigt_profile
except:
    from scipy.special import wofz
    has_voigt = False
from scipy.optimize import leastsq, shgo, minimize
from scipy.interpolate import interp1d
from scipy.fftpack import rfft, irfft, fftfreq
from astropy.io import fits as fits_module

# import ctypes

from PyQt6.QtCore import QThread, QObject, pyqtSignal, pyqtSlot, QMutex, QDate, QTime
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import  QApplication
from ExtendGUI import CustomThreadpool
from DataHandling import DataEntry, DataGroup, RixsMeasurement







