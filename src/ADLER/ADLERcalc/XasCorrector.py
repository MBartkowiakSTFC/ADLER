
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
applying the spectral normalisation to the XAS spectra.
"""

import numpy as np
from scipy.interpolate import interp1d

from PyQt6.QtCore import  QObject, pyqtSignal, pyqtSlot

from ADLER.ADLERcalc.ioUtils import read_fluxcorr_curves

class XasCorrector(QObject):
    def __init__(self, parent = None):
        if parent is None:
            super().__init__()
        else:
            super().__init__(parent)
        self.curves_harm1 = [] # the actual data in 2 columns
        self.curves_harm3 = []
        self.curves_harm5 = []
        self.curves_harm7 = []
        self.total_harm1 = [] # combined curves for each harmonics
        self.total_harm3 = []
        self.total_harm5 = []
        self.total_harm7 = []
        self.limits_harm1 = [0, 0] # the total limits of data available in our referece file for this harmonics
        self.limits_harm3 = [0, 0]
        self.limits_harm5 = [0, 0]
        self.limits_harm7 = [0, 0]
        self.inner_limits_harm1 = [] # the speific limits of each curve
        self.inner_limits_harm3 = []
        self.inner_limits_harm5 = []
        self.inner_limits_harm7 = []
        self.limits_need_updating = [True, True, True, True]
    def readCurves(self, fname, harmnum = 1):
        rcurves = read_fluxcorr_curves(fname)
        if harmnum == 1:
            self.curves_harm1 = rcurves
            temp = np.row_stack(rcurves)
            sorting = np.argsort(temp[:, 0])
            self.total_harm1 = temp[sorting]
            self.limits_need_updating[0] = True
        elif harmnum == 3:
            self.curves_harm3 = rcurves
            temp = np.row_stack(rcurves)
            sorting = np.argsort(temp[:, 0])
            self.total_harm3 = temp[sorting]
            self.limits_need_updating[1] = True
        elif harmnum == 5:
            self.curves_harm5 = rcurves
            temp = np.row_stack(rcurves)
            sorting = np.argsort(temp[:, 0])
            self.total_harm5 = temp[sorting]
            self.limits_need_updating[2] = True
        elif harmnum == 7:
            self.curves_harm7 = rcurves
            temp = np.row_stack(rcurves)
            sorting = np.argsort(temp[:, 0])
            self.total_harm7 = temp[sorting]
            self.limits_need_updating[3] = True
    def checkLimits(self):
        for nc in range(4):
            if self.limits_need_updating[nc]:
                if nc == 0:
                    allcurves = self.curves_harm1
                elif nc == 1:
                    allcurves = self.curves_harm3
                elif nc == 2:
                    allcurves = self.curves_harm5
                elif nc == 3:
                    allcurves = self.curves_harm7
                temp = []
                abstemp = [1200.0, 0.0]
                for cur in allcurves:
                    temp.append((cur[:, 0].min(), cur[:, 0].max()))
                    maxx = max(abstemp[1], cur[:, 0].max())
                    minx = min(abstemp[0],  cur[:, 0].min())
                    abstemp = [minx, maxx]
                if nc == 0:
                    self.limits_harm1 = abstemp
                    self.inner_limits_harm1 = temp
                elif nc == 1:
                    self.limits_harm3 = abstemp
                    self.inner_limits_harm3 = temp
                elif nc == 2:
                    self.limits_harm5 = abstemp
                    self.inner_limits_harm5 = temp
                elif nc == 3:
                    self.limits_harm7 = abstemp
                    self.inner_limits_harm7 = temp
                self.limits_need_updating[nc] = False
    def returnInterpolatedCurve(self, xarray, kind_covered = 'linear', kind_notcovered = 'cubic'):
        """
        The input should be just the x array of the data we are trying to match.
        The returned curve will be the flux correction as a function of photon energy.
        The interpolation method will be different, depending on where the x values are
        compared to the measured reference data.
        """
        self.checkLimits()
        inarr = xarray[np.argsort(xarray)]
        match = [0.0, 0.0, 0.0, 0.0]
        for nh in range(4):
            if nh == 0:
                abslims = self.limits_harm1
                seplims = self.inner_limits_harm1
            elif nh == 1:
                abslims = self.limits_harm3
                seplims = self.inner_limits_harm3
            elif nh == 2:
                abslims = self.limits_harm5
                seplims = self.inner_limits_harm5
            elif nh == 3:
                abslims = self.limits_harm7
                seplims = self.inner_limits_harm7
            match_deg = np.logical_and(inarr >= abslims[0], inarr <= abslims[1]).sum() / len(inarr)
            match[nh] = match_deg # we check the overlap between the x range of the input data and reference data
        match = np.array(match)
        if np.all(match == 0.0):
            print("None of the data ranges match the input data range. Flux correction is not possible.")
            return None
        harmnum = np.argmax(match)
        if harmnum == 0:
            abslims = self.limits_harm1
            seplims = self.inner_limits_harm1
            total = self.total_harm1
            curves = self.curves_harm1
        elif harmnum == 1:
            abslims = self.limits_harm3
            seplims = self.inner_limits_harm3
            total = self.total_harm3
            curves = self.curves_harm3
        elif harmnum == 2:
            abslims = self.limits_harm5
            seplims = self.inner_limits_harm5
            total = self.total_harm5
            curves = self.curves_harm5
        elif harmnum == 3:
            abslims = self.limits_harm7
            seplims = self.inner_limits_harm7
            total = self.total_harm7
            curves = self.curves_harm7
        # interpolation kinds
        # 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next'
        newcurve = np.zeros((len(xarray), 2))
        newcurve[:, 0] = xarray
        newcurve[:, 1] = interp1d(total[:, 0],  total[:, 1], kind = kind_notcovered,  bounds_error=False, fill_value = 0.0)(xarray)
        for nc, curve in enumerate(curves):
            xref = curve[:, 0]
            xmin,  xmax = xref.min(), xref.max()
            criterion = np.logical_and(xarray >= xmin, xarray <= xmax)
            if not np.any(criterion):
                continue # let's not waste time on the curves out of the range
            tempx = xarray[np.where(criterion)]
            newy = interp1d(curve[:, 0], curve[:, 1],  kind = kind_covered,  bounds_error=False, fill_value = 0.0)(tempx)
            newcurve[:, 1][np.where(criterion)] = newy
        return newcurve
