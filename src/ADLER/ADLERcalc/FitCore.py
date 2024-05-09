
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

import numpy as np
import os
import time
from os.path import expanduser
import copy

# import ctypes

from PyQt6.QtCore import  QObject, pyqtSignal, pyqtSlot
from ADLER.ADLERcalc.arrayUtils import merge2curves_errors
from ADLER.ADLERcalc.fitUtils import global_fitting_optimiser, iterative_fitting_optimiser
from ADLER.ADLERcalc.ioUtils import WriteEnergyProfile, read_1D_curve_extended
from ADLER.ADLERcalc.spectrumUtils import place_points_in_bins


class FitCore(QObject):
    cleared = pyqtSignal()
    loaded = pyqtSignal()
    fileparams = pyqtSignal(object)
    logmessage = pyqtSignal(str)
    finished_fitting = pyqtSignal(object)
    finished_overplot = pyqtSignal()
    finished_rixsmap = pyqtSignal()
    finished_merge = pyqtSignal()
    finished_filter = pyqtSignal()
    def __init__(self, master, boxes, logger,  progress_bar = None,  table = None, max_threads = 1, 
                                startpath = expanduser("~")):
        if master is None:
            super().__init__()
        else:
            super().__init__(master)
        self.parlist = []
        self.pardict = {}
        self.maxthreads = max_threads
        self.log = logger
        self.logmessage.connect(self.log.put_logmessage)
        self.progbar = progress_bar
        self.table_obj = table
        self.temp_path = startpath
        self.lastreduction = 1.0
        self.cutoff = 30
        self.binsize = 0.1
        self.separator = ' '
        self.comment = '#'
        self.xcolumn = 0
        self.ycolumn = 1
        self.ecolumn = -1
        self.fullnames = []
        self.orig_logs = []
        self.shortnames = []
        self.timedata = None
        self.prof_numbers = []
        self.mplot_curves = []
        self.mplot_raw_curves = []
        self.mplot_labels =[]
        self.mplot_override = ['X axis',  'Y axis']
        self.mplot_fits = []
        self.mplot_fitparams = []
        self.map2D = None
        self.overplotworked = False
        self.rixs_worked = False
        self.fitsworked = False
        self.mergeworked = False
        self.merged_curve = None
        self.prof_count = 0
        self.legpos = 0
        self.retvals = []
    def assign_boxes(self, boxes):
        self.boxes = boxes
        for db in boxes:
            names, values = db.returnValues()
            self.parlist += names
            for nam in names:
                self.pardict[nam] = values[nam]
            db.values_changed.connect(self.make_params_visible)
        self.make_params_visible()
    @pyqtSlot(object)
    def new_loadparams(self, pdict):
        for kk in pdict.keys():
            kname = str(kk)
            if kk == 'separator':
                self.separator = pdict[kk]
            elif kk == 'comment':
                self.comment = pdict[kk]
            elif kk == 'xcolumn':
                self.xcolumn = pdict[kk]
            elif kk == 'ycolumn':
                self.ycolumn = pdict[kk]
            elif kk == 'ecolumn':
                self.ecolumn = pdict[kk]
    @pyqtSlot()
    def make_params_visible(self):
        for db in self.boxes:
            names, values = db.returnValues()
            for n in names:
                if not n in self.parlist:
                    self.parlist.append(n)
                self.pardict[n] = values[n]
        for k in self.parlist:
            val = self.pardict[k]
            if len(val) == 1:
                val = val[0]
            if 'cuts' in k:
                self.cuts = val
            elif 'eline' in k:
                self.eline = val
            elif 'legpos' in k:
                self.legpos = val
            elif 'offmax' in k:
                self.offmax = val
            elif 'redfac' in k:
                self.redfac = val
            elif 'maxpeaks' in k:
                self.maxpeaks = val+1
            elif 'polyorder' in k:
                self.polyorder = val
            elif 'fixedL' in k:
                self.fixedLwidth = val
            elif 'fixedG' in k:
                self.fixedGwidth = val
            elif 'useedge' in k:
                self.useedge = val
            elif 'penalty' in k:
                self.penalty = val
    def reduce_profiles(self):
        if (len(self.tey_profiles) == len(self.reduced_tey_profiles) and
            len(self.tpy_profiles) == len(self.reduced_tpy_profiles) and
            len(self.raw_tey_profiles) == len(self.reduced_raw_tey_profiles) and
            len(self.raw_tpy_profiles) == len(self.reduced_raw_tpy_profiles)):
            if self.redfac == self.lastreduction:
                return False
        self.reduced_tey_profiles = []
        self.reduced_tpy_profiles = []
        self.reduced_raw_tey_profiles = []
        self.reduced_raw_tpy_profiles = []
        if self.redfac == 1.0:
            for p in self.tey_profiles:
                self.reduced_tey_profiles.append(p)
            for p in self.tpy_profiles:
                self.reduced_tpy_profiles.append(p)
            for p in self.raw_tey_profiles:
                self.reduced_raw_tey_profiles.append(p)
            for p in self.raw_tpy_profiles:
                self.reduced_raw_tpy_profiles.append(p)
        else:
            for num, p in enumerate(self.tey_profiles):
                if self.orig_logs[num] is None:
                    steps = len(p)
                    newsteps = int(round(steps/self.redfac))
                    target = np.zeros([newsteps, 3])
                    target[:, 0] = np.linspace(p[:, 0].min(), p[:, 0].max(), newsteps)
                    newone = merge2curves_errors(p,  target)
                    newone[:, 1:] /= self.redfac
                    self.reduced_tey_profiles.append(newone)
                else:
                    avglog, errlog = place_points_in_bins(copy.deepcopy(self.orig_logs[num]),
                                                                           redfac = self.redfac)
                    newone = np.column_stack([avglog['E'], avglog['CURR2'], errlog['CURR2']])
                    self.reduced_tey_profiles.append(newone)
            for num, p in enumerate(self.tpy_profiles):
                if self.orig_logs[num] is None:
                    steps = len(p)
                    newsteps = int(round(steps/self.redfac))
                    target = np.zeros([newsteps, 3])
                    target[:, 0] = np.linspace(p[:, 0].min(), p[:, 0].max(), newsteps)
                    newone = merge2curves_errors(p,  target)
                    newone[:, 1:] /= self.redfac
                    self.reduced_tpy_profiles.append(newone)
                else:
                    avglog, errlog = place_points_in_bins(copy.deepcopy(self.orig_logs[num]),
                                                                            redfac = self.redfac)
                    newone = np.column_stack([avglog['E'], avglog['CURR1'], errlog['CURR1']])
                    self.reduced_tpy_profiles.append(newone)
            for p in self.raw_tey_profiles:
                self.reduced_raw_tey_profiles.append(p)
            for p in self.raw_tpy_profiles:
                self.reduced_raw_tpy_profiles.append(p)
        self.lastreduction = self.redfac
        return True
    @pyqtSlot()
    def take_table_values(self):
        self.retvals = self.table_obj.return_values()
    def logger(self,  message):
        now = time.gmtime()
        timestamp = ("ProfileCore "
                     +"-".join([str(tx) for tx in [now.tm_mday, now.tm_mon, now.tm_year]])
                     + ',' + ":".join([str(ty) for ty in [now.tm_hour, now.tm_min, now.tm_sec]]) + '| ')
        self.logmessage.emit(timestamp + message)
    @pyqtSlot(object)
    def load_profiles(self,  flist):
        fnames, snames, profiles,  envals,  units, pardicts = [], [], [], [], [], []
        for fnum, fname in enumerate([flist]):
            self.temp_path, short_name = os.path.split(fname)
            try:
                p, e, u, d = read_1D_curve_extended(fname, xcol = self.xcolumn, ycol = self.ycolumn, ecol = self.ecolumn, 
                                                          comment = self.comment, sep = self.separator)
            except:
                self.logger("Could not parse file:" + str(fname))
            else:
                self.prof_numbers.append(self.prof_count+fnum)
                fnames.append(fname)
                profiles.append(p)
                envals.append(e[0])
                units.append(u)
                pardicts.append(d)
                snames.append(short_name)
        self.fullname = fnames[0]
        self.shortname = snames[0]
        self.profile = profiles[0]
        self.prof_count = 1
        self.loaded.emit()
        self.fileparams.emit([self.fullname, self.shortname,  self.profile])
        return self.fullname, self.shortname, self.profile
    @pyqtSlot()
    def clear_profiles(self):
        self.fullname = ""
        self.shortname = ""
        self.profile = []
        self.prof_count = 0
        self.cleared.emit()
    @pyqtSlot(str)
    def save_fit_results(self, fname):
        if self.merged_curve is None:
            self.logger("There is no merged curve to be saved.")
            return None
        else:
            WriteEnergyProfile(fname, self.merged_curve, [])
            return True
            # outname = os.path.join(self.temp_path, self.temp_name+"_HISTOGRAM.txt")
            # WriteEnergyProfile(outname, target, [])
    @pyqtSlot()
    def sequential_fit(self):
        self.fitsworked = False
        if self.profile is []:
            return None
        else:
            temp = self.profile.copy()
            lowlim,  hilim = self.cuts
            temp = temp[np.where(np.logical_and(temp[:, 0] >= lowlim, temp[:, 0]<=hilim))]
            edge = self.useedge > 0.5
            b_order = self.polyorder
            mpeaks = self.maxpeaks
            gauss_fwhm = self.fixedGwidth
            lorentz_fwhm = self.fixedLwidth
            pfactor = self.penalty
            self.logger("Preparing the sequential fit")
            self.logger("edge, b-order, maxpeaks, gauss-width, lorentz-width, penalty")
            self.logger(", ".join([str(x) for x in [edge, b_order, mpeaks, gauss_fwhm, lorentz_fwhm, pfactor]]))
            results,  summary = iterative_fitting_optimiser(temp, maxpeaks = mpeaks, bkg_order = b_order, include_edge = edge,
                                                    one_gauss_fwhm = gauss_fwhm,  one_lorentz_fwhm = lorentz_fwhm, 
                                                    overshoot_penalty = pfactor)
            self.fit_results = results
            self.fit_summary = summary
            self.finished_fitting.emit(results)
            self.fitsworked = True
    @pyqtSlot()
    def global_fit(self):
        self.fitsworked = False
        if self.profile is []:
            return None
        else:
            temp = self.profile.copy()
            lowlim,  hilim = self.cuts
            temp = temp[np.where(np.logical_and(temp[:, 0] >= lowlim, temp[:, 0]<=hilim))]
            edge = self.useedge > 0.5
            b_order = self.polyorder
            mpeaks = self.maxpeaks
            gauss_fwhm = self.fixedGwidth
            lorentz_fwhm = self.fixedLwidth
            pfactor = self.penalty
            self.logger("Preparing the global fit")
            self.logger("edge, b-order, maxpeaks, gauss-width, lorentz-width, penalty")
            self.logger(", ".join([str(x) for x in [edge, b_order, mpeaks, gauss_fwhm, lorentz_fwhm, pfactor]]))
            results,  summary = global_fitting_optimiser(temp, maxpeaks = mpeaks, bkg_order = b_order, include_edge = edge,
                                                    one_gauss_fwhm = gauss_fwhm,  one_lorentz_fwhm = lorentz_fwhm, 
                                                    overshoot_penalty = pfactor)
            self.fit_results = results
            self.fit_summary = summary
            self.finished_fitting.emit(results)
            self.fitsworked = True
    @pyqtSlot()
    def return_fitpars(self):
        pass
