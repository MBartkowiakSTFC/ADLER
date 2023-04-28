
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


class XasCore(QObject):
    cleared = pyqtSignal()
    loaded = pyqtSignal()
    fileparams = pyqtSignal(object)
    logmessage = pyqtSignal(str)
    finished_fitting = pyqtSignal()
    finished_overplot = pyqtSignal()
    finished_rixsmap = pyqtSignal()
    finished_merge = pyqtSignal()
    finished_filter = pyqtSignal()
    finished_flux = pyqtSignal()
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
        self.current_interp = 'linear'
        self.interp_kinds = ['linear','slinear', 'quadratic', 'cubic', 'zero' ]
        self.xguess = []
        self.tey_profiles = []
        self.tpy_profiles = []
        self.raw_tey_profiles = []
        self.raw_tpy_profiles = []
        self.reduced_tey_profiles = []
        self.reduced_tpy_profiles = []
        self.reduced_raw_tey_profiles = []
        self.reduced_raw_tpy_profiles = []
        self.lastreduction = 1.0
        self.cutoff = 30
        self.binsize = 0.1
        self.fullnames = []
        self.orig_logs = []
        self.shortnames = []
        self.timedata = None
        self.prof_numbers = []
        self.mplot_curves = []
        self.mplot_raw_curves = []
        self.mplot_labels =[]
        self.mplot_override = []
        self.mplot_fits = []
        self.mplot_fitparams = []
        self.map2D = None
        self.overplotworked = False
        self.rixs_worked = False
        self.fitsworked = False
        self.mergeworked = False
        self.merged_curve = None
        self.filter_curves = []
        self.filter_labels = []
        self.flux_curves = []
        self.flux_labels = []
        self.filter_override = ["Inverse units", "Fourier Transform"]
        self.map2Dplotax = [(1, 2048,),  (1, 2048)]
        self.prof_count = 0
        self.legpos = 0
        self.retvals = []
        xas = XasCorrector()
        xas.readCurves(resource_path('FluxCorrectionHarm1.txt'), harmnum=1)
        xas.readCurves(resource_path('FluxCorrectionHarm3.txt'), harmnum=3)
        self.xas_corrector = xas
    def assign_boxes(self, boxes):
        self.boxes = boxes
        for db in boxes:
            names, values = db.returnValues()
            self.parlist += names
            for nam in names:
                self.pardict[nam] = values[nam]
            db.values_changed.connect(self.make_params_visible)
        self.make_params_visible()
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
            elif 'bkg_perc' in k:
                self.bkg_perc = val
            elif 'offmax' in k:
                self.offmax = val
            elif 'redfac' in k:
                self.redfac = val
            elif 'legpos' in k:
                self.legpos = val
            elif 'smear' in k:
                self.smear = val
            elif 'binsize' in k:
                self.binsize = val
            elif 'cutoff' in k:
                self.cutoff = val
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
            for nump, p in enumerate(self.tey_profiles):
                if len(p) == 0:
                    self.reduced_tey_profiles.append(self.raw_tey_profiles[nump])
                else:
                    self.reduced_tey_profiles.append(p)
            for nump, p in enumerate(self.tpy_profiles):
                if len(p) == 0:
                    self.reduced_tpy_profiles.append(self.raw_tpy_profiles[nump])
                else:
                    self.reduced_tpy_profiles.append(p)
            for p in self.raw_tey_profiles:
                self.reduced_raw_tey_profiles.append(p)
            for p in self.raw_tpy_profiles:
                self.reduced_raw_tpy_profiles.append(p)
        else:
            for num, p in enumerate(self.tey_profiles):
                xax = self.xguess[num]
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
                    if len(avglog[xax]) > 0:
                        newone = np.column_stack([avglog[xax], avglog['CURR2'], errlog['CURR2']])
                    else:
                        newone = np.column_stack([avglog['TARGET'], avglog['CURR2'], errlog['CURR2']])
                    self.reduced_tey_profiles.append(newone)
            for num, p in enumerate(self.tpy_profiles):
                xax = self.xguess[num]
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
                    if len(avglog[xax]) > 0:
                        newone = np.column_stack([avglog[xax], avglog['CURR1'], errlog['CURR1']])
                    else:
                        newone = np.column_stack([avglog['TARGET'], avglog['CURR2'], errlog['CURR2']])
                    self.reduced_tpy_profiles.append(newone)
            for p in self.raw_tey_profiles:
                self.reduced_raw_tey_profiles.append(p)
            for p in self.raw_tpy_profiles:
                self.reduced_raw_tpy_profiles.append(p)
        self.lastreduction = self.redfac
        return True
    @pyqtSlot(int)
    def setInterpolation(self, newnum):
        self.current_interp = self.interp_kinds[newnum]
        # self.rixs_axis_label = self.rixsaxes[newnum]
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
        fnames, snames, tey_profiles, tpy_profiles, raw_tey_profiles, raw_tpy_profiles = [], [], [], [], [], []
        original_logs = []
        xguesses = []
        for fnum, fname in enumerate(flist):
            self.temp_path, short_name = os.path.split(fname)
            if 'txt' in short_name[-4:]:
                try:
                    envals, tey, tpy = read_1D_xas(fname)
                except:
                    self.logger("Could not parse file:" + str(fname))
                    continue
                else:
                    self.prof_numbers.append(self.prof_count+fnum)
                    fnames.append(fname)
                    xguesses.append('E')
                    snames.append(short_name)
                    tey_profiles.append(np.column_stack([envals, tey, np.zeros(envals.shape)]))
                    tpy_profiles.append(np.column_stack([envals, tpy, np.zeros(envals.shape)]))
                    raw_tey_profiles.append(np.column_stack([envals, tey, np.zeros(envals.shape)]))
                    raw_tpy_profiles.append(np.column_stack([envals, tpy, np.zeros(envals.shape)]))
                    original_logs.append(None)
            else:
                try:
                    varlog = load_only_logs([fname])
                    # xguess = guess_XAS_xaxis(varlog)
                    avglog, errlog, xguess = load_and_average_logs([fname])
                except:
                    self.logger("Could not parse file:" + str(fname))
                else:
                    envals = varlog[xguess]
                    xguesses.append(xguess)
                    self.prof_numbers.append(self.prof_count+fnum)
                    fnames.append(fname)
                    snames.append(short_name)
                    if len(avglog[xguess]) > 0:
                        tey_profiles.append(np.column_stack([avglog[xguess], avglog['CURR2'], errlog['CURR2']]))
                        tpy_profiles.append(np.column_stack([avglog[xguess], avglog['CURR1'], errlog['CURR1']]))
                    else:
                        tey_profiles.append(np.column_stack([avglog['TARGET'], avglog['CURR2'], errlog['CURR2']]))
                        tpy_profiles.append(np.column_stack([avglog['TARGET'], avglog['CURR1'], errlog['CURR1']]))
                    raw_tey_profiles.append(np.column_stack([varlog[xguess],varlog['CURR2'], np.zeros(envals.shape)]))
                    raw_tpy_profiles.append(np.column_stack([varlog[xguess], varlog['CURR1'], np.zeros(envals.shape)]))
                    original_logs.append(varlog)
        self.fullnames += fnames
        self.xguess += xguesses
        self.shortnames += snames
        self.tey_profiles += tey_profiles
        self.tpy_profiles += tpy_profiles
        self.raw_tey_profiles += raw_tey_profiles
        self.raw_tpy_profiles += raw_tpy_profiles
        self.prof_count += len(fnames)
        self.orig_logs += original_logs
        self.loaded.emit()
        self.fileparams.emit([snames, tey_profiles, tpy_profiles, raw_tey_profiles, raw_tpy_profiles])
        return snames, tey_profiles, tpy_profiles, raw_tey_profiles, raw_tpy_profiles
    @pyqtSlot()
    def clear_profiles(self):
        self.fullnames = []
        self.shortnames = []
        self.tey_profiles = []
        self.tpy_profiles = []
        self.raw_tey_profiles = []
        self.raw_tpy_profiles = []
        self.orig_logs = []
        self.xguess = []
        self.reduced_tey_profiles = []
        self.reduced_tpy_profiles = []
        self.reduced_raw_tey_profiles = []
        self.reduced_raw_tpy_profiles = []
        self.timedata = None
        self.prof_numbers = []
        self.orig_logs = []
        self.prof_count = 0
        self.cleared.emit()  
    def manual_merge(self):
        self.mergeworked = False
        if len(self.retvals) < 1:
            self.finished_merge.emit()
            return None
        else:
            self.reduce_profiles()
            nums, xmin, xmax,  usetey, usetpy, names = [], [], [], [], [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                xmin.append(self.retvals[nr][1])
                xmax.append(self.retvals[nr][2])
                usetey.append(self.retvals[nr][3])
                usetpy.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
            nums =np.array(nums)
            xmin =np.array(xmin)
            xmax =np.array(xmax)
            othermin,  othermax = np.min(self.cuts),  np.max(self.cuts)
            fwhm_guess = 0.05
            minx, maxx, stepx = 1e5,-1e5,-1.0    
            curves,  curves2 = [],  []
            rcurves,  rcurves2 = [], []
            labels = []
            for rn, num in enumerate(nums):
                if not usetey[rn]:
                    continue
                dat = self.reduced_tey_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                dat = dat[np.where(dat[:, 0] > tempmin)]
                dat = dat[np.where(dat[:, 0] < tempmax)]
                minx = min(minx, dat[:,0].min())
                maxx = max(maxx, dat[:,0].max())
                step = (dat[1:,0] - dat[:-1,0]).mean()
                stepx = max(stepx, step)
                curves.append(dat)
                rcurves.append(self.reduced_raw_tey_profiles[num])
            for rn, num in enumerate(nums):
                if not usetpy[rn]:
                    continue
                dat = self.reduced_tpy_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                dat = dat[np.where(dat[:, 0] > tempmin)]
                dat = dat[np.where(dat[:, 0] < tempmax)]
                minx = min(minx, dat[:,0].min())
                maxx = max(maxx, dat[:,0].max())
                step = (dat[1:,0] - dat[:-1,0]).mean()
                stepx = max(stepx, step)
                curves2.append(dat)
                rcurves2.append(self.reduced_raw_tpy_profiles[num])
            newx = np.arange(minx, maxx + 0.1*stepx, stepx)
            target = np.zeros((len(newx),3))
            target[:,0] = newx
            for n in range(len(curves)):
                target = merge2curves_errors(curves[n],target)
            target2 = np.zeros((len(newx),3))
            target2[:,0] = newx
            for n in range(len(curves2)):
                target2 = merge2curves_errors(curves2[n],target2)
            self.merged_curve = target
            raw1 = np.row_stack(rcurves)
            raw2 = np.row_stack(rcurves2)
            raw1 = raw1[np.argsort(raw1[:, 0])]
            raw2 = raw2[np.argsort(raw2[:, 0])]
            self.mergeworked = True
            # now we try to add the fitting            
            self.fullnames += ["No file"]
            self.shortnames += ["Merged data"]
            self.tey_profiles += [target]
            self.tpy_profiles += [target2]
            self.raw_tey_profiles += [raw1]
            self.raw_tpy_profiles += [raw2]
            self.prof_count += 1
            self.finished_merge.emit()
            # self.finished_fitting.emit()
            return target
    @pyqtSlot(str)
    def save_merged_profile(self, fname):
        if self.merged_curve is None:
            self.logger("There is no merged curve to be saved.")
            return None
        else:
            WriteEnergyProfile(fname, self.merged_curve, [])
            return True
            # outname = os.path.join(self.temp_path, self.temp_name+"_HISTOGRAM.txt")
            # WriteEnergyProfile(outname, target, [])
    @pyqtSlot(str)
    def save_ticked_profiles(self, fpath):
        self.reduce_profiles()
        nums, xmin, xmax, usetey, usetpy, names = [], [], [], [], [], []
        for nr in range(len(self.retvals)):
            nums.append(self.retvals[nr][0])
            xmin.append(self.retvals[nr][1])
            xmax.append(self.retvals[nr][2])
            usetey.append(self.retvals[nr][3])
            usetpy.append(self.retvals[nr][4])
            names.append(self.retvals[nr][5])
        nums =np.array(nums)
        xmin =np.array(xmin)
        xmax =np.array(xmax)
        othermin,  othermax = np.min(self.cuts),  np.max(self.cuts)
        curves = []
        labels = []   
        for rn, num in enumerate(nums):
            if not usetey[rn]:
                continue
            temp = self.reduced_tey_profiles[num].copy()
            tempmin = max(othermin, xmin[rn])
            tempmax = min(othermax, xmax[rn])
            temp = temp[np.where(temp[:, 0] > tempmin)]
            temp = temp[np.where(temp[:, 0] < tempmax)]
            curves.append(temp)
            labels.append('Saved_TEY_from_' + names[rn])  
        for rn, num in enumerate(nums):
            if not usetpy[rn]:
                continue
            temp = self.reduced_tpy_profiles[num].copy()
            tempmin = max(othermin, xmin[rn])
            tempmax = min(othermax, xmax[rn])
            temp = temp[np.where(temp[:, 0] > tempmin)]
            temp = temp[np.where(temp[:, 0] < tempmax)]
            curves.append(temp)
            labels.append('Saved_TPY_from_' + names[rn])    
        if len(curves) ==0:
            self.logger("There are no curves to be saved.")
            return None
        else:
            for num in range(len(curves)):
                if labels[num][-4:] == '.txt':
                    target = fpath + "/" + labels[num]
                else:
                    target = fpath + "/" + labels[num] + '.txt'
                WriteEnergyProfile(target, curves[num], [])
            return True
            # outname = os.path.join(self.temp_path, self.temp_name+"_HISTOGRAM.txt")
            # WriteEnergyProfile(outname, target, [])
    def fft_curves(self):
        self.overplotworked = False
        if len(self.retvals) < 1:
            self.finished_overplot.emit()
            return None
        else:
            self.reduce_profiles()
            nums, xmin, xmax, usetey, usetpy, names = [], [], [], [], [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                xmin.append(self.retvals[nr][1])
                xmax.append(self.retvals[nr][2])
                usetey.append(self.retvals[nr][3])
                usetpy.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
            nums =np.array(nums)
            xmin =np.array(xmin)
            xmax =np.array(xmax)
            othermin,  othermax = np.min(self.cuts),  np.max(self.cuts)
            curves = []
            labels = []   
            for rn, num in enumerate(nums):
                if not usetey[rn]:
                    continue
                temp = self.reduced_tey_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TEY from ' + names[rn])         
            for rn, num in enumerate(nums):
                if not usetpy[rn]:
                    continue
                temp = self.reduced_tpy_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TPY from ' + names[rn])
            for n in range(len(curves)):
                xaxis = curves[n][:,0]
                xstep = (xaxis[1:] - xaxis[:-1]).mean()
                # new_x = fftfreq(len(xaxis), xstep)
                new_y = rfft(curves[n][:, 1])
                new_x = np.arange(len(new_y)) + 1
                # norm = curves[n][:,1].sum()
                curves[n] = np.column_stack([new_x, new_y, np.zeros(len(new_x))])
                # curves[n] = curves[n][np.where(curves[n][:,1] > 0.0)]
                # curves[n][:,1] *= 100.0
            self.mplot_curves = curves
            self.mplot_labels = labels
            self.mplot_override = ["Inverse units", "Fourier Transform"]
            self.mplot_raw_curves = []
            self.overplotworked = True
            self.finished_overplot.emit()
            return "Done"
    def fft_filter(self):
        self.filter_curves = []
        self.filter_labels = []
        self.filter_units = []
        self.filter_energies = []
        self.overplotworked = False
        if len(self.retvals) < 1:
            self.finished_filter.emit()
            return None
        else:
            self.reduce_profiles()
            nums, xmin, xmax, usetey, usetpy, names = [], [], [], [], [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                xmin.append(self.retvals[nr][1])
                xmax.append(self.retvals[nr][2])
                usetey.append(self.retvals[nr][3])
                usetpy.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
            nums =np.array(nums)
            xmin =np.array(xmin)
            xmax =np.array(xmax)
            othermin,  othermax = np.min(self.cuts),  np.max(self.cuts)
            curves = []
            labels = []
            tags = []
            teylist = []
            tpylist = []
            taggeddict = {}
            firsttpy = 0
            for rn, num in enumerate(nums):
                if not usetey[rn]:
                    continue
                temp = self.reduced_tey_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TEY from ' + names[rn])      
                teylist.append(names[rn])
                tags.append(names[rn])
                taggeddict[names[rn]] = [None, None]
                firsttpy = rn+1
            for rn, num in enumerate(nums):
                if not usetpy[rn]:
                    continue
                temp = self.reduced_tpy_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TPY from ' + names[rn])
                tpylist.append(names[rn])
                tags.append(names[rn])
                taggeddict[names[rn]] = [None, None]
            for n in range(len(curves)):
                xaxis = curves[n][:,0]
                xstep = (xaxis[1:] - xaxis[:-1]).mean()
                new_x = fftfreq(len(xaxis), xstep)
                new_y = rfft(curves[n][:, 1])
                new_y[-self.cutoff:] = 0.0
                result = irfft(new_y)
                # norm = curves[n][:,1].sum()
                curves[n] = np.column_stack([xaxis, result,  np.zeros(len(xaxis))])
                labels[n] = 'Filtered_' + labels[n]
                if n < firsttpy:
                    taggeddict[tags[n]][0] = np.column_stack([xaxis, result,  np.zeros(len(xaxis))])
                else:
                    taggeddict[tags[n]][1] = np.column_stack([xaxis, result,  np.zeros(len(xaxis))])
                # curves[n] = curves[n][np.where(curves[n][:,1] > 0.0)]
                # curves[n][:,1] *= 100.0
            self.filter_curves = []
            self.filter_labels = []
            self.filter_units = []
            self.filter_units.append("Energy [eV]")
            for tag in np.unique(tags):
                # tag = tags[n]
                print("Filtering: tag ", tag)
                self.fullnames += ["No file"]
                self.shortnames += ['Filtered '+tag]
                teyprof = taggeddict[tag][0]
                tpyprof = taggeddict[tag][1]
                if teyprof is not None:
                    self.tey_profiles += [teyprof]
                    self.raw_tey_profiles += [teyprof]
                else:
                    xax = tpyprof[:, 0]
                    self.tey_profiles += [np.column_stack([xax, np.zeros((len(xax), 2))])]
                    self.raw_tey_profiles += [np.column_stack([xax, np.zeros((len(xax), 2))])]
                if tpyprof is not None:
                    self.tpy_profiles += [tpyprof]
                    self.raw_tpy_profiles += [tpyprof]
                else:
                    xax = teyprof[:, 0]
                    self.tpy_profiles += [np.column_stack([xax, np.zeros((len(xax), 2))])]
                    self.raw_tpy_profiles += [np.column_stack([xax, np.zeros((len(xax), 2))])]
                self.filter_curves.append([self.tey_profiles[-1], self.tpy_profiles[-1]])
                self.prof_count += 1
                self.filter_labels.append('Filtered '+tag)
            self.mplot_curves = curves
            self.mplot_raw_curves = []
            self.mplot_labels = labels
            self.mplot_override = ["Inverse units", "Fourier Transform"]
            self.overplotworked = True
            self.finished_filter.emit()
            return "Done"
    def flux_correction(self):
        self.flux_curves = []
        self.flux_labels = []
        self.flux_units = []
        self.flux_energies = []
        self.overplotworked = False
        if len(self.retvals) < 1:
            self.finished_flux.emit()
            return None
        else:
            self.reduce_profiles()
            nums, xmin, xmax, usetey, usetpy, names = [], [], [], [], [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                xmin.append(self.retvals[nr][1])
                xmax.append(self.retvals[nr][2])
                usetey.append(self.retvals[nr][3])
                usetpy.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
            nums =np.array(nums)
            xmin =np.array(xmin)
            xmax =np.array(xmax)
            othermin,  othermax = np.min(self.cuts),  np.max(self.cuts)
            curves = []
            labels = []
            tags = []
            teylist = []
            tpylist = []
            taggeddict = {}
            firsttpy = 0
            for rn, num in enumerate(nums):
                if not usetey[rn]:
                    continue
                temp = self.reduced_tey_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TEY from ' + names[rn])      
                teylist.append(names[rn])
                tags.append(names[rn])
                taggeddict[names[rn]] = [None, None]
                firsttpy = rn+1
            for rn, num in enumerate(nums):
                if not usetpy[rn]:
                    continue
                temp = self.reduced_tpy_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TPY from ' + names[rn])
                tpylist.append(names[rn])
                tags.append(names[rn])
                taggeddict[names[rn]] = [None, None]
            for n in range(len(curves)):
                xaxis = curves[n][:,0]
                # xstep = (xaxis[1:] - xaxis[:-1]).mean()
                fcorr = self.xas_corrector.returnInterpolatedCurve(xaxis, kind_notcovered = self.current_interp)
                fcrit = fcorr[:, 1] == 0.0
                new_y = curves[n][:, 1]/fcorr[:, 1] * 1.602176487 * 10**(-19)
                new_y[fcrit] = 0.0
                # norm = curves[n][:,1].sum()
                curves[n] = np.column_stack([xaxis, new_y,  np.zeros(len(xaxis))])
                labels[n] = 'FluxCorrected_' + labels[n]
                if n < firsttpy:
                    taggeddict[tags[n]][0] = np.column_stack([xaxis, new_y,  np.zeros(len(xaxis))])
                else:
                    taggeddict[tags[n]][1] = np.column_stack([xaxis, new_y,  np.zeros(len(xaxis))])
                # curves[n] = curves[n][np.where(curves[n][:,1] > 0.0)]
                # curves[n][:,1] *= 100.0
            self.flux_curves = []
            self.flux_labels = []
            self.flux_units = []
            self.flux_units.append("Energy [eV]")
            for tag in np.unique(tags):
                # tag = tags[n]
                print("Correcting: tag ", tag)
                self.fullnames += ["No file"]
                self.shortnames += ['FluxCorrected '+tag]
                teyprof = taggeddict[tag][0]
                tpyprof = taggeddict[tag][1]
                if teyprof is not None:
                    self.tey_profiles += [teyprof]
                    self.raw_tey_profiles += [teyprof]
                else:
                    xax = tpyprof[:, 0]
                    self.tey_profiles += [np.column_stack([xax, np.zeros((len(xax), 2))])]
                    self.raw_tey_profiles += [np.column_stack([xax, np.zeros((len(xax), 2))])]
                if tpyprof is not None:
                    self.tpy_profiles += [tpyprof]
                    self.raw_tpy_profiles += [tpyprof]
                else:
                    xax = teyprof[:, 0]
                    self.tpy_profiles += [np.column_stack([xax, np.zeros((len(xax), 2))])]
                    self.raw_tpy_profiles += [np.column_stack([xax, np.zeros((len(xax), 2))])]
                self.flux_curves.append([self.tey_profiles[-1], self.tpy_profiles[-1]])
                self.prof_count += 1
                self.flux_labels.append('FluxCorrected '+tag)
            self.mplot_curves = curves
            self.mplot_raw_curves = []
            self.mplot_labels = labels
            self.mplot_override = ["Energy [eV]", "arb. units"]
            self.overplotworked = True
            self.finished_flux.emit()
            return "Done"
    def multiplot(self):
        self.overplotworked = False
        if len(self.retvals) < 1:
            self.finished_overplot.emit()
            return None
        else:
            self.reduce_profiles()
            nums, xmin, xmax, names = [], [], [], []
            usetey,  usetpy = [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                xmin.append(self.retvals[nr][1])
                xmax.append(self.retvals[nr][2])
                usetey.append(self.retvals[nr][3])
                usetpy.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
            nums =np.array(nums)
            xmin =np.array(xmin)
            xmax =np.array(xmax)
            othermin,  othermax = np.min(self.cuts),  np.max(self.cuts)
            curves = []
            raw_curves = []
            labels = []
            for rn, num in enumerate(nums):
                if not usetey[rn]:
                    continue
                temp = self.reduced_tey_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TEY from ' + names[rn])
            for rn, num in enumerate(nums):
                if not usetey[rn]:
                    continue
                temp = self.reduced_raw_tey_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                raw_curves.append(temp)
                # step = cur[np.where( cur[:,1] < np.percentile(cur[:,1], 60.0) )][:,1].std()
            for rn, num in enumerate(nums):
                if not usetpy[rn]:
                    continue
                temp = self.reduced_tpy_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TPY from ' + names[rn])
            for rn, num in enumerate(nums):
                if not usetpy[rn]:
                    continue
                temp = self.reduced_raw_tpy_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                raw_curves.append(temp)
            for n in range(len(curves)):
                xaxis = curves[n][:,0]
                xstep = xaxis[1:] - xaxis[:-1]
                xstep = np.concatenate([xstep[:1], xstep])
                norm = (np.abs(curves[n][:,1]) * xstep).sum()
                # norm = curves[n][:,1].sum()
                curves[n][:,1:] /= norm
                shift = np.percentile(curves[n][:,1], self.bkg_perc) - 0.01
                curves[n][:,1] -= shift
                raw_curves[n][:, 1:] /= norm
                raw_curves[n][:, 1] -= shift
                # curves[n] = curves[n][np.where(curves[n][:,1] > 0.0)]
                # curves[n][:,1] *= 100.0
            self.mplot_curves = curves
            self.mplot_raw_curves = raw_curves
            self.mplot_labels = labels
            self.mplot_override = ["Energy [eV]", ""]
            self.overplotworked = True
            self.finished_overplot.emit()
            return "Done"
    def absoluteplot(self):
        self.overplotworked = False
        if len(self.retvals) < 1:
            self.finished_overplot.emit()
            return None
        else:
            self.reduce_profiles()
            nums, xmin, xmax, names = [], [], [], []
            usetey,  usetpy = [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                xmin.append(self.retvals[nr][1])
                xmax.append(self.retvals[nr][2])
                usetey.append(self.retvals[nr][3])
                usetpy.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
            nums =np.array(nums)
            xmin =np.array(xmin)
            xmax =np.array(xmax)
            othermin,  othermax = np.min(self.cuts),  np.max(self.cuts)
            curves = []
            raw_curves = []
            labels = []
            for rn, num in enumerate(nums):
                if not usetey[rn]:
                    continue
                temp = self.reduced_tey_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TEY from ' + names[rn])
            for rn, num in enumerate(nums):
                if not usetey[rn]:
                    continue
                temp = self.reduced_raw_tey_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                raw_curves.append(temp)
                # step = cur[np.where( cur[:,1] < np.percentile(cur[:,1], 60.0) )][:,1].std()
            for rn, num in enumerate(nums):
                if not usetpy[rn]:
                    continue
                temp = self.reduced_tpy_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TPY from ' + names[rn])
            for rn, num in enumerate(nums):
                if not usetpy[rn]:
                    continue
                temp = self.reduced_raw_tpy_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                raw_curves.append(temp)
            self.mplot_curves = curves
            self.mplot_raw_curves = raw_curves
            self.mplot_labels = labels
            self.mplot_override = ["Energy [eV]", ""]
            self.overplotworked = True
            self.finished_overplot.emit()
            return "Done"
    def many_as_one(self):
        self.overplotworked = False
        if len(self.retvals) < 1:
            self.finished_overplot.emit()
            return None
        else:
            # self.reduce_profiles()
            nums, xmin, xmax, names = [], [], [], []
            usetey,  usetpy = [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                xmin.append(self.retvals[nr][1])
                xmax.append(self.retvals[nr][2])
                usetey.append(self.retvals[nr][3])
                usetpy.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
            nums =np.array(nums)
            xmin =np.array(xmin)
            xmax =np.array(xmax)
            othermin,  othermax = np.min(self.cuts),  np.max(self.cuts)
            curves = []
            raw_curves = []
            labels = []
            for rn, num in enumerate(nums):
                if not usetey[rn]:
                    continue
                temp = self.raw_tey_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                raw_curves.append(temp)
                labels.append('TEY from ' + names[rn])
                # step = cur[np.where( cur[:,1] < np.percentile(cur[:,1], 60.0) )][:,1].std()
            for rn, num in enumerate(nums):
                if not usetpy[rn]:
                    continue
                temp = self.raw_tpy_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                raw_curves.append(temp)
                labels.append('TPY from ' + names[rn])
            total_points = np.row_stack(raw_curves)
            total_points = total_points[np.argsort(total_points[:, 0])]
            if self.binsize > 0.0:
                binsize = self.binsize
            else:
                binsize = 0.1
            new_x = np.arange(total_points[0, 0] - 0.5*binsize, total_points[-1, 0] + 0.51*binsize, binsize)
            result = place_data_in_bins(total_points, new_x)
            self.mplot_curves = [result]
            self.mplot_raw_curves = raw_curves
            self.mplot_labels = ["All scans, averaged"] + labels
            self.mplot_override = ["Energy [eV]", ""]
            self.merged_curve = result
            self.overplotworked = True   
            self.fullnames += ["No file"]
            self.shortnames += ["Merged data"]
            self.tey_profiles += [result]
            self.tpy_profiles += [result]
            self.raw_tey_profiles += [total_points]
            self.raw_tpy_profiles += [total_points]
            self.prof_count += 1
            self.mergeworked = True   
            self.finished_merge.emit()
            return "Done"
