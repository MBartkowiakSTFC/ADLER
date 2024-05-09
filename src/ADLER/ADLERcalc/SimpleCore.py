
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
The part of the ADLER code defining the backend for
storing and handling the RIXS spectra.
"""

import math
import numpy as np
import os
import time
from os.path import expanduser

from scipy.fftpack import rfft, irfft, fftfreq

# import ctypes

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QApplication


from ADLER.ADLERcalc.RixsMeasurement import RixsMeasurement
from ADLER.ADLERcalc.ioUtils import read_1D_curve_extended,\
                              WriteEnergyProfile
from ADLER.ADLERcalc.imageUtils import elastic_line_anyx
from ADLER.ADLERcalc.arrayUtils import merge2curves
from ADLER.ADLERcalc.fitUtils import gauss_denum

#### Object-Oriented part

class SimpleCore(QObject):
    cleared = pyqtSignal()
    loaded = pyqtSignal()
    fileparams = pyqtSignal(object)
    logmessage = pyqtSignal(str)
    finished_fitting = pyqtSignal()
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
        self.profiles = []
        self.reduced_profiles = []
        self.lastreduction = 1.0
        self.cutoff = 30
        self.fullnames = []
        self.shortnames = []
        self.timedata = None
        self.units = []
        self.energies = []
        self.prof_numbers = []
        self.mplot_curves = []
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
        self.current_rixsmap = 0
        self.rixsaxes = ["Photon energy (eV)", "Temperature (K)", "Q (1/A)", "2 theta (deg.)"]
        self.rixs_axis_label = self.rixsaxes[self.current_rixsmap]
        self.filter_override = ["Inverse units", "Fourier Transform"]
        self.map2Dplotax = [(1, 2048,),  (1, 2048)]
        self.prof_count = 0
        self.legpos = 0
        self.retvals = []
    def possible_rixsmap_axes(self):
        return self.rixsaxes
    def assign_boxes(self, boxes):
        self.boxes = boxes
        for db in boxes:
            names, values = db.returnValues()
            self.parlist += names
            for nam in names:
                self.pardict[nam] = values[nam]
            db.values_changed.connect(self.make_params_visible)
        self.make_params_visible()
    @pyqtSlot(int)
    def rixsmap_axis(self, newnum):
        self.current_rixsmap = newnum
        self.rixs_axis_label = self.rixsaxes[newnum]
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
            elif 'cutoff' in k:
                self.cutoff = val
    def reduce_profiles(self):
        if len(self.profiles) == len(self.reduced_profiles):
            if self.redfac == self.lastreduction:
                return False
        self.reduced_profiles = []
        if self.redfac == 1.0:
            for p in self.profiles:
                self.reduced_profiles.append(p)
        else:
            for p in self.profiles:
                steps = len(p)
                newsteps = int(round(steps/self.redfac))
                target = np.zeros([newsteps, 2])
                target[:, 0] = np.linspace(p[:, 0].min(), p[:, 0].max(), newsteps)
                newone = merge2curves(p,  target)
                self.reduced_profiles.append(newone)
        self.lastreduction = self.redfac
        return True
    @pyqtSlot()
    def take_table_values(self):
        # self.table_obj.update_values()
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
        for fnum, fname in enumerate(flist):
            self.temp_path, short_name = os.path.split(fname)
            try:
                p, e, u, d = read_1D_curve_extended(fname)
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
        self.fullnames += fnames
        self.shortnames += snames
        self.profiles += profiles
        self.energies += envals
        self.units += units
        self.prof_count += len(fnames)
        self.loaded.emit()
        self.fileparams.emit([snames, profiles, envals, units, pardicts])
        return snames, profiles, envals, units, pardicts
    @pyqtSlot()
    def clear_profiles(self):
        self.fullnames = []
        self.shortnames = []
        self.profiles = []
        self.reduced_profiles = []
        self.energies = []
        self.units = []
        self.timedata = None
        self.prof_numbers = []
        self.prof_count = 0
        self.cleared.emit()
    def autofit_many(self):
        self.fitsworked = False
        if len(self.retvals) < 1:
            self.finished_fitting.emit()
            return None
        else:
            self.reduce_profiles()
            nums, Ei, xmin, xmax, xunits, names, temps, twothetas, qs = [], [], [], [], [], [], [], [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                Ei.append(self.retvals[nr][1])
                xmin.append(self.retvals[nr][2])
                xmax.append(self.retvals[nr][3])
                xunits.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
                temps.append(self.retvals[nr][6])
                twothetas.append(self.retvals[nr][7])
                qs.append(self.retvals[nr][8])
            chan,  en,  entran = 0, 0, 0
            for un in xunits:
                if un == 0:
                    chan+=1
                elif un == 1:
                    entran+=1
                elif un == 2:
                    en +=1
            iunits = np.array(xunits)
            if entran >0:
                self.logger("Plotting energy transfer. Other units will be discarded.")
                crit = np.where(iunits ==1)
                unit = 1
                fwhm_guess = 0.05
            elif en > 0:
                self.logger("Plotting energy. Other units will be discarded.")
                crit = np.where(iunits ==2)
                unit = 2
                fwhm_guess = 0.05
            else:
                self.logger("Plotting raw channels. This is most likely a debugging run.")
                crit = np.where(iunits ==0)
                unit = 0
                fwhm_guess = 4.0
            nums =np.array(nums)[crit]
            xvals =np.array(Ei)[crit]
            xmin =np.array(xmin)[crit]
            xmax =np.array(xmax)[crit]
            curves = []
            bkgs = []
            labels = []
            for rn, num in enumerate(nums):
                temp = self.reduced_profiles[num].copy()
                bkg = temp[np.where(temp[:, 1] < np.percentile(temp[:,1], self.bkg_perc))]
                temp = temp[np.where(temp[:, 0] >= xmin[rn])]
                temp = temp[np.where(temp[:, 0] <= xmax[rn])]
                curves.append(temp)
                bkgs.append(bkg)
                labels.append(str(xvals[rn]) + " eV, " + names[rn])
                # step = cur[np.where( cur[:,1] < np.percentile(cur[:,1], 60.0) )][:,1].std()
            width,  widtherr = [], []
            area,  areaerr = [], []
            centre, centreerr = [], []
            peakcurves = []
            for n in range(len(curves)):
                temp = curves[n]
                bkg = bkgs[n]
                fit, peakshape,  chi2 = elastic_line_anyx(temp,  bkg,  init_fwhm = fwhm_guess)
                if fit is None:
                    width.append(-1.0)
                    widtherr.append(-1.0)
                    area.append(0.0)
                    areaerr.append(-1.0)
                    centre.append(0.0)
                    centreerr.append(-1.0)
                    peakcurves.append(np.column_stack([np.arange(10),  np.zeros(10)]))
                else:
                    peak_area = fit[0][0]*abs(fit[0][1])/gauss_denum*(2*np.pi)**0.5
                    peak_area_error = peak_area*((fit[1][0]/fit[0][0])**2 + (fit[1][1]/fit[0][1])**2)**0.5
                    width.append(abs(round(fit[0][1],3)))
                    widtherr.append(abs(round(fit[1][1],3)))
                    centre.append(abs(round(fit[0][2],3)))
                    centreerr.append(abs(round(fit[1][2],3)))
                    area.append(round(peak_area,3))
                    areaerr.append(abs(round(peak_area_error,3)))
                    peakcurves.append(peakshape)
            self.mplot_curves = curves
            self.mplot_labels = labels
            self.mplot_fits = peakcurves
            self.mplot_fitparams = [nums, width,  widtherr,  area,  areaerr, centre, centreerr]
            if unit ==1:
                self.mplot_override = ["Energy transfer [eV]", ""]
            elif unit ==2:
                self.mplot_override = ["Energy [eV]", ""]
            else:
                self.mplot_override = ["Channels", ""]
            self.fitsworked = True
            self.finished_fitting.emit()
            return "Done"
    def fit_many(self):
        self.fitsworked = False
        if len(self.retvals) < 1:
            self.finished_fitting.emit()
            return None
        else:
            self.reduce_profiles()
            nums, Ei, xmin, xmax, xunits, names, temps, twothetas, qs = [], [], [], [], [], [], [], [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                Ei.append(self.retvals[nr][1])
                xmin.append(self.retvals[nr][2])
                xmax.append(self.retvals[nr][3])
                xunits.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
                temps.append(self.retvals[nr][6])
                twothetas.append(self.retvals[nr][7])
                qs.append(self.retvals[nr][8])
            chan,  en,  entran = 0, 0, 0
            for un in xunits:
                if un == 0:
                    chan+=1
                elif un == 1:
                    entran+=1
                elif un == 2:
                    en +=1
            iunits = np.array(xunits)
            if entran >0:
                self.logger("Plotting energy transfer. Other units will be discarded.")
                crit = np.where(iunits ==1)
                unit = 1
                fwhm_guess = 0.05
            elif en > 0:
                self.logger("Plotting energy. Other units will be discarded.")
                crit = np.where(iunits ==2)
                unit = 2
                fwhm_guess = 0.05
            else:
                self.logger("Plotting raw channels. This is most likely a debugging run.")
                crit = np.where(iunits ==0)
                unit = 0
                fwhm_guess = 4.0
            nums =np.array(nums)[crit]
            xvals =np.array(Ei)[crit]
            xmin =np.array(xmin)[crit]
            xmax =np.array(xmax)[crit]
            curves = []
            bkgs = []
            labels = []
            for rn, num in enumerate(nums):
                temp = self.reduced_profiles[num].copy()
                bkg = temp[np.where(temp[:, 1] < np.percentile(temp[:,1], self.bkg_perc))]
                temp = temp[np.where(temp[:, 0] >= xmin[rn])]
                temp = temp[np.where(temp[:, 0] <= xmax[rn])]
                curves.append(temp)
                bkgs.append(bkg)
                labels.append(str(xvals[rn]) + " eV, " + names[rn])
                # step = cur[np.where( cur[:,1] < np.percentile(cur[:,1], 60.0) )][:,1].std()
            width,  widtherr = [], []
            area,  areaerr = [], []
            centre, centreerr = [], []
            peakcurves = []
            for n in range(len(curves)):
                temp = curves[n]
                bkg = bkgs[n]
                fit, peakshape, chi2 = elastic_line_anyx(temp,  bkg, olimits = self.eline,  init_fwhm = fwhm_guess)
                if fit is None:
                    width.append(-1.0)
                    widtherr.append(-1.0)
                    area.append(0.0)
                    areaerr.append(-1.0)
                    peakcurves.append(np.column_stack([np.arange(10),  np.zeros(10)]))
                else:
                    peak_area = fit[0][0]*abs(fit[0][1])/gauss_denum*(2*np.pi)**0.5
                    peak_area_error = peak_area*((fit[1][0]/fit[0][0])**2 + (fit[1][1]/fit[0][1])**2)**0.5
                    width.append(abs(round(fit[0][1],3)))
                    widtherr.append(abs(round(fit[1][1],3)))
                    centre.append(abs(round(fit[0][2],3)))
                    centreerr.append(abs(round(fit[1][2],3)))
                    area.append(round(peak_area,3))
                    areaerr.append(abs(round(peak_area_error,3)))
                    peakcurves.append(peakshape)
            self.mplot_curves = curves
            self.mplot_labels = labels
            self.mplot_fits = peakcurves
            self.mplot_fitparams = [nums, width,  widtherr,  area,  areaerr, centre,  centreerr]
            if unit ==1:
                self.mplot_override = ["Energy transfer [eV]", ""]
            elif unit ==2:
                self.mplot_override = ["Energy [eV]", ""]
            else:
                self.mplot_override = ["Channels", ""]
            self.fitsworked = True
            self.finished_fitting.emit()
            return "Done"    
    def manual_merge(self):
        self.mergeworked = False
        if len(self.retvals) < 1:
            self.finished_merge.emit()
            return None
        else:
            self.reduce_profiles()
            nums, Ei, xmin, xmax, xunits, names, temps, twothetas, qs = [], [], [], [], [], [], [], [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                Ei.append(self.retvals[nr][1])
                xmin.append(self.retvals[nr][2])
                xmax.append(self.retvals[nr][3])
                xunits.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
                temps.append(self.retvals[nr][6])
                twothetas.append(self.retvals[nr][7])
                qs.append(self.retvals[nr][8])
            chan,  en,  entran = 0, 0, 0
            for un in xunits:
                if un == 0:
                    chan+=1
                elif un == 1:
                    entran+=1
                elif un == 2:
                    en +=1
            iunits = np.array(xunits)        
            if entran >0:
                self.logger("Merging energy transfer. Other units will be discarded.")
                crit = np.where(iunits ==1)
                self.merged_units = ["Energy transfer [eV]",  "Intensity [arb. units]"]
                unit = 1
                units = "Energy Transfer [eV]"
                fwhm_guess = 0.05
            elif en > 0:
                self.logger("Merging energy. This result is not likely to be useful.")
                crit = np.where(iunits ==2)
                self.merged_units = ["Energy [eV]",  "Intensity [arb. units]"]
                unit = 2
                units = "Energy [eV]"
                fwhm_guess = 0.05
            else:
                self.logger("Merging raw channels. This is probably a bad idea.")
                crit = np.where(iunits ==0)
                self.merged_units = ["Detector channels [pixel]",  "Intensity [arb. units]"]
                units = "Detector channels"
                unit = 0
                fwhm_guess = 4.0
            nums =np.array(nums)[crit]
            xvals =np.array(Ei)[crit]
            xmin =np.array(xmin)[crit]
            xmax =np.array(xmax)[crit]        
            minx, maxx, stepx = 1e5,-1e5,-1.0    
            curves = []
            labels = []
            for rn, num in enumerate(nums):
                dat = self.reduced_profiles[num].copy()
                dat = dat[np.where(dat[:, 0] > xmin[rn])]
                dat = dat[np.where(dat[:, 0] < xmax[rn])]
                minx = min(minx, dat[:,0].min())
                maxx = max(maxx, dat[:,0].max())
                step = (dat[1:,0] - dat[:-1,0]).mean()
                stepx = max(stepx, step)
                curves.append(dat)
            newx = np.arange(minx, maxx + 0.1*stepx, stepx)
            target = np.zeros((len(newx),2))
            target[:,0] = newx
            for n in range(len(curves)):
                target = merge2curves(curves[n],target)
            self.merged_curve = target
            self.merged_units = units
            self.merged_energy = str(xvals.mean())
            self.merged_temperature = str(np.array(temps).mean())
            self.merged_2theta = str(np.array(twothetas).mean())
            self.merged_q = str(np.array(qs).mean())
            self.mergeworked = True
            # now we try to add the fitting            
            self.fullnames += ["No file"]
            self.shortnames += ["Merged data"]
            self.profiles += [target]
            self.energies += [self.merged_energy]
            self.units += [units]
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
        nums, labels,  curves = [], [], []
        for nr in range(len(self.retvals)):
            nums.append(self.retvals[nr][0])
        for rn, num in enumerate(nums):
            curves.append(self.reduced_profiles[num].copy() )
            labels.append(self.shortnames[rn])
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
            nums, Ei, xmin, xmax, xunits, names, temps, twothetas, qs = [], [], [], [], [], [], [], [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                Ei.append(self.retvals[nr][1])
                xmin.append(self.retvals[nr][2])
                xmax.append(self.retvals[nr][3])
                xunits.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
                temps.append(self.retvals[nr][6])
                twothetas.append(self.retvals[nr][7])
                qs.append(self.retvals[nr][8])
            nums =np.array(nums)
            xvals =np.array(Ei)
            xmin =np.array(xmin)
            xmax =np.array(xmax)
            curves = []
            labels = []
            for rn, num in enumerate(nums):
                temp = self.reduced_profiles[num].copy()
                temp = temp[np.where(temp[:, 0] > xmin[rn])]
                temp = temp[np.where(temp[:, 0] < xmax[rn])]
                curves.append(temp)
                labels.append(str(xvals[rn]) + " eV, " + names[rn])
                # step = cur[np.where( cur[:,1] < np.percentile(cur[:,1], 60.0) )][:,1].std()
            for n in range(len(curves)):
                xaxis = curves[n][:,0]
                xstep = (xaxis[1:] - xaxis[:-1]).mean()
                # new_x = fftfreq(len(xaxis), xstep)
                new_y = rfft(curves[n][:, 1])
                new_x = np.arange(len(new_y)) + 1
                # norm = curves[n][:,1].sum()
                curves[n] = np.column_stack([new_x, new_y])
                # curves[n] = curves[n][np.where(curves[n][:,1] > 0.0)]
                # curves[n][:,1] *= 100.0
            self.mplot_curves = curves
            self.mplot_labels = labels
            self.mplot_override = ["Inverse units", "Fourier Transform"]
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
            nums, Ei, xmin, xmax, xunits, names, temps, twothetas, qs = [], [], [], [], [], [], [], [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                Ei.append(self.retvals[nr][1])
                xmin.append(self.retvals[nr][2])
                xmax.append(self.retvals[nr][3])
                xunits.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
                temps.append(self.retvals[nr][6])
                twothetas.append(self.retvals[nr][7])
                qs.append(self.retvals[nr][8])
            nums =np.array(nums)
            xvals =np.array(Ei)
            xmin =np.array(xmin)
            xmax =np.array(xmax)
            curves = []
            labels = []
            for rn, num in enumerate(nums):
                temp = self.reduced_profiles[num].copy()
                temp = temp[np.where(temp[:, 0] > xmin[rn])]
                temp = temp[np.where(temp[:, 0] < xmax[rn])]
                curves.append(temp)
                labels.append(names[rn])
                # step = cur[np.where( cur[:,1] < np.percentile(cur[:,1], 60.0) )][:,1].std()
            for n in range(len(curves)):
                xaxis = curves[n][:,0]
                xstep = (xaxis[1:] - xaxis[:-1]).mean()
                new_x = fftfreq(len(xaxis), xstep)
                new_y = rfft(curves[n][:, 1])
                new_y[-self.cutoff:] = 0.0
                result = irfft(new_y)
                # norm = curves[n][:,1].sum()
                curves[n] = np.column_stack([xaxis, result])
                labels[n] = 'Filtered_' + labels[n]
                # curves[n] = curves[n][np.where(curves[n][:,1] > 0.0)]
                # curves[n][:,1] *= 100.0
            self.filter_curves = curves
            self.filter_labels = labels
            self.filter_units = []
            self.filter_temperatures = temps
            self.filter_2thetas = twothetas
            self.filter_qs = qs
            for xu in xunits:
                if xu == 1:
                    self.filter_units.append("Energy Transfer [eV]")
                elif xu == 2:
                    self.filter_units.append("Energy [eV]")
                elif xu == 0:
                    self.filter_units.append("Detector channels")
                else:
                    self.filter_units.append("???")
            self.filter_energies = Ei
            for n,  lab in enumerate(self.filter_labels):
                self.fullnames += ["No file"]
                self.shortnames += [lab]
                self.profiles += [self.filter_curves[n]]
                self.energies += [self.filter_energies[n]]
                self.units += [self.filter_units[n]]
                self.prof_count += 1
            self.mplot_curves = curves
            self.mplot_labels = labels
            self.mplot_override = ["Inverse units", "Fourier Transform"]
            self.overplotworked = True
            self.finished_filter.emit()
            return "Done"
    def multiplot(self):
        self.overplotworked = False
        if len(self.retvals) < 1:
            self.finished_overplot.emit()
            return None
        else:
            self.reduce_profiles()
            nums, Ei, xmin, xmax, xunits, names, temps, twothetas, qs = [], [], [], [], [], [], [], [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                Ei.append(self.retvals[nr][1])
                xmin.append(self.retvals[nr][2])
                xmax.append(self.retvals[nr][3])
                xunits.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
                temps.append(self.retvals[nr][6])
                twothetas.append(self.retvals[nr][7])
                qs.append(self.retvals[nr][8])
            chan,  en,  entran = 0, 0, 0
            for un in xunits:
                if un == 0:
                    chan+=1
                elif un == 1:
                    entran+=1
                elif un == 2:
                    en +=1
            iunits = np.array(xunits)
            if entran >0:
                self.logger("Plotting energy transfer. Other units will be discarded.")
                crit = np.where(iunits ==1)
                unit = 1
            elif en > 0:
                self.logger("Plotting energy. Other units will be discarded.")
                crit = np.where(iunits ==2)
                unit = 2
            else:
                self.logger("Plotting raw channels. This is most likely a debugging run.")
                crit = np.where(iunits ==0)
                unit = 0
            nums =np.array(nums)[crit]
            xvals =np.array(Ei)[crit]
            xmin =np.array(xmin)[crit]
            xmax =np.array(xmax)[crit]
            curves = []
            labels = []
            for rn, num in enumerate(nums):
                temp = self.reduced_profiles[num].copy()
                temp = temp[np.where(temp[:, 0] > xmin[rn])]
                temp = temp[np.where(temp[:, 0] < xmax[rn])]
                curves.append(temp)
                labels.append(str(xvals[rn]) + " eV, " + names[rn])
                # step = cur[np.where( cur[:,1] < np.percentile(cur[:,1], 60.0) )][:,1].std()
            for n in range(len(curves)):
                xaxis = curves[n][:,0]
                xstep = xaxis[1:] - xaxis[:-1]
                xstep = np.concatenate([xstep[:1], xstep])
                norm = (np.abs(curves[n][:,1]) * xstep).sum()
                # norm = curves[n][:,1].sum()
                curves[n][:,1] /= norm
                shift = np.percentile(curves[n][:,1], self.bkg_perc) - 0.01
                curves[n][:,1] -= shift
                # curves[n] = curves[n][np.where(curves[n][:,1] > 0.0)]
                # curves[n][:,1] *= 100.0
            self.mplot_curves = curves
            self.mplot_labels = labels
            if unit ==1:
                self.mplot_override = ["Energy transfer [eV]", ""]
            elif unit ==2:
                self.mplot_override = ["Energy [eV]", ""]
            else:
                self.mplot_override = ["Channels", ""]
            self.overplotworked = True
            self.finished_overplot.emit()
            return "Done"
    def rixsmap(self):
        self.rixs_worked = False
        if len(self.retvals) <= 1:
            self.finished_rixsmap.emit()
            return None
        else:
            self.reduce_profiles()
            nums, Ei, xmin, xmax, xunits, names, temps, twothetas, qs = [], [], [], [], [], [], [], [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                Ei.append(self.retvals[nr][1])
                xmin.append(self.retvals[nr][2])
                xmax.append(self.retvals[nr][3])
                xunits.append(self.retvals[nr][4])
                temps.append(self.retvals[nr][6])
                twothetas.append(self.retvals[nr][7])
                qs.append(self.retvals[nr][8])
            counter = 0
            for e in Ei:
                if e > 0.0:
                    counter += 1
            if not (counter > 0):
                return None
            chan,  en,  entran = 0, 0, 0
            for un in xunits:
                if un == 0:
                    chan+=1
                elif un == 1:
                    entran+=1
                elif un == 2:
                    en +=1
            iunits = np.array(xunits)
            if entran >0:
                self.logger("Plotting energy transfer. Other units will be discarded.")
                crit = np.where(iunits ==1)
            elif en > 0:
                self.logger("Plotting energy. Other units will be discarded.")
                crit = np.where(iunits ==2)
            else:
                self.logger("Plotting raw channels. This is most likely a debugging run.")
                crit = np.where(iunits ==0)
            nums =np.array(nums)[crit]
            if self.current_rixsmap == 0:
                xvals =np.array(Ei)[crit]
            elif self.current_rixsmap == 1:
                xvals =np.array(temps)[crit]
            elif self.current_rixsmap == 2:
                xvals =np.array(twothetas)[crit]
            elif self.current_rixsmap == 3:
                xvals =np.array(qs)[crit]
            xmin =np.array(xmin)[crit]
            xmax =np.array(xmax)[crit]
            ocurves, curves = [], []
            step = 0.0
            ymin, ymax, ystep = 1e5, -1e5, 100.0
            for rn, num in enumerate(nums):
                temp = self.reduced_profiles[num].copy()
                temp = temp[np.where(temp[:, 0] > xmin[rn])]
                temp = temp[np.where(temp[:, 0] < xmax[rn])]
                ocurves.append(temp)
            for n in range(len(ocurves)):
                cxaxis = ocurves[n][:,0]
                cxstep = cxaxis[1:] - cxaxis[:-1]
                cxstep = np.concatenate([cxstep[:1], cxstep])
                ymin = min(ymin, cxaxis.min())
                ymax = max(ymax, cxaxis.max())
                ystep = min(ystep, cxstep.max())
                norm = (ocurves[n][:,1] * cxstep).sum()
                ocurves[n][:,1] /= norm
                shift = np.percentile(ocurves[n][:,1], 75.0) - 0.01
                ocurves[n][:,1] -= shift
                ocurves[n] = ocurves[n][np.where(ocurves[n][:,1] > 0.0)]
                ocurves[n][:,1] *= 100.0
            sequence = np.argsort(xvals)
            # print("RIXSMAP energies: ", xvals)
            # print("RIXSMAP sequence: ", sequence)
            xvals = xvals[sequence]
            crit = np.where(xvals > 0.0)
            # print(crit)
            xvals = xvals[crit]
            for ns, se in enumerate(sequence):
                # print(se)
                if ns in crit[0]:
                    curves.append(ocurves[se])
            # curves = curves[crit]
            # now we need to define a grid for the map
            yaxis = np.arange(ymin, ymax+0.1*ystep, ystep)
            npixy = len(yaxis)
            xmin = xvals.min()
            xmax = xvals.max()
            xstep = max((xvals[1:] - xvals[:-1]).min(), 0.05)
            xaxis = np.arange(xmin - xstep, xmax + xstep*1.01, xstep/5.0)
            xmask = np.zeros(len(xaxis))
            npixx = len(xaxis)
            # define the 2D arrays as needed
            map_array = np.zeros((npixy,npixx)).astype(np.float64)
            # assign correct values
            # mcurves = []
            for n in range(len(curves)):
                xcrit = np.abs(xaxis - xvals[n])
                pos, = np.where(xcrit == xcrit.min())
                target = np.zeros((npixy,2))
                target[:,0] = yaxis.copy()
                print("RIXSMAP curve min/max:",curves[n][:,1].min(),curves[n][:,1].max())
                yvals = merge2curves(curves[n], target)[:,1]
                # mcurves.append(merge2curves(curves[n], target))
                map_array[:,pos] = yvals.reshape(map_array[:,pos].shape)
                print("RIXSMAP pos, posshape: ", pos, map_array[:,pos].shape)
                # map_array[:len(curves[n]),pos] = curves[n][:,1].reshape(map_array[:len(curves[n]),pos].shape)
                xmask[pos] = 1
            # apply smearing
            # return None
            # print("RIXSMAP array min/max:",map_array.min(),map_array.max())
            virt_array = np.zeros(map_array.shape).astype(np.float64)
            weight_array = np.zeros(len(xaxis))
            # smearwidth = 2.0 # meV
            smearwidth = self.smear
            gridstep = xstep/5.0
            width = int(math.ceil(smearwidth/gridstep))
            # print("RIXSMAP step, gridstep, width", xstep, gridstep, width)
            for n in range(len(xaxis)):
                if xmask[n]:
                    w_axis = np.zeros(len(xaxis))
                    w_axis[n] = 1.0
                    neglim, poslim = 0,0
                    for s in range(1,width):
                        if n-s >= 0:
                            neglim = s
                            if xmask[n-s]:
                                break
                    for s in range(1,neglim):
                        if n-s >= 0:
                            w_axis[n-s] = 1.0 - s/float(neglim)
                    for s in range(1,width):
                        if n+s < len(xmask):
                            poslim = s
                            if xmask[n+s]:
                                break
                    for s in range(1,poslim):
                        if n+s < len(xmask):
                            w_axis[n+s] = 1.0 - s/float(poslim)
                    for s in range(len(xaxis)):
                        if w_axis[s] > 0.0:
                            if xmask[s] > 0.0:
                                virt_array[:,s] = map_array[:,s].copy()
                                weight_array[s] = 1.0
                            else:
                                virt_array[:,s] += w_axis[s]*(map_array[:,n].copy())
                                weight_array[s] += w_axis[s]
            print("RIXSMAP weigth array: ", weight_array)
            for n in range(len(xaxis)):
                if weight_array[n] > 0.0:
                    virt_array[:,n] /= weight_array[n]
            # plot!
            self.map2D = [virt_array, map_array]
            self.map2Dplotax = [(ymin,ymax), (xmin, xmax)]
            self.rixs_worked = True
            self.finished_rixsmap.emit()
            return "Done"


class NewSimpleCore(QStandardItemModel):
    cleared = pyqtSignal()
    loaded = pyqtSignal()
    fileparams = pyqtSignal(object)
    logmessage = pyqtSignal(str)
    finished_fitting = pyqtSignal()
    finished_overplot = pyqtSignal()
    finished_rixsmap = pyqtSignal()
    finished_merge = pyqtSignal()
    finished_filter = pyqtSignal()
    def __init__(self, master, boxes, logger,  progress_bar = None,  table_headers = None, max_threads = 1, 
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
        # self.table_obj = table
        self.temp_path = startpath
        self.profiles = []
        self.reduced_profiles = []
        self.matching_numbers = []
        self.lastreduction = 1.0
        self.cutoff = 30
        self.fullnames = []
        self.shortnames = []
        self.timedata = None
        self.units = []
        self.energies = []
        self.prof_numbers = []
        self.mplot_curves = []
        self.mplot_labels =[]
        self.mplot_override = ["Energy transfer [eV]", ""]
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
        self.fwhm_guess = 4.0
        self.arbitrary_range = np.array([-8192.0, 8192.0])
        self.current_rixsmap = 0
        self.fname_suffix = "eV"
        self.rixsaxes = ["Photon energy (eV)", "Temperature (K)", "Q (1/A)", "2 theta (deg.)"]
        self.rixs_axis_label = self.rixsaxes[self.current_rixsmap]
        self.current_rixsmap_ax2 = 0
        self.rixsaxes2 = ["Absolute energy (eV)", "Energy transfer (eV)", "Detector channels"]
        self.rixs_axis2_label = self.rixsaxes2[self.current_rixsmap_ax2]
        self.current_1dplot_axis = 1
        self.plotaxes = ["Absolute energy (eV)", "Energy transfer (eV)", "Detector channels"]
        self.normoptions = ["Integrated ring current",  "Total counts", "Time", "Number of scans",
                                    "Peak area", "Peak position", "Peak width", "Arbitrary range"]
        self.normflags = [False] * len(self.normoptions)
        self.plot_axis_label = self.plotaxes[self.current_1dplot_axis]
        self.filter_override = ["Inverse units", "Fourier Transform"]
        self.map2Dplotax = [(1, 2048,),  (1, 2048)]
        self.prof_count = 0
        self.legpos = 0
        self.cnames = table_headers
        self.col_order = table_headers
        self.setHorizontalHeaderLabels(table_headers)
    def possible_normalisation_choices(self):
        return self.normoptions
    def possible_rixsmap_axes(self):
        return self.rixsaxes
    def possible_rixsmap_axes2(self):
        return self.rixsaxes2
    def possible_plot_axes(self):
        return self.plotaxes
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
    def normalisation_flags(self, flaglist):
        self.normflags = flaglist
    @pyqtSlot(int)
    def rixsmap_axis(self, newnum):
        self.current_rixsmap = newnum
        self.rixs_axis_label = self.rixsaxes[newnum]
    @pyqtSlot(int)
    def rixsmap_axis_Y(self, newnum):
        self.current_rixsmap_ax2 = newnum
        self.rixs_axis2_label = self.rixsaxes[newnum]
    @pyqtSlot(int)
    def plot_axis(self, newnum):
        self.current_1dplot_axis = newnum
        self.plot_axis_label = self.rixsaxes[newnum]
        if newnum == 1:
            self.mplot_override = ["Energy transfer [eV]", ""]
            self.fwhm_guess = 0.05
            self.fname_suffix = "eV"
        elif newnum ==0:
            self.mplot_override = ["Energy [eV]", ""]
            self.fwhm_guess = 0.05
            self.fname_suffix = "absEnergy"
        else:
            self.mplot_override = ["Channels", ""]
            self.fwhm_guess = 4.0
            self.fname_suffix = "channels"
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
            elif 'cutoff' in k:
                self.cutoff = val
        self.arbitrary_range[0:2] = [self.eline[0], self.eline[1]]
    def reduce_profiles(self,  for_rixs=False):
        self.reduced_profiles = []
        self.matching_numbers = []
        temp = []
        norms = []
        flags = []
        unit = -1
        self.update_ticks()
        if for_rixs:
            dummy_x = np.linspace(-50, 50, 256)
            dummy_y = np.zeros(dummy_x.shape)
            if self.current_rixsmap_ax2 ==0:
                unit = 0
                for p in self.profiles:
                    temp.append(p.profile_absEnergy.copy())
            if self.current_rixsmap_ax2 ==1:
                unit = 1
                for p in self.profiles:
                    temp.append(p.profile_eV.copy())
            if self.current_rixsmap_ax2 ==2:
                unit = 2
                for p in self.profiles:
                    temp.append(p.profile_channels.copy())
        else:
            if self.current_1dplot_axis == 0:
                unit = 0
                for p in self.profiles:
                    temp.append(p.profile_absEnergy.copy())
                dummy_x = np.linspace(200, 1200, 256)
            elif self.current_1dplot_axis == 1:
                unit = 1
                for p in self.profiles:
                    temp.append(p.profile_eV.copy())
                dummy_x = np.linspace(-50, 50, 256)
            elif self.current_1dplot_axis == 2:
                unit = 2
                for p in self.profiles:
                    temp.append(p.profile_channels.copy())
                dummy_x = np.linspace(1, 2048, 256)
            dummy_y = np.zeros(dummy_x.shape)
        for p in self.profiles:
            flags.append(p.active)
            p.set_integration_range(self.arbitrary_range, unit)
            norms.append(p.norm_number(self.normflags, unit))
        print(flags)
        for nn,  p in enumerate(temp):
            steps = len(p)
            norm = norms[nn]
            if flags[nn]:
                self.matching_numbers.append(nn)
                if steps <1:
                    print("Slot ",nn,": Profile missing, skipping.")
                    self.reduced_profiles.append(np.column_stack([dummy_x, dummy_y]))
                else:
                    tprof = p.copy()
                    tprof[:, 1] /= norm
                    newsteps = int(round(steps/self.redfac))
                    target = np.zeros([newsteps, 2])
                    target[:, 0] = np.linspace(p[:, 0].min(), p[:, 0].max(), newsteps)
                    newone = merge2curves(p,  target)
                    self.reduced_profiles.append(newone)
        self.lastreduction = self.redfac
        return True
    # @pyqtSlot()
    # def take_table_values(self):
        # self.table_obj.update_values()
    #   self.retvals = self.table_obj.return_values()
    def logger(self,  message):
        now = time.gmtime()
        timestamp = ("ProfileCore "
                     +"-".join([str(tx) for tx in [now.tm_mday, now.tm_mon, now.tm_year]])
                     + ',' + ":".join([str(ty) for ty in [now.tm_hour, now.tm_min, now.tm_sec]]) + '| ')
        self.logmessage.emit(timestamp + message)
    @pyqtSlot(object)
    def load_profiles(self,  flist):
        for fnum, fname in enumerate(flist):
            self.temp_path, short_name = os.path.split(fname)
            temp = RixsMeasurement()
            if '.yaml' in short_name[-5:]:
                temp.read_yaml(fname)
            elif '.txt' in short_name[-4:]:
                temp.read_extended_ADLER(fname)
            self.profiles.append(temp)
            self.add_row(temp)
        self.loaded.emit()
    @pyqtSlot()
    def clear_profiles(self):
        self.fullnames = []
        self.shortnames = []
        self.profiles = []
        self.reduced_profiles = []
        self.energies = []
        self.units = []
        self.timedata = None
        self.prof_numbers = []
        self.prof_count = 0
        self.clear()
        self.setHorizontalHeaderLabels(self.cnames)
        self.col_order = self.cnames
        self.cleared.emit()
    def autofit_many(self):
        self.fitsworked = False
        curves = []
        bkgs = []
        labels = []
        self.reduce_profiles()
        nums, Ei, names, temps, twothetas, qs = [], [], [], [], [], []
        for nr in self.matching_numbers:
            rmeas = self.profiles[nr]
            pdict = rmeas.summariseCrucialParts()
            nums.append(nr)
            Ei.append(rmeas.energy)
            names.append(rmeas.shortsource)
            temps.append(pdict['temperature'])
            twothetas.append(pdict['arm_theta'])
            qs.append(pdict['Q'])
        for rn, num in enumerate(nums):
            temp = self.reduced_profiles[rn].copy()
            bkg = temp[np.where(temp[:, 1] < np.percentile(temp[:,1], self.bkg_perc))]
            curves.append(temp)
            bkgs.append(bkg)
            labels.append(str(Ei[rn]) + " eV, " + names[rn])
        width,  widtherr = [], []
        area,  areaerr = [], []
        centre, centreerr = [], []
        peakcurves = []
        for n in range(len(curves)):
            temp = curves[n]
            bkg = bkgs[n]
            fit, peakshape,  chi2 = elastic_line_anyx(temp,  bkg,  init_fwhm = self.fwhm_guess)
            row = nums[n]
            if fit is None:
                width.append(-1.0)
                widtherr.append(-1.0)
                area.append(0.0)
                areaerr.append(-1.0)
                centre.append(0.0)
                centreerr.append(-1.0)
                peakcurves.append(np.column_stack([np.arange(10),  np.zeros(10)]))
            else:
                peak_area = fit[0][0]*abs(fit[0][1])/gauss_denum*(2*np.pi)**0.5
                peak_area_error = peak_area*((fit[1][0]/fit[0][0])**2 + (fit[1][1]/fit[0][1])**2)**0.5
                width.append(abs(round(fit[0][1],3)))
                widtherr.append(abs(round(fit[1][1],3)))
                centre.append(abs(round(fit[0][2],3)))
                centreerr.append(abs(round(fit[1][2],3)))
                area.append(round(peak_area,3))
                areaerr.append(abs(round(peak_area_error,3)))
                peakcurves.append(peakshape)
            for nn, d in enumerate([width[-1], widtherr[-1], area[-1], areaerr[-1], centre[-1], centreerr[-1]]):
                column = nn + 6
                intermediate = str(d).strip("()[]'")
                try:
                    interm2 = round(float(intermediate), 3)
                except:
                    temp = QStandardItem(intermediate)
                else:
                    temp = QStandardItem(str(interm2))
                self.setItem(row, column, temp)
        self.mplot_curves = curves
        self.mplot_labels = labels
        self.mplot_fits = peakcurves
        self.mplot_fitparams = [nums, width,  widtherr,  area,  areaerr, centre, centreerr]
        self.fitsworked = True
        self.finished_fitting.emit()
        return "Done"
    def fit_many(self):
        self.fitsworked = False
        curves = []
        bkgs = []
        labels = []
        self.reduce_profiles()
        nums, Ei, names, temps, twothetas, qs = [], [], [], [], [], []
        for nr in self.matching_numbers:
            rmeas = self.profiles[nr]
            pdict = rmeas.summariseCrucialParts()
            nums.append(nr)
            Ei.append(rmeas.energy)
            names.append(rmeas.shortsource)
            temps.append(pdict['temperature'])
            twothetas.append(pdict['arm_theta'])
            qs.append(pdict['Q'])
        for rn, num in enumerate(nums):
            temp = self.reduced_profiles[rn].copy()
            bkg = temp[np.where(temp[:, 1] < np.percentile(temp[:,1], self.bkg_perc))]
            curves.append(temp)
            bkgs.append(bkg)
            labels.append(str(Ei[rn]) + " eV, " + names[rn])
        width,  widtherr = [], []
        area,  areaerr = [], []
        centre, centreerr = [], []
        peakcurves = []
        for n in range(len(curves)):
            temp = curves[n]
            bkg = bkgs[n]
            fit, peakshape,  chi2 = elastic_line_anyx(temp,  bkg, olimits = self.eline, init_fwhm = self.fwhm_guess)
            row = nums[n]
            if fit is None:
                width.append(-1.0)
                widtherr.append(-1.0)
                area.append(0.0)
                areaerr.append(-1.0)
                centre.append(0.0)
                centreerr.append(-1.0)
                peakcurves.append(np.column_stack([np.arange(10),  np.zeros(10)]))
            else:
                peak_area = fit[0][0]*abs(fit[0][1])/gauss_denum*(2*np.pi)**0.5
                peak_area_error = peak_area*((fit[1][0]/fit[0][0])**2 + (fit[1][1]/fit[0][1])**2)**0.5
                width.append(abs(round(fit[0][1],3)))
                widtherr.append(abs(round(fit[1][1],3)))
                centre.append(abs(round(fit[0][2],3)))
                centreerr.append(abs(round(fit[1][2],3)))
                area.append(round(peak_area,3))
                areaerr.append(abs(round(peak_area_error,3)))
                peakcurves.append(peakshape)
            for nn, d in enumerate([width[-1], widtherr[-1], area[-1], areaerr[-1], centre[-1], centreerr[-1]]):
                column = nn + 6
                intermediate = str(d).strip("()[]'")
                try:
                    interm2 = round(float(intermediate), 3)
                except:
                    temp = QStandardItem(intermediate)
                else:
                    temp = QStandardItem(str(interm2))
                self.setItem(row, column, temp)
        self.mplot_curves = curves
        self.mplot_labels = labels
        self.mplot_fits = peakcurves
        self.mplot_fitparams = [nums, width,  widtherr,  area,  areaerr, centre, centreerr]
        self.fitsworked = True
        self.finished_fitting.emit()
        return "Done"
    def manual_merge(self):
        self.update_ticks()
        templist = []
        for pr in self.profiles:
            if pr.active:
                templist.append(pr)
        if len(templist) > 0:
            newcurve = RixsMeasurement()
            newcurve.shortsource = "MergedCurve"
            for xp in templist:
                newcurve = newcurve + xp
            self.profiles.append(newcurve)
            self.add_row(newcurve)
        self.finished_merge.emit()
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
        nums, labels,  curves = [], [], []
        self.reduce_profiles()
        nums, Ei, names, temps, twothetas, qs = [], [], [], [], [], []
        for nr in self.matching_numbers:
            rmeas = self.profiles[nr]
            pdict = rmeas.summariseCrucialParts()
            nums.append(nr)
            Ei.append(rmeas.energy)
            names.append(rmeas.shortsource)
            temps.append(pdict['temperature'])
            twothetas.append(pdict['arm_theta'])
            qs.append(pdict['Q'])
        for rn, num in enumerate(nums):
            curves.append(self.reduced_profiles[rn].copy() )
            labels.append(str(self.item(num, 0).text()))
        if len(curves) ==0:
            self.logger("There are no curves to be saved.")
            return None
        else:
            for num in range(len(curves)):
                if labels[num][-4:] == '.txt':
                    target = fpath + "/" + labels[num][:-4] + "_" + self.fname_suffix + ".txt"
                else:
                    target = fpath + "/" + labels[num] + "_" + self.fname_suffix + '.txt'
                WriteEnergyProfile(target, curves[num], [])
            return True
            # outname = os.path.join(self.temp_path, self.temp_name+"_HISTOGRAM.txt")
            # WriteEnergyProfile(outname, target, [])
    def fft_curves(self):
        self.reduce_profiles()
        nums, Ei, names, temps, twothetas, qs = [], [], [], [], [], []
        for nr in self.matching_numbers:
            rmeas = self.profiles[nr]
            pdict = rmeas.summariseCrucialParts()
            nums.append(nr)
            Ei.append(rmeas.energy)
            names.append(rmeas.shortsource)
            temps.append(pdict['temperature'])
            twothetas.append(pdict['arm_theta'])
            qs.append(pdict['Q'])
        curves = []
        labels = []
        for rn, num in enumerate(nums):
            temp = self.reduced_profiles[rn].copy()
            curves.append(temp)
            labels.append(str(Ei[rn]) + " eV, " + names[rn])
            # step = cur[np.where( cur[:,1] < np.percentile(cur[:,1], 60.0) )][:,1].std()
        nums =np.array(nums)
        xvals =np.array(Ei)
        for n in range(len(curves)):
            xaxis = curves[n][:,0]
            xstep = (xaxis[1:] - xaxis[:-1]).mean()
            # new_x = fftfreq(len(xaxis), xstep)
            new_y = rfft(curves[n][:, 1])
            new_x = np.arange(len(new_y)) + 1
            # norm = curves[n][:,1].sum()
            curves[n] = np.column_stack([new_x, new_y])
            # curves[n] = curves[n][np.where(curves[n][:,1] > 0.0)]
            # curves[n][:,1] *= 100.0
        self.mplot_curves = curves
        self.mplot_labels = labels
        self.mplot_override = ["Inverse units", "Fourier Transform"]
        self.overplotworked = True
        self.finished_overplot.emit()
        return "Done"
    def fft_filter(self):
        self.reduce_profiles()
        nums, Ei, names, temps, twothetas, qs = [], [], [], [], [], []
        for nr in range(len(self.retvals)):
            nums.append(self.retvals[nr][0])
            Ei.append(self.retvals[nr][1])
            names.append(self.retvals[nr][2])
            temps.append(self.retvals[nr][3])
            twothetas.append(self.retvals[nr][4])
            qs.append(self.retvals[nr][5])
        nums =np.array(nums)
        xvals =np.array(Ei)
        curves = []
        labels = []
        for rn, num in enumerate(nums):
            temp = self.reduced_profiles[num].copy()
            curves.append(temp)
            labels.append(names[rn])
            # step = cur[np.where( cur[:,1] < np.percentile(cur[:,1], 60.0) )][:,1].std()
        for n in range(len(curves)):
            xaxis = curves[n][:,0]
            xstep = (xaxis[1:] - xaxis[:-1]).mean()
            new_x = fftfreq(len(xaxis), xstep)
            new_y = rfft(curves[n][:, 1])
            new_y[-self.cutoff:] = 0.0
            result = irfft(new_y)
            # norm = curves[n][:,1].sum()
            curves[n] = np.column_stack([xaxis, result])
            labels[n] = 'Filtered_' + labels[n]
            # curves[n] = curves[n][np.where(curves[n][:,1] > 0.0)]
            # curves[n][:,1] *= 100.0
        self.filter_curves = curves
        self.filter_labels = labels
        self.filter_units = []
        self.filter_temperatures = temps
        self.filter_2thetas = twothetas
        self.filter_qs = qs
        for xu in xunits:
            if xu == 1:
                self.filter_units.append("Energy Transfer [eV]")
            elif xu == 2:
                self.filter_units.append("Energy [eV]")
            elif xu == 0:
                self.filter_units.append("Detector channels")
            else:
                self.filter_units.append("???")
        self.filter_energies = Ei
        for n,  lab in enumerate(self.filter_labels):
            self.fullnames += ["No file"]
            self.shortnames += [lab]
            self.profiles += [self.filter_curves[n]]
            self.energies += [self.filter_energies[n]]
            self.units += [self.filter_units[n]]
            self.prof_count += 1
        self.mplot_curves = curves
        self.mplot_labels = labels
        self.mplot_override = ["Inverse units", "Fourier Transform"]
        self.overplotworked = True
        self.finished_filter.emit()
        return "Done"
    def multiplot(self):
        self.reduce_profiles()
        nums, Ei, names, temps, twothetas, qs = [], [], [], [], [], []
        for nr in self.matching_numbers:
            rmeas = self.profiles[nr]
            pdict = rmeas.summariseCrucialParts()
            nums.append(nr)
            Ei.append(rmeas.energy)
            names.append(rmeas.shortsource)
            temps.append(pdict['temperature'])
            twothetas.append(pdict['arm_theta'])
            qs.append(pdict['Q'])
        curves = []
        labels = []
        for rn, num in enumerate(nums):
            temp = self.reduced_profiles[rn].copy()
            curves.append(temp)
            labels.append(str(Ei[rn]) + " eV, " + names[rn])
            # step = cur[np.where( cur[:,1] < np.percentile(cur[:,1], 60.0) )][:,1].std()
        self.mplot_curves = curves
        self.mplot_labels = labels
        self.overplotworked = True
        self.finished_overplot.emit()
        return "Done"
    def rixsmap(self):
        self.reduce_profiles(for_rixs=True)
        nums, Ei, names, temps, twothetas, qs = [], [], [], [], [], []
        for nr in self.matching_numbers:
            rmeas = self.profiles[nr]
            pdict = rmeas.summariseCrucialParts()
            nums.append(nr)
            Ei.append(rmeas.energy)
            names.append(rmeas.shortsource)
            temps.append(pdict['temperature'])
            twothetas.append(pdict['arm_theta'])
            qs.append(pdict['Q'])
        curves = []
        labels = []
        for rn, num in enumerate(nums):
            temp = self.reduced_profiles[rn].copy()
            curves.append(temp)
            labels.append(str(Ei[rn]) + " eV, " + names[rn])
            # step = cur[np.where( cur[:,1] < np.percentile(cur[:,1], 60.0) )][:,1].std()
        counter = 0
        for e in Ei:
            if e > 0.0:
                counter += 1
        if not (counter > 0):
            return None
        chan,  en,  entran = 0, 0, 0
        ocurves, curves = [], []
        step = 0.0            
        if self.current_rixsmap == 0:
            xvals =np.array(Ei)
        elif self.current_rixsmap == 1:
            xvals =np.array(temps)
        elif self.current_rixsmap == 2:
            xvals =np.array(twothetas)
        elif self.current_rixsmap == 3:
            xvals =np.array(qs)
        ymin, ymax, ystep = 1e5, -1e5, 100.0
        for rn, num in enumerate(nums):
            temp = self.reduced_profiles[num].copy()
            ocurves.append(temp)
        for n in range(len(ocurves)):
            cxaxis = ocurves[n][:,0]
            cxstep = cxaxis[1:] - cxaxis[:-1]
            cxstep = np.concatenate([cxstep[:1], cxstep])
            ymin = min(ymin, cxaxis.min())
            ymax = max(ymax, cxaxis.max())
            ystep = min(ystep, cxstep.max())
            norm = (ocurves[n][:,1] * cxstep).sum()
            ocurves[n][:,1] /= norm
            shift = np.percentile(ocurves[n][:,1], 75.0) - 0.01
            ocurves[n][:,1] -= shift
            ocurves[n] = ocurves[n][np.where(ocurves[n][:,1] > 0.0)]
            ocurves[n][:,1] *= 100.0
        sequence = np.argsort(xvals)
        # print("RIXSMAP energies: ", xvals)
        # print("RIXSMAP sequence: ", sequence)
        xvals = xvals[sequence]
        crit = np.where(xvals > 0.0)
        # print(crit)
        xvals = xvals[crit]
        for ns, se in enumerate(sequence):
            # print(se)
            if ns in crit[0]:
                curves.append(ocurves[se])
        # curves = curves[crit]
        # now we need to define a grid for the map
        yaxis = np.arange(ymin, ymax+0.1*ystep, ystep)
        npixy = len(yaxis)
        xmin = xvals.min()
        xmax = xvals.max()
        xstep = max((xvals[1:] - xvals[:-1]).min(), 0.05)
        xaxis = np.arange(xmin - xstep, xmax + xstep*1.01, xstep/5.0)
        xmask = np.zeros(len(xaxis))
        npixx = len(xaxis)
        # define the 2D arrays as needed
        map_array = np.zeros((npixy,npixx)).astype(np.float64)
        # extra output: text file
        map_xyz = np.zeros([npixx*npixy, 3])
        # assign correct values
        # mcurves = []
        for n in range(len(curves)):
            xcrit = np.abs(xaxis - xvals[n])
            pos, = np.where(xcrit == xcrit.min())
            target = np.zeros((npixy,2))
            target[:,0] = yaxis.copy()
            print("RIXSMAP curve min/max:",curves[n][:,1].min(),curves[n][:,1].max())
            yvals = merge2curves(curves[n], target)[:,1]
            # mcurves.append(merge2curves(curves[n], target))
            map_array[:,pos] = yvals.reshape(map_array[:,pos].shape)
            print("RIXSMAP pos, posshape: ", pos, map_array[:,pos].shape)
            # map_array[:len(curves[n]),pos] = curves[n][:,1].reshape(map_array[:len(curves[n]),pos].shape)
            xmask[pos] = 1
        # apply smearing
        # return None
        # print("RIXSMAP array min/max:",map_array.min(),map_array.max())
        virt_array = np.zeros(map_array.shape).astype(np.float64)
        weight_array = np.zeros(len(xaxis))
        # smearwidth = 2.0 # meV
        smearwidth = self.smear
        gridstep = xstep/5.0
        width = int(math.ceil(smearwidth/gridstep))
        # print("RIXSMAP step, gridstep, width", xstep, gridstep, width)
        for n in range(len(xaxis)):
            if xmask[n]:
                w_axis = np.zeros(len(xaxis))
                w_axis[n] = 1.0
                neglim, poslim = 0,0
                for s in range(1,width):
                    if n-s >= 0:
                        neglim = s
                        if xmask[n-s]:
                            break
                for s in range(1,neglim):
                    if n-s >= 0:
                        w_axis[n-s] = 1.0 - s/float(neglim)
                for s in range(1,width):
                    if n+s < len(xmask):
                        poslim = s
                        if xmask[n+s]:
                            break
                for s in range(1,poslim):
                    if n+s < len(xmask):
                        w_axis[n+s] = 1.0 - s/float(poslim)
                for s in range(len(xaxis)):
                    if w_axis[s] > 0.0:
                        if xmask[s] > 0.0:
                            virt_array[:,s] = map_array[:,s].copy()
                            weight_array[s] = 1.0
                        else:
                            virt_array[:,s] += w_axis[s]*(map_array[:,n].copy())
                            weight_array[s] += w_axis[s]
        print("RIXSMAP weigth array: ", weight_array)
        for n in range(len(xaxis)):
            if weight_array[n] > 0.0:
                virt_array[:,n] /= weight_array[n]
        # text output
        counter = 0
        for nx in np.arange(npixx):
            for ny in np.arange(npixy):
                map_xyz[counter, 0] = xaxis[nx]
                map_xyz[counter, 1] = yaxis[ny]
                map_xyz[counter, 2] = virt_array[ny, nx]
                counter += 1
        dump = open(self.temp_path + '/' + 'rixsmap_text.xyz', 'w')
        for n in np.arange(len(map_xyz)):
            dump.write(" ".join([str(xxx) for xxx in map_xyz[n]]) + '\n')
        dump.close()
        # plot!
        self.map2D = [virt_array, map_array]
        self.map2Dplotax = [(ymin,ymax), (xmin, xmax)]
        self.rixs_worked = True
        self.finished_rixsmap.emit()
        return "Done"    
    @pyqtSlot()
    def textToClipboard(self):
        # print("This should copy the table to clipboard.")
        result = ""
        rows = []
        for nr in range(self.rowCount()):
            row = []
            for nc in range(self.columnCount()):
                temptext = self.item(nr, nc).text()
                row.append(temptext)
            rows.append(row)
        for r in rows:
            onerow = " ".join(r)
            result += onerow + '\n'
        clip = QApplication.clipboard()
        clip.clear()
        clip.setText(result)
    @pyqtSlot()
    def excelToClipboard(self):
        # print("This should copy the table to clipboard in a format suitable for a spreadsheet.")
        result = ""
        rows = []
        for nr in range(self.rowCount()):
            row = []
            for nc in range(self.columnCount()):
                temptext = self.item(nr, nc).text()
                row.append(temptext)
            rows.append(row)
        for r in rows:
            onerow = "\t".join(r)
            result += onerow + '\n'
        clip = QApplication.clipboard()
        clip.setText(result)
    def add_row(self, rixsmeas):
        self.busy = True
# tabnames = ['Filename', 'Ei (eV)', 'Temperature (K)', '2 theta (deg)',  'Q (1/A)',  'Use it?', 'FWHM', '+/- dFWHM',  'Int.',  '+/- dInt.',  'Centre',  '+/- dCentre']
        # self.table.blockSignals(True)
        temp = len(self.col_order)*[QStandardItem("")]
        pdict = rixsmeas.summariseCrucialParts()
        if self.current_1dplot_axis == 0:
            fitpars = rixsmeas.fitting_params_absEnergy
        elif self.current_1dplot_axis == 1:
            fitpars = rixsmeas.fitting_params_eV
        else:
            fitpars = rixsmeas.fitting_params_channels
        for nn, d in enumerate([rixsmeas.shortsource, rixsmeas.energy, pdict['temperature'], pdict['arm_theta'], pdict['Q'], rixsmeas.active, 
                                           fitpars['fwhm'], fitpars['fwhm_error'], fitpars['area'], fitpars['area_error'], fitpars['centre'], fitpars['centre_error']]):
            intermediate = str(d).strip("()[]'")
            try:
                interm2 = round(float(intermediate), 3)
            except:
                temp[nn] = QStandardItem(intermediate)
            else:
                temp[nn] = QStandardItem(str(interm2))
        chkBoxItem = temp[5]
        chkBoxItem.setCheckable(True)
        if rixsmeas.active:
            chkBoxItem.setCheckState(Qt.CheckState.Checked)
        else:
            chkBoxItem.setCheckState(Qt.CheckState.Unchecked)
        # chkBoxItem.stateChanged.connect(rixsmeas.setActive)
        self.appendRow(temp)
        self.busy = False
        # self.table.blockSignals(False)
        # self.needanupdate.emit()
    @pyqtSlot()
    def clear_table(self):
        for nr in range(0, self.rowCount())[::-1]:
            self.removeRows(nr, 1)
        self.gotvals.emit()
    def return_values(self):
        final = []            
        for nr in range(0,  self.rowCount()):
            for nc in [5]:
                self.useit[nr] = (self.item(nr, nc).checkState() == Qt.CheckState.Checked)
        for nr in range(len(self.useit)):
            if self.useit[nr]:
                rowdata = [nr]
                rowdata += [self.Ei[nr],  self.name[nr], 
                                  self.temperature[nr], self.twotheta[nr], self.Q[nr]]
                final.append(rowdata)
        return final
    @pyqtSlot()
    def update_ticks(self):
        if self.busy:
            return None
        self.busy = True
        for nr in range(0,  self.rowCount()):
            modind = self.item(nr, 5).index().row()
            print(nr,  modind)
            self.profiles[nr].active = (self.item(nr, 5).checkState() == Qt.CheckState.Checked)
                    # self.useit[nr-1] = not self.useit[nr-1]
        self.busy = False
