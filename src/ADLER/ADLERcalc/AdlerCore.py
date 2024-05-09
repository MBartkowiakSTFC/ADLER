
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

from scipy.optimize import leastsq
from scipy.interpolate import interp1d
from scipy.fftpack import rfft, irfft, fftfreq
from PyQt6.QtCore import QThread, QObject, pyqtSignal, pyqtSlot

from ADLER.ADLERcalc.RixsMeasurement import RixsMeasurement
from ADLER.ADLERcalc.ioUtils import WriteProfile, simplify_number_range, resource_path, simple_read
from ADLER.ADLERcalc.fitUtils import polynomial, fit_polynomial, gauss_denum
from ADLER.ADLERcalc.imageUtils import elastic_line, curvature_profile, make_profile,\
                                 apply_offset_to_2D_data, SphericalCorrection,\
                                 make_histogram, make_stripe
#from ADLER.ADLERcalc.spectrumUtils import Shift
from ADLER.ADLERcalc.qtObjects import MergeCurves, ShiftProfilesParallel, Worker, CustomThreadpool
from ADLER.ADLERcalc.arrayUtils import rand_mt, profile_offsets
from ADLER.ADLERdata.AdlerData import DATA

class NewAdlerCore(QObject):
    fittingresult = pyqtSignal(object)
    energyresult = pyqtSignal(object)
    curvatureresult = pyqtSignal(object)
    segmentresult = pyqtSignal(object)
    historesult = pyqtSignal(object)
    logmessage = pyqtSignal(str)
    params_fitting = pyqtSignal(object)
    params_energy = pyqtSignal(object)
    finished_preprocess = pyqtSignal()
    finished_process = pyqtSignal()
    finished_offsets = pyqtSignal()
    finished_2D = pyqtSignal()
    finished_poly = pyqtSignal()
    finished_fft = pyqtSignal()
    finished_calcfft = pyqtSignal()
    did_nothing = pyqtSignal()
    def __init__(self, master, boxes, logger,  progress_bar = None,  thr_exit = None, max_threads = 1, 
                                startpath = expanduser("~")):
        if master is None:
            super().__init__()
        else:
            super().__init__(master)
        self.master = master
        self.parlist = []
        self.pardict = {}
        self.maxthreads = max_threads
        self.log = logger
        self.logmessage.connect(self.log.put_logmessage)
        self.progbar = progress_bar
        self.thread_exit = thr_exit
        self.temp_path = startpath
        self.temp_name = ""
        self.tdb_profile = None
        # self.threadpool = QThreadPool(self)
        # self.threadpool = QThreadPool.globalInstance()
        self.threadpool = CustomThreadpool(MAX_THREADS = max_threads)
        # self.threadpool.setMaxThreadCount(max_threads)
        self.the_object = RixsMeasurement(max_threads = max_threads,
                                          tdb_profile=self.tdb_profile)
        self.data,  self.header,  self.logvals,  self.logvalnames = None,  None,  None,  None
        self.processing_history = []
        self.data2D = None
        self.original_data2D = None
        self.corrected_data2D = None
        self.raw_profile= None
        self.component_profiles = []
        self.component_rawprofiles = []
        self.summed_rawprofile = None
        self.timedata = None
        self.summed_adjusted_rawprofile = None
        self.name_as_segments = []
        self.fft_plots =[]
        self.energies = []
        self.energy_profile = None
        self.fitted_peak_energy = None
        self.fitting_params_energy = None
        self.fitting_textstring_energy = None
        self.fitted_peak_channels = None
        self.fitting_params_channels = None
        self.fitting_textstring_channels = None
        self.plotax,  self.plotax1D = None,  None
        self.nom_eline = None
        self.curvature,  self.curvature_fit,  self.curvature_params = None,  None,  None
        self.segment_plot = None
        self.histogram_plot = None
        self.curvature_corrected = False
        self.fft_applied = False
        self.chan_override = "Channels"
    @pyqtSlot()
    def pass_calibration(self):
        calibration = []
        for nr in range(self.calib_datamodel.rowCount()):
            row = []
            for nc in range(self.calib_datamodel.columnCount()):
                row.append(self.calib_datamodel.item(nr, nc).text())
            calibration.append(row)
        self.calibration = calibration
        self.the_object.assignCalibration(calibration)
        self.the_object.makeEnergyCalibration()
    def assign_calibration(self, datamodel):
        self.calib_datamodel = datamodel
    def assign_boxes(self,  boxes):
        self.boxes = boxes
        for db in boxes:
            names, values = db.returnValues()
            self.parlist += names
            for nam in names:
                self.pardict[nam] = values[nam]
            db.values_changed.connect(self.make_params_visible)
        self.make_params_visible()
    def get_external_params(self, newdict):
        templist = [str(x) for x in self.pardict.keys()]
        for k in self.parlist:
            try:
                val = newdict[k]
            except:
                continue
            else:
                if k in templist:
                    self.pardict[k] = val
    def update_results(self):
        self.data2D = self.the_object.merged_data
        self.timedata = [len(self.the_object.times), self.the_object.times]
        self.raw_profile = self.the_object.profile_channels
        self.energy_profile = self.the_object.profile_eV        
        self.header = self.the_object.textheader
        self.logvals = self.the_object.the_log
        self.logvalnames = [str(x) for x in self.logvals.keys()]
        self.energies = self.the_object.energies
        self.energy = self.the_object.energy
        self.shortnames = self.the_object.shortnames
        try:
            self.plotax = [(1,self.data2D.shape[0]), (1,self.data2D.shape[1])]
            datashape = self.data2D.shape
            self.det_width = datashape[1]
            self.det_length = datashape[0]
        except:
            self.plotax = [(1,2048), (1, 2048)]
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
            elif 'bpp' in k:
                self.bpp = val
            elif 'tdb' in k:
                self.tdb_fact = val
            elif 'cray' in k:
                self.cray = val
            elif 'ffts' in k:
                self.ffts = val
            elif 'poly' in k:
                self.poly = val
            elif 'eline' in k:
                self.eline = val
            elif 'bkg_perc' in k:
                self.bkg_perc = val
            elif 'segsize' in k:
                self.segsize = val
            elif 'eVpair1' in k:
                self.eVpair1 = val
            elif 'eVpair2' in k:
                self.eVpair2 = val
            elif 'redfac' in k:
                self.redfac = val
    def thread_start(self, target_function,  args = []):
        # 1 - create Worker and Thread inside the Form
        self.obj = Worker()  # no parent!
        self.thread = QThread()  # no parent!
        # 2 - Connect Worker`s Signals to Form method slots to post data.
        # self.obj.intReady.connect(self.onIntReady)
        # 3 - Move the Worker object to the Thread object
        self.obj.moveToThread(self.thread)
        # 4 - Connect Worker Signals to the Thread slots
        self.obj.finished.connect(self.thread.quit)
        # 5 - Connect Thread started signal to Worker operational slot method
        self.thread.started.connect(self.obj.run_repeated)
        # custom config
        self.obj.assignFunction(target_function)
        self.obj.assignArgs(args)
        self.obj.set_waittime(0.1)
        #
        if self.progbar is not None:
            self.progbar.setRange(0, len(args))
            self.obj.intReady.connect(self.progbar.setValue)
        # * - Thread finished signal will close the app if you want!
        self.thread.finished.connect(self.thread_exit)
        # 6 - Start the thread
        self.thread.start()
    @pyqtSlot()
    def save_last_params(self, lastfunction = None):
        try:
            source = open(os.path.join(expanduser("~"),'.ADLERcore.txt'), 'w')
        except:
            return None
        else:
            source.write('Lastdir: '+str(self.temp_path) + '\n')
            source.write('Lastfile: '+str(self.temp_name) + '\n')
            for kk in self.parlist:
                source.write(" ".join([str(u) for u in [kk, self.pardict[kk] ]]) + '\n')
            if not lastfunction == None:
                source.write('Last function called: ' + str(lastfunction) + '\n')
            source.close()
    def logger(self, message):
        now = time.gmtime()
        timestamp = ("AdlerCore "
                     +"-".join([str(tx) for tx in [now.tm_mday, now.tm_mon, now.tm_year]])
                     + ',' + ":".join([str(ty) for ty in [now.tm_hour, now.tm_min, now.tm_sec]]) + '| ')
        self.logmessage.emit(timestamp + message)
    def read_tdb_profile(self,  fname = 'BKG_OVERRIDE.txt'):
        try:
            source = open(fname, 'r')
        except:
            self.logger("Did not find an override file BKG_OVERRIDE.txt - using built-in background from June 2020.")
            source = open(DATA._files['Structured_bkg_per_second'], 'r')
        else:
            self.logger("Loading background per channel per second from BKG_OVERRIDE.txt")
        arr = []
        for line in source:
            arr.append([float(x) for x in line.split()])
        source.close()
        self.tdb_profile = np.array(arr)
    def load_files(self,  flist):
        if len(flist) > 0:           
            self.temp_path = os.path.split(flist[0])[0]
            self.name_as_segments = simplify_number_range(flist)
            self.temp_name = "_".join(['Processed']+self.name_as_segments)
            self.the_object = RixsMeasurement(max_threads = self.maxthreads,
                                              tdb_profile=self.tdb_profile)
            self.the_object.loadFiles(flist)
            self.pass_calibration()
            self.the_object.postprocess()
            self.the_object.removeCrays(self.cray)
            self.the_object.subtractBackground(self.bpp)
            self.the_object.subtractDarkCurrent(self.tdb_fact)
            self.the_object.makeSeparateProfiles(redfactor = self.redfac, cuts = self.cuts)
    def adjust_offsets_automatically(self,  tolerance = 10.0):
        profiles,  offsets = [],  [0.0]
        shifted = []
        for prof in self.the_object.individual_profiles_channels:
            profiles.append(prof.copy())
        temp = np.zeros(profiles[0].shape)
        temp[:, 0] = profiles[0][:, 0]
        temp2 = temp.copy()
        shifted.append(profiles[0].copy())
        for prof in profiles[1:]:
            p = prof.copy()
            pfit, pcov, infodict, errmsg, success = leastsq(profile_offsets, [0.0, 0.0, 1.0],
                                                    args = (p, profiles[0], self.threadpool, self.progbar, self.maxthreads), 
                                                    full_output = 1)
            offsets.append(pfit[0])
            # results = shgo(shgo_profile_offsets, [(-25.0, 25.0)], args = (p, profiles[0], self.threadpool, self.progbar, 1), 
            #                     sampling_method = 'sobol'
            #                     )
            # shift = results['x'][0]
            # offsets.append(shift)
            p[:, 0] -= pfit[0]
            shifted.append(p)
#         runner = ShiftProfilesParallel(profiles, mthreads = self.maxthreads)
#         runner.runit()
#        offsets = [0.0] + runner.shiftlist
        # for n, prof in enumerate(profiles[1:]):
        #     p = prof.copy()
        #     off = offsets[n+1]
        #     p[:, 0] -= off
        #     shifted.append(p)
        self.component_profiles = shifted
        self.component_rawprofiles = profiles
        for p in profiles:
            temp[:, 1] += p[:, 1]
        for s in shifted:
            # the_object = MergeCurves(s, temp2,  tpool = self.threadpool, pbar = self.progbar)
            the_object = MergeCurves(s, temp2,  tpool = None, pbar = self.progbar,  mthreads = self.maxthreads)
            the_object.runit()
            temp2 = the_object.postprocess()
            # temp2 = merge2curves(s, temp2)
        self.summed_rawprofile = temp
        self.summed_adjusted_rawprofile = temp2
        return offsets,  shifted
    def adjust_offsets_expensive(self,  tolerance = 10.0):
        profiles,  offsets = [],  [0.0]
        shifted = []
        for prof in self.the_object.individual_profiles_channels:
            profiles.append(prof.copy())
        temp = np.zeros(profiles[0].shape)
        temp[:, 0] = profiles[0][:, 0]
        temp2 = temp.copy()
        shifted.append(profiles[0].copy())
#        for prof in profiles[1:]:
#            p = prof.copy()
#            #pfit, pcov, infodict, errmsg, success = leastsq(profile_offsets, [0.0, 1.0, 0.0],
#            #                                                                       args = (p, profiles[0], self.threadpool, self.progbar, self.maxthreads), 
#            #                                                                       full_output = 1)
#            #offsets.append(pfit[0])
#            results = shgo(shgo_profile_offsets, [(-25.0, 25.0)], args = (p, profiles[0], self.threadpool, self.progbar, 1), 
#                                sampling_method = 'sobol'
#                                )
#            shift = results['x'][0]
#            offsets.append(shift)
#            p[:, 0] -= shift
#            shifted.append(p)
        runner = ShiftProfilesParallel(profiles, mthreads = self.maxthreads)
        runner.runit()
        offsets = [0.0] + runner.shiftlist
        for n, prof in enumerate(profiles[1:]):
            p = prof.copy()
            off = offsets[n+1]
            p[:, 0] -= off
            shifted.append(p)
        self.component_profiles = shifted
        self.component_rawprofiles = profiles
        for p in profiles:
            temp[:, 1] += p[:, 1]
        for s in shifted:
            # the_object = MergeCurves(s, temp2,  tpool = self.threadpool, pbar = self.progbar)
            the_object = MergeCurves(s, temp2,  tpool = None, pbar = self.progbar,  mthreads = self.maxthreads)
            the_object.runit()
            temp2 = the_object.postprocess()
            # temp2 = merge2curves(s, temp2)
        self.summed_rawprofile = temp
        self.summed_adjusted_rawprofile = temp2
        return offsets,  shifted
    @pyqtSlot(object)
    def apply_offsets(self,  offsets):
        self.the_object.mergeProfiles(offsets)
        self.the_object.mergeArrays(offsets)
        self.finished_offsets.emit()
    @pyqtSlot(object)
    def justload_manyfiles(self, flist):
        self.load_files(flist)
        offsets = np.zeros(len(flist))
        self.apply_offsets(offsets)
        self.offsets = offsets
        self.update_results()
        self.finished_2D.emit()
        self.save_last_params()
    @pyqtSlot(object)
    def preprocess_manyfiles(self, flist):
        self.load_files(flist)
        off,  shift = self.adjust_offsets_automatically()
        self.offsets = off
        self.individual_profiles = self.the_object.individual_profiles_channels
        fnames = self.the_object.shortnames
        energies = self.the_object.energies.copy().ravel()
        self.individual_labels = []
        for n in range(len(fnames)):
            self.individual_labels.append(str(round(energies[n], 3)) + " eV, " + fnames[n])
        self.finished_preprocess.emit()
        self.save_last_params()
    @pyqtSlot(object)
    def expensive_merge_manyfiles(self, flist):
        self.load_files(flist)
        off,  shift = self.adjust_offsets_expensive()
        self.offsets = off
        self.apply_offsets(off)
        self.update_results()
        self.finished_process.emit()
        self.finished_2D.emit()
        self.save_last_params()
    @pyqtSlot(object)
    def process_manyfiles(self, flist):
        self.load_files(flist)
        off,  shift = self.adjust_offsets_automatically()
        self.offsets = off
        self.apply_offsets(off)
        self.update_results()
        self.finished_process.emit()
        self.finished_2D.emit()
        self.save_last_params()
    @pyqtSlot()
    def generate_mock_dataset(self):
        self.the_object = RixsMeasurement(tdb_profile=self.tdb_profile)
        self.the_object.fakeData(self.cuts, self.bpp)
        self.offsets = [0.0]
        self.the_object.postprocess()
        self.update_results()
        self.finished_process.emit()
        self.finished_2D.emit()
    @pyqtSlot()
    def finalise_manyfiles(self):
        self.apply_offsets(self.offsets)
        self.update_results()
        self.original_data2D = self.the_object.merged_data.copy()
        self.data2D = self.original_data2D
        self.finished_2D.emit()
    def curve_profile(self):  
        lcut,  hcut = self.cuts[0:2]
        if self.data2D is not None:
            curve = curvature_profile(self.data2D, blocksize = self.segsize, percentile = self.bkg_perc,
                                                olimits = self.eline)
            pfit, pcov, infodict, errmsg, success = leastsq(fit_polynomial, [curve[:,1].mean(), 0.0, 0.0], args = (curve,), full_output = 1)
            curvefit = polynomial(pfit, curve[:,0])
            curvefit = np.column_stack([curve[:,0], curvefit])
            self.curvature = curve
            self.curvature_fit = curvefit
            self.curvature_params = pfit
            self.curvatureresult.emit([curve,  curvefit,  pfit])
        else:
            self.did_nothing.emit()
    def calculate_fft(self):
        if self.data2D is None:
            self.logger("No data has been loaded for curvature correction")
            self.did_nothing.emit()
            return None
        tempdata = rfft(self.data2D)
        fft_min, fft_max, fft_mean, fft_sum = tempdata.min(0), tempdata.max(0), tempdata.mean(0), np.abs(tempdata).sum(0)
        x_freqs = fftfreq(self.data2D.shape[1],1.0)
        plots = []
        for nx in [fft_min, fft_max, fft_mean, fft_sum]:
            tempval = nx / nx.sum()
            plots.append(np.column_stack([x_freqs,tempval]))
        self.fft_plots = plots
        self.finished_calcfft.emit()
        return "Done"
    @pyqtSlot()
    def correct_fft(self):
        self.fft_applied = False
        fftmin,  fftmax = self.ffts
        if fftmin is None or fftmax is None:
            self.did_nothing.emit()
            self.finished_fft.emit()
            return None
        if fftmin < -1e5 or fftmax < -1e5:
            self.logger("The FFT filter limits need to be set first!")
            self.finished_fft.emit() 
            self.did_nothing.emit()
            return None
        if self.data2D is None:
            self.logger("No data is available to be processed by the FFT filter")
            self.finished_fft.emit()
            self.did_nothing.emit()
            return None
        else:
            data = self.data2D
        tempdata = rfft(data) 
        x_freqs = fftfreq(data.shape[1],1.0)
        m1 = np.argmax(x_freqs > fftmin)
        m2 = np.argmax(x_freqs > fftmax)
        filler = interp1d(np.concatenate([x_freqs[m1-3:m1], x_freqs[m2:m2+3]]),
                                np.column_stack([tempdata[:, m1-3:m1], tempdata[:,m2:m2+3]]),
                               kind = "slinear")
        # filler = scint.interp1d([x_freqs[m1-3:m1].mean(), x_freqs[m2:m2+3].mean()], 
        #             np.column_stack([tempdata[:, m1-3:m1].mean(1), tempdata[:,m2:m2+3].mean(1)]))
        tempdata[:,m1:m2] = filler(x_freqs[m1:m2])
        print("Now the general smoothing")
        for i in range(tempdata.shape[0]):
            tempdata[i,-3:] = tempdata[i,-3:].mean()
        result = irfft(tempdata)
        self.corrected_data2D = result
        self.data2D = self.corrected_data2D
        self.processing_history.append(['FFTFilterApplied',  fftmin,  fftmax])
        self.fft_applied = True
        self.finished_fft.emit()
        return "Done"
    @pyqtSlot()
    def apply_poly(self):
        self.curvature_corrected = False
        data = self.data2D
        if data is None:
            self.logger("No data available to which curvature correction could be applied.")
            self.finished_poly.emit()
            self.did_nothing.emit()
            return None
        lcut,  hcut = self.cuts[0:2]
        pmin,  pmax = self.eline
        poly = self.poly
        perc = self.bkg_perc
        if poly is not None:
            if poly[0] < -99.0 and poly[1] < -99.0 and poly[2] < -99.0:
                self.logger("Please specify some realistic parameters for the curvature correction.")
                self.finished_poly.emit()
                self.did_nothing.emit()
                return None
            elif poly[0] > 5.0 and poly[1] >5.0 and poly[2] > 5.0:
                curve = curvature_profile(data[pmin:pmax], blocksize = 1, percentile = perc, override = None, olimits = (0, pmax-pmin))
                curve[:,0] -= 0.5
                newcurve = interp1d(curve[:,0], curve[:,1], fill_value = "extrapolate")(np.arange(lcut,hcut))
                data = SphericalCorrection(data, poly, locut = lcut, hicut = hcut, direct_offsets = newcurve)
            else:
                data = SphericalCorrection(data, poly, locut = lcut, hicut = hcut)
            self.data2D = data
            self.processing_history.append(['CurvatureCorrectionApplied',  poly[0],  poly[1],  poly[2]])
            self.curvature_corrected = True
            self.finished_poly.emit()
            return "Done"
    def make_stripe(self):
        self.segment_plot = None
        if self.data2D is None:
            self.logger("No data has been loaded for curvature correction")
            self.did_nothing.emit()
            return None
        if self.eline[0] < 0 or self.eline[1] < self.eline[0]:
            self.logger("The elastic line limits are ill-defined. Not processing further.")
            self.did_nothing.emit()
            return None
        self.segment_plot = make_stripe(self.data2D, self.cuts, self.eline)
        self.segmentresult.emit([self.segment_plot,  [(self.eline[0],  self.eline[1]), (self.cuts[0]+1, self.cuts[1])]])
        return self.segment_plot,  [(self.eline[0],  self.eline[1]), (self.cuts[0]+1, self.cuts[1])]
    def make_histogram(self):
        self.histogram_plot = None
        if self.data2D is None:
            self.logger("No data has been loaded for histogram generation.")
            self.did_nothing.emit()
            return None
        if self.eline[0] < 0 or self.eline[1] < self.eline[0]:
            self.logger("The elastic line limits are ill-defined. Not processing further.")
            self.did_nothing.emit()
            return None
        self.histogram_plot = make_histogram(self.data2D, self.cuts, self.eline)
        self.historesult.emit([self.histogram_plot,  [(self.eline[0],  self.eline[1]), (self.cuts[0]+1, self.cuts[1])]])
        vals,  bins = self.histogram_plot
        xvals = (bins[1:] + bins[:-1])*0.5
        temphist = np.column_stack([xvals,  vals])
        WriteProfile(os.path.join(self.temp_path, self.temp_name+"_HISTOGRAM.txt"), temphist)
        return self.histogram_plot,  [(self.eline[0],  self.eline[1]), (self.cuts[0]+1, self.cuts[1])]
    @pyqtSlot()
    def autoprocess_file(self,  isfinal = True):
        self.process_file(guess = True,  final = isfinal)
    @pyqtSlot()
    def process_file(self,  guess = False,  final = True):  
        if self.data2D is None:
            self.did_nothing.emit()
            return None,  None, None, None, None
        self.the_object.makeSeparateProfiles(redfactor = self.redfac, cuts = self.cuts)
        self.the_object.mergeProfiles(self.offsets)
        if guess:
            self.the_object.fitPeak(redfac = self.redfac)
        else:
            self.the_object.fitPeak(manual_limits = self.eline, redfac = self.redfac)
        self.nom_eline = float(self.the_object.nominal_elastic_line_position)
        if self.the_object.mev_per_channel > 0.0:
            # self.chan_override = "Channels, " + str(round(self.the_object.mev_per_channel, 3)) + " meV/channel"
            # thanks for changing your mind every 20 minutes. It's gone now.
            self.chan_override = "Channels"
        else:
            self.chan_override = "Channels"
        self.raw_profile = self.the_object.profile_channels.copy()
        self.plotax1D = [(self.cuts[2],self.cuts[3]), (self.cuts[0]+1, self.cuts[1]+1)]
        self.the_object.write_extended_ADLER(os.path.join(self.temp_path, self.temp_name + '_extended1D.txt'))
        self.the_object.write_yaml(os.path.join(self.temp_path, self.temp_name))
        # self.the_object.write_yaml(os.path.join(self.temp_path, self.temp_name), full_output = True, compressed = True)
        # WriteProfile(os.path.join(self.temp_path, self.temp_name+"_1D.txt"), self.raw_profile,
        #                 header = self.header.split('\n'), params = [self.parlist,  self.pardict], 
        #                 varlog = [self.logvalnames,  self.logvals])
        self.fitted_peak_channels = self.the_object.fitted_peak_channels
        self.fitting_textstring_channels = self.the_object.fitting_textstring_channels
        tempdict = self.the_object.fitting_params_channels
        self.fitting_params_channels = [[tempdict['maxval'], tempdict['fwhm'], tempdict['centre']], 
                                                       [tempdict['maxval_error'], tempdict['fwhm_error'], tempdict['centre_error']]]
        peak_area = tempdict['area']
        peak_area_error = tempdict['area_error']
        fit = self.fitting_params_channels # this is not a mistake. 
        fitstring = self.the_object.fitting_textstring_channels
        bkg = self.the_object.background_channels
        peak = self.fitted_peak_channels.copy()
        tempthing = self.raw_profile.copy()
        if final:
            self.fittingresult.emit([tempthing,  bkg, peak,  fit,  fitstring])
            if fit is not None:
                self.params_fitting.emit({ 'FIT_centre':np.array([round(fit[0][2],3), abs(round(fit[1][2],3))]), 
                                                   'FIT_fwhm':np.array([abs(round(fit[0][1],3)), abs(round(fit[1][2],3))]), 
                                                   'FIT_area':np.array([round(peak_area,3), abs(round(peak_area_error,3))])})
        return tempthing,  bkg, peak,  fit,  fitstring
    @pyqtSlot()
    def auto_eV_profile(self):
        self.eV_profile(guess = True)
    @pyqtSlot()
    def eV_profile(self,  guess = False): 
        if self.data2D is None:
            self.logger("There is no file to be processed.")
            self.did_nothing.emit()
            return None, None,  None, None,  None
        emin,  emax = self.eline
        eguess = self.the_object.eline()
        if (eguess is None):
            self.logger("Invalid energy range, skipping the calculation")
            self.did_nothing.emit()
            return None, None,  None, None,  None
        else:
            self.the_object.makeSeparateProfiles(redfactor = self.redfac, cuts = self.cuts)
            self.the_object.mergeProfiles(self.offsets)
            if guess:
                self.the_object.fitPeak(redfac = self.redfac)
            else:
                self.the_object.fitPeak(manual_limits = self.eline, redfac = self.redfac)
            self.the_object.makeEnergyProfile()
            self.the_object.makePeakInEnergy()
            self.energy_profile = self.the_object.profile_eV.copy()
            # profi, back, peak, fit = self.the_object # I still need to find the right parts of the data structure.
            peak = self.the_object.fitted_peak_eV
            back = self.the_object.background_eV
            temp = self.the_object.fitting_params_eV
            fitstring = self.the_object.fitting_textstring_eV
            fit = [[temp['maxval'], temp['fwhm'], temp['centre']],
                     [temp['maxval_error'], temp['fwhm_error'], temp['centre_error']]]
            peak_area, peak_area_error = temp['area'], temp['area_error']
            self.logger('heigth, FWHM, centre, baseline')
            self.logger(str(fit[0]))
            self.logger(str(fit[1]))
            # self.energy_profile = profi              
            self.the_object.write_extended_ADLER(os.path.join(self.temp_path, self.temp_name + '_extended1D.txt'))
            self.the_object.write_yaml(os.path.join(self.temp_path, self.temp_name))
            # self.the_object.write_yaml(os.path.join(self.temp_path, self.temp_name), full_output = True, compressed = True)
            # WriteProfile(os.path.join(self.temp_path, self.temp_name+"_1D_deltaE.txt"), self.energy_profile,
            #             header = self.header.split('\n'), params = [self.parlist,  self.pardict], 
            #             varlog = [self.logvalnames,  self.logvals])
            self.fitted_peak_energy = peak
            self.fitting_params_energy = fit
            self.fitting_textstring_energy = fitstring
            self.energyresult.emit([self.energy_profile.copy(), back, peak, fit,  fitstring])
            self.params_energy.emit({  'ENERGY_centre':np.array([0.0, abs(round(1000.0*fit[1][2],3))]), 
                                                    'ENERGY_fwhm': np.array([abs(round(1000.0*fit[0][1],3)), abs(round(1000.0*fit[1][1],3))]), 
                                                    'ENERGY_area':np.array([round(peak_area,3), abs(round(peak_area_error,3))])})
            return self.energy_profile, back, peak, fit,  fitstring
