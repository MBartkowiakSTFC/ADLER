
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
This part of the ADLER GUI is designed for finding the correction parameters
for 2D images. The way things have been going so far, it is
hardly ever used.
"""

import os
import time
import copy
from os.path import expanduser

import numpy as np
from PyQt6.QtCore import pyqtSlot, pyqtSignal,  QSize,  QThread
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QSizePolicy, QWidget, QFileDialog,   \
                                                QPushButton,  QVBoxLayout, QHBoxLayout, QFormLayout, QScrollArea
# from PyQt5 import sip
from VariablesGUI import VarBox
from ADLERcalc.AdlerCore import NewAdlerCore as AdlerCore
from ExtendGUI import AdlerTab
from ADLERplot.Plotter import Plotter

mpl_scale = 1.0
mpl_figure_scale = 1.0
font_scale = 1.0
last_dir_saved = expanduser("~")
MAX_THREADS = 1

try:
    source = open(os.path.join(expanduser("~"),'.ADLconfig.txt'), 'r')
except:
    pass
else:
    for line in source:
        toks = line.split()
        if len(toks) > 1:
            if toks[0] == 'Matplotlib_scale:':
                try:
                    mpl_scale = float(toks[1])
                except:
                    pass
            if toks[0] == 'Matplotlib_figure_scale:':
                try:
                    mpl_figure_scale = float(toks[1])
                except:
                    pass
            if toks[0] == 'Font_scale:':
                try:
                    font_scale = float(toks[1])
                except:
                    pass
    source.close()


# this is a Windows thing
# ctypes.windll.kernel32.SetDllDirectoryW('.')

# simple mathematical functions are defined here

GlobFont = QFont('Sans Serif', int(12*font_scale))

oldval = 0.0

#### GUI part

loading_variables = [
{'Name': 'Detector cutoff',  'Unit':'pixel',  'Value':np.array([0, 2048]),  'Key' : 'cuts', 
                               'MinValue':np.array([0, 0]),  'MaxValue':np.array([2048, 2048]),  'Length': 2,  'Type':'int',
                               'Comment':'The pixel columns outside of the limits will be removed from the detector image on loading.'}, 
{'Name': 'Background per pixel',  'Unit':'counts',  'Value':960.0,   'Key' : 'bpp', 
                               'MinValue':0.0,  'MaxValue':1e5,  'Length': 1,  'Type':'float',
                               'Comment':'This constant will be subtracted from the total counts in every pixel of the detector. \nThe default value is based on the readout noise for the standard parameters of the PEAXIS detector.'},
{'Name': 'Time-dependent background',  'Unit':'fraction',  'Value':0.0,   'Key' : 'tdb', 
                               'MinValue':0.0,  'MaxValue':1e3,  'Length': 1,  'Type':'float',
                               'Comment':'A curve determined by dark-current measurements will be subtracted from the detector counts. \nThis is scaled by the total measurement time. Normally this parameter should be set to 1.'},
{'Name': 'Cosmic ray correction factor',  'Unit':'StdDev',  'Value':3,  'Key' : 'cray', 
                               'MinValue':-10.0,  'MaxValue':1e5,  'Length': 1,  'Type':'float',
                               'Comment':'For each horizontal line of pixels, the count values will be replaced with the line average if they are more than \nFACTOR x StdDev away from the average. Lower factor means more count values removed from the detector image.'}
]
line_variables = [
{'Name': 'Elastic line limits',  'Unit':'pixel',  'Value':np.array([0, 2048]),   'Key' : 'eline', 
                               'MinValue':np.array([-10, -10]),  'MaxValue':np.array([2048, 2048]),  'Length': 2,  'Type':'int',
                               'Comment':'Limits for manual fitting of the elastic line position, AND for curvature fitting. \nSetting BOTH limits to NEGATIVE values will make ADLER output the spectrum in\n Absolute Energy (eV) units, if the energy calibration is defined and valid.'},   
{'Name': 'Detection limit for BKG',  'Unit':'percentile',  'Value':75,  'Key' : 'bkg_perc', 
                               'MinValue':0.0,  'MaxValue':100.0,  'Length': 1,  'Type':'float',
                               'Comment':'The average of the y values of the curve up to this percentile \nwill be used as a fixed, constant background in the elastic line fitting.'}, 
{'Name': 'Curvature segment size',  'Unit':'pixel',  'Value':16, 'Key' : 'segsize', 
                               'MinValue':1,  'MaxValue':1024,  'Length': 1,  'Type':'int',
                               'Comment':'The detector image is separated into vertical strips of\n N pixels for elastic line curvature fitting, where N is the value specified here.'}, 
]
energy_variables = [
{'Name': 'Energy calibration pair 1',  'Unit':'pixel, eV',  'Value':np.array([-1.0, -1.0]), 'Key' : 'eVpair1', 
                                      'MinValue':-1e5*np.ones(2),
                                      'MaxValue':1e5*np.ones(2),
                                      'Length': 2,  'Type':'float',
                               'Comment':'The position in pixels of the elastic line at the specified photon energy. Defines the energy scale of the detector.'}, 
{'Name': 'Energy calibration pair 2',  'Unit':'pixel, eV',  'Value':np.array([-1.0, -1.0]), 'Key' : 'eVpair2', 
                                      'MinValue':-1e5*np.ones(2),
                                      'MaxValue':1e5*np.ones(2),
                                      'Length': 2,  'Type':'float',
                               'Comment':'The position in pixels of the elastic line at the specified photon energy. Defines the energy scale of the detector.'}, 
]
correction_variables = [
{'Name': 'Curvature Correction',  'Unit':'N/A',  'Value':-100.0*np.ones(3), 'Key' : 'poly', 
                                      'MinValue':-1e15*np.ones(3),
                                      'MaxValue':1e15*np.ones(3),
                                      'Length': 3,  'Type':'float',
                               'Comment':'The parameters of the quadratic function defining the curvature of the elastic line. \nAll the pixels of the 2D detector image will be shifted to make the elastic line straight (i.e. horizontal) using these parameters.'}, 
{'Name': 'Removed FFT region',  'Unit':'N/A',  'Value':-1e9*np.ones(2), 'Key' : 'ffts', 
                                      'MinValue':-1e15*np.ones(2),
                                      'MaxValue':1e15*np.ones(2),
                                      'Length': 2,  'Type':'float',
                               'Comment':'This region of the Fourier transform of the detector counts as a function of position will be removed and interpolated'}, 
{'Name': 'Reduction factor',  'Unit':'N/A',  'Value':1.0, 'Key' : 'redfac', 
                                      'MinValue':0.25,
                                      'MaxValue':10.0,
                                      'Length': 1,  'Type':'float',
                               'Comment':'Higher values of reduction factor lead to coarser binning of the data, \nwhich can be useful for plotting noisy data sets which do not require high resolution.'}, 
]

class CorrectionsTab(AdlerTab):
    for_preprocess = pyqtSignal(object)
    for_process = pyqtSignal(object)
    for_simplemerge = pyqtSignal(object)
    for_shiftedmerge = pyqtSignal()
    for_poly = pyqtSignal()
    for_fft = pyqtSignal()
    for_fake = pyqtSignal()
    def __init__(self, master,  canvas,  log,  mthreads = 1, startpath = None,  app = None):
        super().__init__(master)
        self.master = master
        self.canvas, self.figure, self.clayout = canvas
        self.params = [(loading_variables, "File Loading"),  (line_variables,  "Elastic Line"),
                               (energy_variables,  "Energy Calibration"),  (correction_variables,  "Corrections")]
        self.plotter = Plotter(figure = self.figure)
        self.parnames = []
        self.pardict = {}
        self.log = log
        self.filelist = None
        self.core = AdlerCore(None, None,  self.log,  max_threads = mthreads, 
                                        thr_exit = self.unblock_interface,  startpath = startpath, 
                                        progress_bar = self.progbar)
        self.currentpath = startpath
        self.boxes = self.make_layout()
        self.core.assign_boxes(self.boxes)
        self.core.fittingresult.connect(self.plot_fitting_results)
        self.core.energyresult.connect(self.plot_energy_results)
        self.core.curvatureresult.connect(self.plot_curve_profile)
        self.core.segmentresult.connect(self.plot_segment)
        self.core.historesult.connect(self.plot_histogram)
        self.core.finished_preprocess.connect(self.show_1D_comparison)
        # self.core.finished_preprocess.connect(self.trigger_logplotter)
        self.core.finished_2D.connect(self.show_2D_plot)
        # self.core.finished_process.connect(self.trigger_logplotter)
        self.core.finished_poly.connect(self.after_ccorr)
        self.core.finished_fft.connect(self.after_fft)
        self.core.finished_calcfft.connect(self.show_fft)
        #
        self.for_preprocess.connect(self.core.preprocess_manyfiles)
        self.for_process.connect(self.core.process_manyfiles)
        self.for_simplemerge.connect(self.core.justload_manyfiles)
        self.for_shiftedmerge.connect(self.core.finalise_manyfiles)
        self.for_poly.connect(self.core.apply_poly)
        self.for_fft.connect(self.core.correct_fft)
        self.for_fake.connect(self.core.generate_mock_dataset)
        #
        self.core.fittingresult.connect(self.flip_buttons)
        self.core.energyresult.connect(self.flip_buttons)
        self.core.curvatureresult.connect(self.flip_buttons)
        self.core.segmentresult.connect(self.flip_buttons)
        self.core.finished_preprocess.connect(self.flip_buttons)
        self.core.finished_preprocess.connect(self.flip_buttons)
        self.core.finished_2D.connect(self.flip_buttons)
        self.core.finished_process.connect(self.flip_buttons)
        self.core.finished_poly.connect(self.flip_buttons)
        self.core.finished_fft.connect(self.flip_buttons)
        self.core.finished_calcfft.connect(self.flip_buttons)
        self.core.did_nothing.connect(self.flip_buttons)
        #
        self.corethread = QThread()
        if app is not None:
            app.aboutToQuit.connect(self.corethread.quit)
            app.aboutToQuit.connect(self.cleanup)
        self.base.destroyed.connect(self.corethread.quit)
        self.core.read_tdb_profile()
        self.core.moveToThread(self.corethread)
        self.corethread.start()
    def make_layout(self):
        #base = QWidget(self.master)
        # base.setStyleSheet("background-color:rgb(240,220,220)")
        #self.base=base
        base = self.base
        base.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        base_layout = QHBoxLayout(base)
        base_layout.addWidget(self.canvas)
        boxes_base = QWidget(base)
        boxes_layout = QVBoxLayout(boxes_base)
        scroll = QScrollArea(widgetResizable=True)
        scroll.setWidget(boxes_base)
        base_layout.addWidget(scroll)
        # base_layout.addWidget(boxes_base)
        button_base = QWidget(base)
        button_layout = QFormLayout(button_base)
        button_layout.setVerticalSpacing(2)
        button_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        boxes_layout.addWidget(button_base)
        # self.progbar = QProgressBar(base)
        boxes = []
        for el in self.params:
            if el[1] == "Corrections":
                temp = VarBox(boxes_base, el[0],  el[1],  prec_override=9)
            else:
                temp = VarBox(boxes_base, el[0],  el[1])
            boxes.append(temp)
            boxes_layout.addWidget(temp.base)
            # temp.values_changed.connect(self.read_inputs)
        boxes_layout.addWidget(self.progbar)
        # structure of vars: label, dictionary keys, tooltip
        col1 = "background-color:rgb(30,180,20)"
        col2 = "background-color:rgb(0,210,210)"
        col3 = "background-color:rgb(50,150,250)"
        #
        button_list= [
        ['Choose Input Files', self.load_file_button, 'Pick the binary file to be processed.', 
            col1, 'Files'], # 0
        ['Reload!', self.reload_file, 'Process the same file again.', 
            col1, 'Files'], # 1
        ['Get Parameters', self.load_params_from_file, 'Read processing parameters from an existing output file.', 
            '', 'Files'], # 2
        ['Fake Data', self.fake_data_button, 'Create a dataset made of random numbers', 
            col1, 'Files'], # 3
        ['Fast Fit', self.core.autoprocess_file, 'Create a 1D profile, guess line position.', 
            col1, 'Processing'], # 4
        ['Make Profile', self.core.process_file, 'Create a 1D profile.', 
            col1, 'Processing'], # 5
        ['Fourier Transform', self.core.calculate_fft, 'Calculate the Fourier transform of the data.', 
            col2, 'Processing'], # 6
        ['Fit Curvature', self.core.curve_profile, 'Fit the curvature of the elastic line.', 
            col2, 'Processing'], # 7
        ['Plot Horizontal', self.core.make_stripe, 'Plot the intensity profile of the elastic line across the detector.', 
            col1, 'QC'], # 8
        ['Direct Merge', self.merge_files_simple, 'Stack up all the files, pixel per pixel.', 
            col1, 'Merging'], # 9
        ['Shifted Merge', self.merge_files_offsets, 'Shift the files relative to each other to find the best match.', 
            col1, 'Merging'], # 10
        ['Correct Curvature', self.correct_curvature, 'Shift the pixels vertically, column per column, to eliminate the curvature.', 
            col1, 'Corrections'], # 11
        ['FFT Filter', self.fft_filter, 'Filter out a part of the frequency spectrum of the data. AKA "Remove Stripes".', 
            col1, 'Corrections'], # 12
        ['Histogram of counts', self.core.make_histogram, 'Plot the distribution of the pixel counts in the selected detector area.', 
            col1, 'QC'], # 13
        ]
        self.active_buttons = np.zeros(len(button_list)).astype(int)
        self.active_buttons[0] = 1
        self.active_buttons[2:4] = 1
        self.button_list = []
        button_dict = {}
        for bl in button_list:
            temp = self.MakeButton(button_base, bl[0],  bl[1],  bl[2])
            self.button_list.append(temp)
            if bl[3]:
                temp.setStyleSheet(bl[3])
            if bl[4] in button_dict.keys():
                button_dict[bl[4]].append(temp)
            else:
                button_dict[bl[4]] = [temp]
        for k in button_dict.keys():
            bbase = QWidget(button_base)
            blayout = QHBoxLayout(bbase)
            for but in button_dict[k]:
                blayout.addWidget(but)
            button_layout.addRow(k,  bbase)
        self.button_base = button_base
        self.boxes_base = boxes_base
        self.flip_buttons()
        return boxes
    @pyqtSlot()
    def cleanup(self):
        self.corethread.quit()
        self.corethread.wait()
#    def fake_data_button(self):
#        self.core.generate_mock_dataset()
#        plot2D_sliders(self.core.data2D, self.core.plotax, fig = self.figure)
#        # self.filelist = result
#        self.active_buttons[0:9] = 1
#        self.active_buttons[9:11] = 0
#        self.active_buttons[11:13] = 1
#        self.flip_buttons()
    def fake_data_button(self):
        self.block_interface()
        self.for_fake.emit()
        self.active_buttons[0:9] = 1
        self.active_buttons[9:11] = 0
        self.active_buttons[11:14] = 1
    @pyqtSlot()
    def show_1D_comparison(self):
        profs = [self.core.summed_rawprofile,  self.core.summed_adjusted_rawprofile]
        self.plotter.plot1D(profs, fig = self.figure, text = "Pick the better profile!", 
            label_override = ['Channels',  'Counts'], curve_labels = ['Simple Merge',  'Shifted Merge'])
    @pyqtSlot()
    def show_2D_plot(self):
        self.plotter.plot2D_sliders(self.core.data2D, self.core.plotax, fig = self.figure)
    @pyqtSlot(object)
    def plot_fitting_results(self,  fittinglist):
        profi, back, peak, fitpars, text = fittinglist
        if profi is None:
            self.logger('No data available to be processed.')
            return None
        elif back is None or peak is None or text is None:
            self.plotter.plot1D([profi], fig = self.figure, text = "", 
                  label_override = ['Channels',  'Counts'], curve_labels = ['Data'])
        else:
            self.plotter.plot1D([profi,  back,  peak], fig = self.figure, text = text, 
                  label_override = ['Channels',  'Counts'], curve_labels = ['Data',  'Background', 'Fit'])
    @pyqtSlot(object)
    def plot_energy_results(self, fittinglist):
        profi, back, peak, fitpars, text = fittinglist
        if profi is None:
            self.logger('No data available to be processed.')
            return None
        elif back is None or peak is None or text is None:
            self.plotter.plot1D([profi], fig = self.figure, text = "", 
                  label_override = ['Energy [eV]',  'Counts'], curve_labels = ['Data'])
        else:
            self.plotter.plot1D([profi,  back,  peak], fig = self.figure, text = text, 
                  label_override = ['Energy transfer [eV]',  'Counts'], curve_labels = ['Data',  'Background', 'Fit'])
    @pyqtSlot(object)
    def plot_curve_profile(self,  curvelist):
        if self.core.curvature is not None:
            curve,  fit,  fitparam = curvelist
            starty,  endy = 1,  2048
            if self.core.eline[0] >0:
                starty = self.core.eline[0]
            if self.core.eline[1] >0:
                endy = self.core.eline[1]
            # plotax = [(starty,  endy), (self.core.cuts[0],  self.core.cuts[1])]
            self.plotter.plot1D([curve,  fit], fig = self.figure, text = "", 
                  label_override = ['Channels',  'Channels'], curve_labels = ['Elastic line position', 'Curvature fit'])
            rkey= ['poly']
            rval = {'poly':self.core.curvature_params}
            for b in self.boxes:
                b.takeValues(rkey, rval)
    def logger(self, message):
        now = time.gmtime()
        timestamp = ("-".join([str(tx) for tx in [now.tm_mday, now.tm_mon, now.tm_year]])
                     + ',' + ":".join([str(ty) for ty in [now.tm_hour, now.tm_min, now.tm_sec]]) + '| ')
        self.log.setReadOnly(False)
        self.log.append("CORRECTIONS :" + timestamp + message)
        self.log.setReadOnly(True)
    def MakeButton(self, parent, text, function, tooltip = ""):
        button = QPushButton(text, parent)
        if tooltip:
            button.setToolTip(tooltip)
        button.clicked.connect(function)
        button.setSizePolicy(QSizePolicy.Policy.Preferred,QSizePolicy.Policy.Expanding)
        button.setMinimumSize(QSize(48, 25))
        # button.setMaximumSize(QSize(300, 100))
        button.setFont(GlobFont)
        return button
    def load_file_button(self):
        result, ftype = QFileDialog.getOpenFileNames(self.master,
           'Load data from one or more Andor camera files:', self.currentpath,
           'Andor binary file (*.sif);;All files (*.*)')
        if len(result) > 1:
            self.block_interface()
            self.logger("Preparing to load many files")
            self.filelist = result
            self.active_buttons[0:3] = 1
            self.active_buttons[3:14] = 0
            self.active_buttons[9:11] = 1    
            self.for_preprocess.emit(result)
            newpath, shortname = os.path.split(result[0])
            self.conf_update.emit({'PATH_corrections': newpath})
            self.currentpath = newpath
        elif len(result) > 0:
            self.block_interface()
            self.logger("Preparing to load one file.")
            self.filelist = result
            self.active_buttons[0:14] = 1
            self.active_buttons[9:11] = 0    
            self.for_process.emit(result)
            newpath, shortname = os.path.split(result[0])
            self.conf_update.emit({'PATH_corrections': newpath})
            self.currentpath = newpath
    def merge_files_simple(self):
        if len(self.filelist) > 0:
            self.block_interface()
            self.active_buttons[0:9] = 1
            self.active_buttons[9:11] = 0
            self.active_buttons[11:14] = 1
            self.for_simplemerge.emit(self.filelist)
    def merge_files_offsets(self):
        if len(self.filelist) > 0:
            self.block_interface()
            self.active_buttons[0:9] = 1
            self.active_buttons[9:11] = 0
            self.active_buttons[11:14] = 1
            self.for_shiftedmerge.emit()
    def reload_file(self):
        if self.filelist is not None:
            if len(self.filelist) > 1:
                self.block_interface()
                self.logger("Preparing to load many files")
                self.active_buttons[0:3] = 1
                self.active_buttons[3:14] = 0
                self.active_buttons[9:11] = 1    
                self.for_preprocess.emit(self.filelist)
            elif len(self.filelist) > 0:
                self.block_interface()
                self.logger("Preparing to load one file.")
                self.active_buttons[0:14] = 1
                self.active_buttons[9:11] = 0    
                self.for_process.emit(self.filelist) 
    def load_params_from_file(self):
        result, ftype = QFileDialog.getOpenFileName(self.master, 'Load ADLER parameters from output file header.', self.currentpath,
           'ADLER 1D file (*.txt);;ADLER 1D file, server mode (*.asc);;All files (*.*)')
        if result == None:
            self.logger('No valid file chosen, parameters not loaded.')
        else:
            newpath, shortname = os.path.split(result)
            self.conf_update.emit({'PATH_corrections': newpath})
            self.currentpath = newpath
            self.logger('Attempting to load parameters from file: ' + str(result))
            vnames,  vdict = [],  {}
            try:
                source = open(result, 'r')
            except:
                self.logger('Could not open the file. Parameters not loaded.')
                return None
            else:
                for line in source:
                    if not '#' in line[:2]:
                        continue
                    toks = line.strip("\'[]()\"").split()
                    if len(toks) < 3:
                        continue
                    else:
                        for n, potval in enumerate([xx.strip('[](),') for xx in toks[4:]]):
                            try:
                                temp = float(potval)
                            except:
                                toks[4+n] = '-1'
                            else:
                                toks[4+n] = potval
                        if 'Profile' in toks[1]:
                            if 'LCut' in toks[2]: 
                                if 'cuts' in vnames:
                                    vdict['cuts'][0] = toks[4]
                                else:
                                    vnames.append('cuts')
                                    vdict['cuts'] = [toks[4], '']
                            elif 'HCut' in toks[2]: 
                                if 'cuts' in vnames:
                                    vdict['cuts'][1] = toks[4]
                                else:
                                    vnames.append('cuts')
                                    vdict['cuts'] = ['',  toks[4]]
                            elif 'ELine'  in toks[2]:
                                vnames.append('eline')
                                if len(toks) > 5:
                                    vdict['eline'] = [toks[4], toks[5]]
                                elif len(toks) >4:
                                    vdict['eline'] = [str(let) for let in [int(float(toks[4]))-10,  int(float(toks[4]))+10]]
                            elif 'BkgPP'  in toks[2]:
                                vnames.append('bpp')
                                vdict['bpp'] = [toks[4]]
                            elif 'BkgPercentile'  in toks[2]:
                                vnames.append('bkg_perc')
                                vdict['bkg_perc'] = [toks[4]]
                            elif 'CRay'  in toks[2]:
                                vnames.append('cray')
                                vdict['cray'] = [toks[4]]
                            elif 'Blocksize'  in toks[2]:
                                vnames.append('segsize')
                                vdict['segsize'] = [toks[4]]
                            elif 'Pair1'  in toks[2]:
                                vnames.append('eVpair1')
                                vdict['eVpair1'] = [toks[4],  toks[5]]
                            elif 'Pair2'  in toks[2]:
                                vnames.append('eVpair2')
                                vdict['eVpair2'] = [toks[4],  toks[5]]
                            elif 'RedFac'  in toks[2]:
                                vnames.append('redfac')
                                vdict['redfac'] = [toks[4]]
                            elif 'Poly'  in toks[2]:
                                if len(toks) > 6:
                                    vnames.append('poly')
                                    vdict['poly'] = [toks[4],  toks[5],  toks[6]]
                            elif 'FFTCuts'  in toks[2]:
                                vnames.append('ffts')
                                if len(toks) > 5:
                                    vdict['ffts'] = [toks[4],  toks[5]]
                        elif 'ADLER' in toks[1]:
                            if toks[2] in vnames:
                                vdict[toks[2]] = toks[4:]
                            else:
                                vnames.append(toks[2])
                                vdict[toks[2]] = toks[4:]
                source.close()
                for b in self.boxes:
                    b.takeValues(vnames,  vdict)
    @pyqtSlot()
    def show_fft(self):
        forplot = self.core.fft_plots[-1].copy()
        fpmax = forplot[:, 1].max()
        forplot = forplot[np.where(forplot[:, 1] < fpmax)]
        self.plotter.plot1D([forplot], fig = self.figure, text = "Normalised Fourier Transform plot", 
                label_override = ['Channels$^{-1}$',  'Signal'], curve_labels = ['Sum'])
    def fft_filter(self):
        self.block_interface()
        self.active_buttons[12:13] = 0
        self.for_fft.emit()
    @pyqtSlot()
    def after_ccorr(self):
        if self.core.curvature_corrected:
            self.plotter.plot2D_sliders(self.core.data2D, self.core.plotax, fig = self.figure)
        else:
            self.active_buttons[11:13] = 1
            self.flip_buttons()
    @pyqtSlot()
    def after_fft(self):
        if self.core.fft_applied:
            self.plotter.plot2D_sliders(self.core.data2D, self.core.plotax, fig = self.figure)
        else:
            self.active_buttons[12:13] = 1
            self.flip_buttons()
    def correct_curvature(self):
        self.block_interface()
        self.active_buttons[11:13] = 0
        self.for_poly.emit()
    @pyqtSlot(object)
    def plot_segment(self, seglist):
        stripe,  plotax = seglist
        self.plotter.plot1D([stripe], fig = self.figure,
                text = "Horizontal profile, rows " +str(self.core.eline[0]) + " to " + str(self.core.eline[1]), 
                  label_override = ['Channels',  'Counts'], curve_labels = ['stripe 1'])
    @pyqtSlot(object)
    def plot_histogram(self, seglist):
        data,  plotax = seglist
        hist, binlims = data
        stripe = np.column_stack([(binlims[:-1] + binlims[1:])*0.5, hist])
        self.plotter.plot1D([stripe], fig = self.figure,
                text = "Value distribution in the detector segment", 
                  label_override = ['Counts',  'Occurences'], curve_labels = ['detector values'])
