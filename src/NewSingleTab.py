
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
This is the core component of the ADLER GUI.
SingleTab is a widget where a single measurement
(possibly consisting of many files) can be loaded
into ADLER and analysed.
"""

import os
import time
import copy
from os.path import expanduser

import numpy as np
from PyQt6.QtCore import pyqtSlot, pyqtSignal, QSize,  QThread
from PyQt6.QtGui import QFont, QStandardItem
from PyQt6.QtWidgets import QFrame, QSizePolicy, QWidget, QFileDialog,   \
                                                QPushButton,  QVBoxLayout, QHBoxLayout, QFormLayout, QScrollArea
# from PyQt6 import QtGui, QtCore, QtWidgets
# from PyQt6 import sip
from VariablesGUI import VarBox
from ADLERcalc.AdlerCore import NewAdlerCore as AdlerCore
from ADLERcalc.DataHandling import RixsMeasurement
from ExtendGUI import AdlerTab, PeaxisDataModel, PeaxisTableView
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

# import matplotlib.pyplot as mpl
# # from matplotlib.backends import qt_compat
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar2QTAgg
# from matplotlib.widgets import Slider

# this is a Windows thing
# ctypes.windll.kernel32.SetDllDirectoryW('.')

# simple mathematical functions are defined here

GlobFont = QFont('Sans Serif', int(12*font_scale))

oldval = 0.0

#### GUI part

loading_variables = [
{'Name': 'Detector cutoff LRBT',  'Unit':'pixel',  'Value':np.array([0, 2048, 0, 2048]),  'Key' : 'cuts', 
                               'MinValue':np.array([0, 0, 0, 0]),  'MaxValue':np.array([8192, 8192, 8192, 8192]),  'Length': 4,  'Type':'int', 
                               'WarnZones':[np.array([1, 2047])], 
                               "comment":"Limits of the 2D detector image to be used in processing.\n sequence: LEFT RIGHT BOTTOM TOP"}, 
{'Name': 'Background per pixel',  'Unit':'counts',  'Value':960.78154,   'Key' : 'bpp', 
                               'MinValue':0.0,  'MaxValue':1e5,  'Length': 1,  'Type':'float'},
{'Name': 'Time-dependent background',  'Unit':'fraction',  'Value':1.0,   'Key' : 'tdb', 
                               'MinValue':0.0,  'MaxValue':1e3,  'Length': 1,  'Type':'float', 
                               'WarnZones':[np.array([1.001, 1e5]), np.array([0.0, 0.999])]},
{'Name': 'Cosmic ray correction factor',  'Unit':'StdDev',  'Value':3,  'Key' : 'cray', 
                               'MinValue':-10.0,  'MaxValue':1e5,  'Length': 1,  'Type':'float', 
                               'WarnZones':[np.array([-10.0, 2.99])]}
]
line_variables = [
{'Name': 'Elastic line limits',  'Unit':'pixel',  'Value':np.array([0, 2048]),   'Key' : 'eline', 
                               'MinValue':np.array([-10, -10]),  'MaxValue':np.array([2048, 2048]),  'Length': 2,  'Type':'int'},   
{'Name': 'Detection limit for BKG',  'Unit':'percentile',  'Value':75,  'Key' : 'bkg_perc', 
                               'MinValue':0.0,  'MaxValue':100.0,  'Length': 1,  'Type':'float'}, 
{'Name': 'Curvature segment size',  'Unit':'pixel',  'Value':16, 'Key' : 'segsize', 
                               'MinValue':1,  'MaxValue':1024,  'Length': 1,  'Type':'int'}, 
]
# energy_variables = [
# {'Name': 'Energy calibration pair 1',  'Unit':'pixel, eV',  'Value':np.array([-1.0, -1.0]), 'Key' : 'eVpair1', 
#                                       'MinValue':-1e5*np.ones(2),
#                                       'MaxValue':1e5*np.ones(2),
#                                       'Length': 2,  'Type':'float'}, 
# {'Name': 'Energy calibration pair 2',  'Unit':'pixel, eV',  'Value':np.array([-1.0, -1.0]), 'Key' : 'eVpair2', 
#                                       'MinValue':-1e5*np.ones(2),
#                                       'MaxValue':1e5*np.ones(2),
#                                       'Length': 2,  'Type':'float'}, 
# ]
correction_variables = [
{'Name': 'Curvature Correction',  'Unit':'N/A',  'Value':-100.0*np.ones(3), 'Key' : 'poly', 
                                      'MinValue':-1e15*np.ones(3),
                                      'MaxValue':1e15*np.ones(3),
                                      'Length': 3,  'Type':'float'}, 
{'Name': 'Removed FFT region',  'Unit':'N/A',  'Value':-1e9*np.ones(2), 'Key' : 'ffts', 
                                      'MinValue':-1e15*np.ones(2),
                                      'MaxValue':1e15*np.ones(2),
                                      'Length': 2,  'Type':'float'}, 
{'Name': 'Reduction factor',  'Unit':'N/A',  'Value':1.0, 'Key' : 'redfac', 
                                      'MinValue':0.25,
                                      'MaxValue':10.0,
                                      'Length': 1,  'Type':'float'}, 
]

class SingleTab(AdlerTab):
    """This tab is central to the ADLER interface.
    It is the part of the GUI that deals with processing
    the results of a single measurement. Combining files,
    offset correction, background subtraction - all these
    are applied to the data in this tab.

    It is connected to an instance of AdlerCore
    which performs all the operations on the data.
    """
    for_preprocess = pyqtSignal(object)
    for_process = pyqtSignal(object)
    for_simplemerge = pyqtSignal(object)
    for_shiftedmerge = pyqtSignal()
    for_expensivemerge = pyqtSignal(object)
    for_poly = pyqtSignal()
    for_fft = pyqtSignal()
    for_fake = pyqtSignal()
    # got_valid_dir = pyqtSignal()
    def __init__(self, master,  canvas,  log,  mthreads = 1, startpath = None,  logplotter = None,  app = None, 
                                dialog = False, accept_button = None):
        super().__init__(master)
        self.master = master
        self.dialog = dialog
        self.extpath = '.'
        self.acc_button = accept_button
        # self.progbar = None
        self.canvas, self.figure, self.clayout = canvas
        self.plotter = Plotter(figure = self.figure)
        self.params = [(loading_variables, "File Loading"),  (line_variables,  "Elastic Line"),
                            (correction_variables,  "Corrections")]
        self.parnames = []
        self.pardict = {}
        self.log = log
        self.logplotter = logplotter
        self.filelist = None
        # self.destroyed.connect(self.cleanup)
        self.core = AdlerCore(None,  None,  self.log,  max_threads = mthreads, 
                                        thr_exit = self.unblock_interface,  startpath = startpath, 
                                        progress_bar = self.progbar)
        self.currentpath = startpath
        self.boxes = self.make_layout()
        self.core.assign_boxes(self.boxes)
        self.core.fittingresult.connect(self.plot_fitting_results)
        self.core.energyresult.connect(self.plot_energy_results)
        self.core.curvatureresult.connect(self.plot_curve_profile)
        self.core.segmentresult.connect(self.plot_segment)
        self.core.finished_preprocess.connect(self.show_1D_comparison)
        self.core.finished_preprocess.connect(self.trigger_logplotter)
        self.core.finished_2D.connect(self.show_2D_plot)
        self.core.finished_process.connect(self.trigger_logplotter)
        self.core.finished_poly.connect(self.after_ccorr)
        self.core.finished_fft.connect(self.after_fft)
        #
        self.for_preprocess.connect(self.core.preprocess_manyfiles)
        self.for_process.connect(self.core.process_manyfiles)
        self.for_simplemerge.connect(self.core.justload_manyfiles)
        self.for_shiftedmerge.connect(self.core.finalise_manyfiles)
        self.for_expensivemerge.connect(self.core.expensive_merge_manyfiles)
        self.for_poly.connect(self.core.apply_poly)
        self.for_fft.connect(self.core.correct_fft)
        self.for_fake.connect(self.core.generate_mock_dataset)
        self.calib_model.itemChanged.connect(self.core.pass_calibration)
        self.calib_model.itemChanged.connect(self.calib_table.resizeColumnsToContents)
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
    @pyqtSlot()
    def cleanup(self):
        """This method is called on exit. Before
        the GUI is destroyed, the thread running in the background
        (and operating the AdlerCore) should exit.
        """
        self.corethread.quit()
        self.corethread.wait()
    def make_layout(self):
        """This method creates and positions
        all the widgets of the tab.
        """
        # base = QWidget(self.master)
        # self.base=base
        base = self.base
        base.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        base_layout = QHBoxLayout(base)
        if self.dialog and self.acc_button is not None:
            extra = QWidget(base)
            extra_layout = QVBoxLayout(extra)
            extra_layout.addWidget(self.acc_button)
            extra_layout.addWidget(self.canvas)
            base_layout.addWidget(extra)
        else:
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
        # 
        self.calib_model = PeaxisDataModel(boxes_base, ["Filename", "Line position [channels]", "Beam energy [eV]", "Date"])
        self.calib_table = PeaxisTableView(boxes_base)
        self.calib_table.setSortingEnabled(True)
        self.calib_table.setModel(self.calib_model)
        for n in range(5):
            line = [QStandardItem(str(x)) for x in ['---', '-1.0', '-1.0', 'DD/MM/YYYY']]
            self.calib_model.appendRow(line)
        boxes_layout.addWidget(self.calib_table)
        self.core.assign_calibration(self.calib_model)
        # self.progbar = QProgressBar(base)
        boxes = []
        for el in self.params:
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
        ['Get Calibration', self.load_calibration_from_files, 'Read energy calibration from existing output files.', 
            '', 'Files'], # 2
        ['Get Parameters', self.load_params_from_file, 'Read processing parameters from an existing output file.', 
            '', 'Files'], # 2
        ['Fast Fit', self.core.autoprocess_file, 'Create a 1D profile, guess line position.', 
            col1, 'Processing'], # 3
        ['Make Profile', self.core.process_file, 'Create a 1D profile.', 
            col1, 'Processing'], # 4
        ['Energy Fast Fit', self.core.auto_eV_profile, 'Same as Fast Fit, but using energy units on the x axis.', 
            col2, 'Processing'], # 5
        ['Energy Profile', self.core.eV_profile, 'Same as Make Profile, but using energy units on the x axis.', 
            col2, 'Processing'], # 6
        ['Plot Curvature', self.core.curve_profile, 'Plot position of the peak as a function of horizontal position.', 
            col3, 'QC'], # 7
        ['Plot Horizontal', self.core.make_stripe, 'Plot the intensity profile of the elastic line across the detector.', 
            col3, 'QC'], # 8
        ['Direct Merge', self.merge_files_simple, 'Stack up all the files, pixel per pixel.', 
            col1, 'Merging'], # 9
        ['Shifted Merge', self.merge_files_offsets, 'Shift the files relative to each other to find the best match.', 
            col1, 'Merging'], # 10
        ['Expensive Merge', self.merge_files_expensive, 'Shift the files relative to each other using a global minumum search algorithm.', 
            col1, 'Merging'], # 11
        ['Correct Curvature', self.correct_curvature, 'Stack up all the files, pixel per pixel.', 
            col3, 'Corrections'], # 12
        ['FFT Filter', self.fft_filter, 'Shift the files relative to each other to find the best match.', 
            col3, 'Corrections'], # 13
        ['Fake Data', self.fake_data_button, 'Look for processing artifacts by working on random numbers.', 
            col3, 'QC'], # 14
        ]
        self.active_buttons = np.zeros(len(button_list)).astype(np.int)
        self.active_buttons[0] = 1
        self.active_buttons[2:4] = 1
        self.active_buttons[14] = 1
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
    def logger(self, message):
        now = time.gmtime()
        timestamp = ("-".join([str(tx) for tx in [now.tm_mday, now.tm_mon, now.tm_year]])
                     + ',' + ":".join([str(ty) for ty in [now.tm_hour, now.tm_min, now.tm_sec]]) + '| ')
        self.log.setReadOnly(False)
        self.log.append('MainTab :' + timestamp + message)
        self.log.setReadOnly(True)
    def MakeButton(self, parent, text, function, tooltip = ""):
        button = QPushButton(text, parent)
        if tooltip:
            button.setToolTip(tooltip)
        button.clicked.connect(function)
        # button.clicked.connect(self.block_interface)
        button.setSizePolicy(QSizePolicy.Policy.Preferred,QSizePolicy.Policy.Expanding)
        button.setMinimumSize(QSize(48, 25))
        # button.setMaximumSize(QSize(300, 100))
        button.setFont(GlobFont)
        return button
    def set_extpath(self, pathname):
        self.extpath = pathname
    @pyqtSlot(object)
    def load_file_indirect(self, flist):
        if len(flist) > 1:
            self.block_interface()
            self.logger("Preparing to load many files")
            self.filelist = []
            for name  in flist:
                self.filelist.append(os.path.join(self.extpath, name))
            result = self.filelist
            self.active_buttons[0:4] = 1
            self.active_buttons[4:15] = 0
            self.active_buttons[10:13] = 1    
            self.for_preprocess.emit(result)
            newpath, shortname = os.path.split(result[0])
            self.conf_update.emit({'PATH_data': newpath})
            self.currentpath = newpath
            # self.got_valid_dir.emit()
        elif len(flist) > 0:
            self.block_interface()
            self.logger("Preparing to load one file.")
            self.filelist = [os.path.join(self.extpath, flist[0])]
            result = self.filelist
            self.active_buttons[0:15] = 1
            self.active_buttons[10:13] = 0    
            self.for_process.emit(result)
            newpath, shortname = os.path.split(result[0])
            self.conf_update.emit({'PATH_data': newpath})
            self.currentpath = newpath
    def load_file_button(self):
        result, ftype = QFileDialog.getOpenFileNames(self.master,
           'Load data from one or more Andor camera files:', self.currentpath,
           # 'Andor binary file (*.sif);;All files (*.*)'
           'Andor binary file (*.sif);;FITS format (*.fits);;Old PEAXIS text files(*.asc);;All files (*.*)')
        if len(result) > 1:
            self.block_interface()
            self.logger("Preparing to load many files")
            self.filelist = result
            self.active_buttons[0:4] = 1
            self.active_buttons[4:15] = 0
            self.active_buttons[10:13] = 1    
            self.for_preprocess.emit(result)
            newpath, shortname = os.path.split(result[0])
            self.conf_update.emit({'PATH_data': newpath})
            self.currentpath = newpath
            # self.got_valid_dir.emit()
        elif len(result) > 0:
            self.block_interface()
            self.logger("Preparing to load one file.")
            self.filelist = result
            self.active_buttons[0:15] = 1
            self.active_buttons[10:13] = 0    
            self.for_process.emit(result)
            newpath, shortname = os.path.split(result[0])
            self.conf_update.emit({'PATH_data': newpath})
            self.currentpath = newpath
            # self.got_valid_dir.emit()
    def flip_buttons(self):
        if self.dialog:
            self.active_buttons[0] = 0
            self.active_buttons[15] = 0
        super().flip_buttons()
    def fake_data_button(self):
        self.block_interface()
        self.for_fake.emit()
        self.active_buttons[0:10] = 1
        self.active_buttons[10:13] = 0
        self.active_buttons[13:15] = 1
    @pyqtSlot()
    def show_1D_comparison(self):
        self.plotter.plot_new_profiles(separate =self.core.individual_profiles,
                                   merged = [self.core.summed_rawprofile,  self.core.summed_adjusted_rawprofile],
                                    fig = self.figure, text = "Pick the better profile!", label_override = ['Channels',  'Counts'],
                                    separate_labels= self.core.individual_labels, merged_labels= ['Simple Merge',  'Shifted Merge'], 
                                    title = "Breakdown of the individual profiles.")
    @pyqtSlot()
    def show_2D_plot(self):
        if self.core.timedata[0] > 1:
            descr = " files"
        else:
            descr = " file"
        titlebar = "+".join([str(x) for x in np.unique(self.core.energies)]) + " eV, " +\
                    str(self.core.timedata[1].sum()) + " s in " + str(self.core.timedata[0]) + descr +"\n" + \
                    self.core.temp_name
        self.plotter.plot2D_sliders(self.core.data2D, self.core.plotax, fig = self.figure, text = titlebar)
    @pyqtSlot()
    def trigger_logplotter(self):
        if self.logplotter is not None:
            self.logplotter.clear_list()
            self.logplotter.takeData(self.core.logvals)
    def merge_files_simple(self):
        if len(self.filelist) > 0:
            self.block_interface()
            self.active_buttons[0:10] = 1
            self.active_buttons[10:15] = 0
            self.active_buttons[13:15] = 1
            self.for_simplemerge.emit(self.filelist)
    def merge_files_offsets(self):
        if len(self.filelist) > 0:
            self.block_interface()
            self.active_buttons[0:10] = 1
            self.active_buttons[10:13] = 0
            self.active_buttons[13:15] = 1
            self.for_shiftedmerge.emit()
    def merge_files_expensive(self):
        if len(self.filelist) > 0:
            self.block_interface()
            self.active_buttons[0:10] = 1
            self.active_buttons[10:13] = 0
            self.active_buttons[13:15] = 1
            self.for_expensivemerge.emit(self.filelist)
    def reload_file(self):
        if self.filelist is not None:
            if len(self.filelist) > 1:
                self.block_interface()
                self.logger("Preparing to load many files")
                self.active_buttons[0:4] = 1
                self.active_buttons[4:15] = 0
                self.active_buttons[10:13] = 1    
                self.for_preprocess.emit(self.filelist)
            elif len(self.filelist) > 0:
                self.block_interface()
                self.logger("Preparing to load one file.")
                self.active_buttons[0:15] = 1
                self.active_buttons[10:13] = 0    
                self.for_process.emit(self.filelist)    
    def autoprocess_file_button(self):
        self.block_interface()
        obj, thread = self.thread_locknload(self.core.process_file,  args = [True])
        thread.start()
    def process_file_button(self):
        self.block_interface()
        obj, thread = self.thread_locknload(self.core.process_file)
        thread.start()
    @pyqtSlot(object)
    def plot_fitting_results(self,  fittinglist):
        profi, back, peak, fitpars, text = fittinglist
        if profi is None:
            self.logger('No data available to be processed.')
            return None
        elif back is None or peak is None or text is None:
            if self.core.timedata[0] > 1:
                descr = " files"
            else:
                descr = " file"
            titlebar = "+".join([str(x) for x in np.unique(self.core.energies)]) + " eV, " +\
                        str(self.core.timedata[1].sum()) + " s in " + str(self.core.timedata[0]) + descr +"\n" + \
                        self.core.temp_name
            self.plotter.plot1D([profi], fig = self.figure, text = "",  title = titlebar, 
                  label_override = [self.core.chan_override, 'Counts'], curve_labels = ['Data'], eline = self.core.nom_eline, 
                       efactor = (round(self.core.the_object.mev_per_channel, 3)))
        else:
            if self.core.timedata[0] > 1:
                descr = " files"
            else:
                descr = " file"
            titlebar = "+".join([str(x) for x in np.unique(self.core.energies)]) + " eV, " +\
                        str(self.core.timedata[1].sum()) + " s in " + str(self.core.timedata[0]) + descr +"\n" + \
                        self.core.temp_name
            self.plotter.plot1D([profi,  back,  peak], fig = self.figure, text = text,  title = titlebar, 
                  label_override = [self.core.chan_override, 'Counts'], curve_labels = ['Data',  'Background', 'Fit'], eline = self.core.nom_eline, 
                      efactor = (round(self.core.the_object.mev_per_channel, 3)))
    def load_params_from_dict(self, tdict):
        self.logger('Attempting to load parameters from a dictionary')
        vnames = [str(xx) for xx in tdict.keys()]
        vdict = {}
        for vn in vnames:
            try:
                len(tdict[vn])
            except:
                vdict[vn] =np.array([tdict[vn]])
            else:
                vdict[vn] =np.array(tdict[vn])
        for b in self.boxes:
            b.takeValues(vnames,  vdict)
    def load_calibration_from_files(self):
        print('Opening directory:', self.core.temp_path)
        results, ftype = QFileDialog.getOpenFileNames(self.master,
           'Load ADLER calibration from output files.',
            self.currentpath,
           'ADLER 1D file (*.txt);;ADLER YAML file (*.yaml);;All files (*.*)')
        if len(results) < 1:
            self.logger('No valid file chosen, calibration not loaded.')
        else:
            for result in results:
                newpath, shortname = os.path.split(result)
                self.conf_update.emit({'PATH_data': newpath})
                self.currentpath = newpath
                self.logger('Attempting to load calibration from file: ' + str(result))
                vnames,  vdict = [],  {}
                calibration = []
                if 'yaml' in result.split('.')[-1]:
                    temp = RixsMeasurement()
                    try:
                        temp.read_yaml(result)
                    except:
                        self.logger('Could not open the file. Calibration not loaded.')
                        return None
                else:
                    temp = RixsMeasurement()
                    temp.read_extended_ADLER(result)
                    try:
                        temp.read_extended_ADLER(result)
                    except:
                        self.logger('Could not open the file. Calibration not loaded.')
                        return None
                energy = temp.energy
                position = temp.fitting_params_channels['centre']
                name = shortname
                try:
                    date = temp.start_points[0]
                except:
                    date = '--/--/----'
                tableline = [QStandardItem(str(x)) for x in [name, position, energy, date]]
                self.calib_model.appendRow(tableline)
            self.core.pass_calibration()
    def load_params_from_file(self):
        print('Opening directory:', self.core.temp_path)
        result, ftype = QFileDialog.getOpenFileName(self.master,
           'Load ADLER parameters from output file header.',
            self.currentpath,
           'ADLER 1D file (*.txt);;ADLER 1D file, server mode (*.asc);;All files (*.*)')
        if result == None:
            self.logger('No valid file chosen, parameters not loaded.')
        else:
            newpath, shortname = os.path.split(result)
            self.conf_update.emit({'PATH_data': newpath})
            self.currentpath = newpath
            self.logger('Attempting to load parameters from file: ' + str(result))
            vnames,  vdict = [],  {}
            calibration = []
            try:
                source = open(result, 'r')
            except:
                self.logger('Could not open the file. Parameters not loaded.')
                return None
            else:
                for line in source:
                    if not '#' in line[:2]:
                        continue
                    if 'Calibration:' in line:
                        calibration.append('Calibration:'.join(line.split('Calibration:')[1:]))
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
                for line in calibration:
                    toks = line.split(',')
                    vals = toks[0].strip('[](),')
                    energy = float(vals.split()[0].strip('[](),'))
                    position = float(vals.split()[1].strip('[](),'))
                    name = toks[1]
                    date = toks[2]
                    tableline = [QStandardItem(str(x)) for x in [name, position, energy, date]]
                    self.calib_model.appendRow(tableline)
                self.core.pass_calibration()
    def curve_profile(self):
        obj, thread = self.thread_locknload(self.core.curve_profile)
        thread.start()
    @pyqtSlot(object)
    def plot_curve_profile(self,  curvelist):
        if self.core.curvature is not None:
            if self.core.timedata[0] > 1:
                descr = " files"
            else:
                descr = " file"
            titlebar = "+".join([str(x) for x in np.unique(self.core.energies)]) + " eV, " +\
                        str(self.core.timedata[1].sum()) + " s in " + str(self.core.timedata[0]) + descr +"\n" + \
                        self.core.temp_name
            curve,  fit,  fitparam = curvelist
            starty,  endy = 1,  2048
            if self.core.eline[0] >0:
                starty = self.core.eline[0]
            if self.core.eline[1] >0:
                endy = self.core.eline[1]
            # plotax = [(starty,  endy), (self.core.cuts[0],  self.core.cuts[1])]
            self.plotter.plot1D([curve,  fit], fig = self.figure, text = self.core.temp_name,  title = titlebar, 
                  label_override = ['Channels',  'Channels'], curve_labels = ['Elastic line position', 'Curvature fit'])
    def fft_filter(self):
        self.block_interface()
        self.active_buttons[14:15] = 0
        self.for_fft.emit()
    @pyqtSlot()
    def after_ccorr(self):
        if self.core.curvature_corrected:
            self.plotter.plot2D_sliders(self.core.data2D, self.core.plotax, fig = self.figure, text = self.core.temp_name)
        else:
            self.active_buttons[13:15] = 1
            self.flip_buttons()
    @pyqtSlot()
    def after_fft(self):
        if self.core.fft_applied:
            self.plotter.plot2D_sliders(self.core.data2D, self.core.plotax, fig = self.figure, text = self.core.temp_name)
        else:
            self.active_buttons[14:15] = 1
            self.flip_buttons()
    def correct_curvature(self):
        self.block_interface()
        self.active_buttons[13:15] = 0
        self.for_poly.emit()
    def segment_plot_button(self):
        if self.core.data2D is None:
            self.logger("No data has been loaded for curvature correction")
            return None
        if self.core.eline[0] < 0 or self.core.eline[1] < self.core.eline[0]:
            self.logger("The elastic line limits are ill-defined. Not processing further.")
            return None
        obj, thread = self.thread_locknload(self.core.make_stripe)
        thread.start()
    @pyqtSlot(object)
    def plot_segment(self, seglist):
        stripe,  plotax = seglist
        self.plotter.plot1D([stripe], fig = self.figure, title = self.core.temp_name, 
                text = "Horizontal profile, rows " +str(self.core.eline[0]) + " to " + str(self.core.eline[1]), 
                  label_override = ['Channels',  'Counts'], curve_labels = ['stripe 1'])
    def autoeV_profile_button(self):
        obj, thread = self.thread_locknload(self.core.eV_profile,  args = [True])
        thread.start()
    def eV_profile_button(self):
        obj, thread = self.thread_locknload(self.core.eV_profile)
        thread.start()
    @pyqtSlot(object)
    def plot_energy_results(self, fittinglist):
        profi, back, peak, fitpars, text = fittinglist
        if profi is None:
            self.logger('No data available to be processed.')
            return None
        elif back is None or peak is None or text is None:
            if self.core.timedata[0] > 1:
                descr = " files"
            else:
                descr = " file"
            titlebar = "+".join([str(x) for x in np.unique(self.core.energies)]) + " eV, " +\
                        str(self.core.timedata[1].sum()) + " s in " + str(self.core.timedata[0]) + descr +"\n" + \
                        self.core.temp_name
            self.plotter.plot1D([profi], fig = self.figure, text = "",  title = titlebar, 
                  label_override = ['Energy [eV]',  'Counts'], curve_labels = ['Data'], 
                  efactor = (round(self.core.the_object.mev_per_channel, 3)))
        else:
            if self.core.timedata[0] > 1:
                descr = " files"
            else:
                descr = " file"
            titlebar = "+".join([str(x) for x in np.unique(self.core.energies)]) + " eV, " +\
                        str(self.core.timedata[1].sum()) + " s in " + str(self.core.timedata[0]) + descr +"\n" + \
                        self.core.temp_name
            self.plotter.plot1D([profi,  back,  peak], fig = self.figure, text = text,  title = titlebar, 
                  label_override = ['Energy transfer [eV]',  'Counts'], curve_labels = ['Data',  'Background', 'Fit'], 
                  efactor = (round(self.core.the_object.mev_per_channel, 3)))
    def merge_files_eV_button(self):
        temppar = self.read_inputs(self.params, newfields = None)
        temppar = self.validate_params(temppar, newfields = self.input_fields)
        self.update_params(temppar)
        result, ftype = QFileDialog.getOpenFileNames(self.master, 'Load data from the Andor camera file:', self.currentpath,
               'Andor binary file (*.sif);;All files (*.*)')
        newpath, shortname = os.path.split(result[0])
        self.conf_update.emit({'PATH_data': newpath})
        self.currentpath = newpath
        self.merge_files_eV(result, ftype)
    def merge_eV_multiple(self):
        self.logger('Wizard has prepared the list of files')
        self.logger(str(self.manyfiles))
        self.logger(str(self.manypars))
        self.merge_files_eV(self.manyfiles, [], parlist = self.manypars)
