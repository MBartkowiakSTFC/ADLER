
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
# Copyright (C) Maciej Bartkowiak, 2019-2022

__doc__ = """
This is the main window of the ADLER GUI.
All the other tab instances are created here.
"""

import sys
import numpy as np
import os
import time
import copy
from os.path import expanduser
# this is a Windows thing

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

try:
    import ctypes
    ctypes.windll.kernel32.SetDllDirectoryW(resource_path('.'))
except ImportError:
    print("Ignoring ctypes import. If you don't have it, you don't need it.")
except AttributeError:
    print("Ignoring ctypes import. If you don't have it, you don't need it.")

# from PyQt5 import Qt as testQt
from PyQt6.QtCore import pyqtSlot, QSize, QMetaObject, QLocale, QObject, QThread, QMutex, QSortFilterProxyModel
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame,  QTabWidget, QSizePolicy, QApplication,  QMainWindow, \
                                                QPushButton,  QVBoxLayout, QWidget, \
                                                QLineEdit, QHBoxLayout, QAbstractItemView, \
                                                QFileDialog, QLabel
# from PyQt5 import QtGui, QtCore, QtWidgets
# from PyQt5 import sip
from NewSingleTab import SingleTab
from CorrectionsTab import CorrectionsTab
from CompareRixsTab import NewPostprocessingTab as PostprocessingTab
from XASplottingTab import XASplottingTab
from FittingTab import FittingTab
from LogviewerTab import LogviewerTab
from BeamlineTab import BeamlineTab
from ExtendGUI import LogBox
from FileFinder import PeaxisDataModel
from ExperimentTree import TableView

ADLER_VERSION_STRING = "4.0 from 03.06.2022"

mpl_scale = 1.0
mpl_figure_scale = 1.0
font_scale = 1.0
last_dir_saved = expanduser("~")
MAX_THREADS = 1

paths = {}
# print(paths)

try:
    source = open(os.path.join(expanduser("~"),'.ADLconfig.txt'), 'r')
except:
    paths['data'] = '.'
    paths['corrections'] = '.'
    paths['overview'] = '.'
    paths['postprocessing'] = '.'
    paths['variableviewer']  = '.'
    paths['xasplotting'] = '.'
    paths['beamline'] = '.'
    paths['fitting'] = '.'
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
            if toks[0] == 'Lastdir:':
                try:
                    last_dir_saved = toks[1]
                except:
                    pass
            if toks[0] == 'MAX_THREADS:':
                try:
                    MAX_THREADS = int(float(toks[1]))
                except:
                    pass
            if toks[0] == 'PATH_data:':
                try:
                    paths['data'] = toks[1]
                except:
                    paths['data'] = '.'
            elif 'data' not in paths.keys():
                paths['data'] = '.'
            if toks[0] == 'PATH_corrections:':
                try:
                    paths['corrections'] = toks[1]
                except:
                    paths['corrections'] = '.'
            elif 'corrections' not in paths.keys():
                paths['corrections'] = '.'
            if toks[0] == 'PATH_xasplotting:':
                try:
                    paths['xasplotting'] = toks[1]
                except:
                    paths['xasplotting'] = '.'
            elif 'xasplotting' not in paths.keys():
                paths['xasplotting'] = '.'
            if toks[0] == 'PATH_postprocessing:':
                try:
                    paths['postprocessing'] = toks[1]
                except:
                    paths['postprocessing'] = '.'
            elif 'postprocessing' not in paths.keys():
                paths['postprocessing'] = '.'
            if toks[0] == 'PATH_variableviewer:':
                try:
                    paths['variableviewer'] = toks[1]
                except:
                    paths['variableviewer'] = '.'
            elif 'variableviewer' not in paths.keys():
                paths['variableviewer'] = '.'
            if toks[0] == 'PATH_overview:':
                try:
                    paths['overview'] = toks[1]
                except:
                    paths['overview'] = '.'
            elif 'overview' not in paths.keys():
                paths['overview'] = '.'
            if toks[0] == 'PATH_beamline:':
                try:
                    paths['beamline'] = toks[1]
                except:
                    paths['beamline'] = '.'
            elif 'beamline' not in paths.keys():
                paths['beamline'] = '.'
            if toks[0] == 'PATH_fitting:':
                try:
                    paths['fitting'] = toks[1]
                except:
                    paths['fitting'] = '.'
            elif 'fitting' not in paths.keys():
                paths['fitting'] = '.'
    source.close()
# print(paths)

# Some users with extremely high screen resolution were unable to read the text
# in the matplotlib figures.
# The scaling parameters set in the .ADLconfig.txt file can be used to
# increase the figure and font size.
import matplotlib
matplotlib.use("Qt5Agg")
matplotlib.rcParams['legend.fontsize'] = int(14*mpl_scale)
matplotlib.rcParams['legend.borderpad'] = 0.3*mpl_scale
matplotlib.rcParams['legend.labelspacing'] = 0.2*mpl_scale
matplotlib.rcParams['xtick.labelsize'] = int(16*mpl_scale)
matplotlib.rcParams['ytick.labelsize'] = int(16*mpl_scale)
matplotlib.rcParams['axes.titlesize'] = int(16*mpl_scale)
matplotlib.rcParams['axes.labelsize'] = int(16*mpl_scale)
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['font.size'] = int(14*mpl_scale)
import matplotlib.pyplot as mpl
# from matplotlib.backends import qt_compat
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar2QTAgg
# from matplotlib.widgets import Slider

GlobFont = QFont('Sans Serif', int(12*font_scale))

oldval = 0.0

#### filesystem monitoring part
def FindUnprocessedFiles(fpath):
    infiles, outfiles = [], []
    # with os.scandir(fpath) as it:
    for entry in os.scandir(fpath):
        if entry.is_file():
            tokens = entry.name.split('.')
            name, extension = '.'.join(tokens[:-1]), tokens[-1]
            if extension == 'sif':
                infiles.append(name)
            elif extension == 'asc':
                if name[-3:] == '_1D':
                    outfiles.append(name[:-3])
                else:
                    outfiles.append(name[:-3])
    unp_files = []
    for fnam in infiles:
        if not fnam in outfiles:
            unp_files.append(fnam)
    return unp_files

#### GUI part

loading_variables = [
{'Name': 'Detector cutoff',  'Unit':'pixel',  'Value':np.array([0, 2048]),  'Key' : 'cuts', 
                               'MinValue':np.array([0, 0]),  'MaxValue':np.array([2048, 2048]),  'Length': 2,  'Type':'int',
                               'Comment':''}, 
{'Name': 'Background per pixel',  'Unit':'counts',  'Value':960.0,   'Key' : 'bpp', 
                               'MinValue':0.0,  'MaxValue':1e5,  'Length': 1,  'Type':'float',
                               'Comment':''},
{'Name': 'Cosmic ray correction factor',  'Unit':'StdDev',  'Value':3,  'Key' : 'cray', 
                               'MinValue':-10.0,  'MaxValue':1e5,  'Length': 1,  'Type':'float',
                               'Comment':''}
]
line_variables = [
{'Name': 'Elastic line limits',  'Unit':'pixel',  'Value':np.array([0, 2048]),   'Key' : 'eline', 
                               'MinValue':np.array([-10, -10]),  'MaxValue':np.array([2048, 2048]),  'Length': 2,  'Type':'int',
                               'Comment':''},   
{'Name': 'Detection limit for BKG',  'Unit':'percentile',  'Value':75,  'Key' : 'bkg_perc', 
                               'MinValue':0.0,  'MaxValue':100.0,  'Length': 1,  'Type':'float',
                               'Comment':''}, 
{'Name': 'Curvature segment size',  'Unit':'pixel',  'Value':16, 'Key' : 'segsize', 
                               'MinValue':1,  'MaxValue':1024,  'Length': 1,  'Type':'int',
                               'Comment':''}, 
]
energy_variables = [
{'Name': 'Energy calibration pair 1',  'Unit':'pixel, eV',  'Value':np.array([-1.0, -1.0]), 'Key' : 'eVpair1', 
                                      'MinValue':-1e5*np.ones(2),
                                      'MaxValue':1e5*np.ones(2),
                                      'Length': 2,  'Type':'float',
                               'Comment':''}, 
{'Name': 'Energy calibration pair 2',  'Unit':'pixel, eV',  'Value':np.array([-1.0, -1.0]), 'Key' : 'eVpair2', 
                                      'MinValue':-1e5*np.ones(2),
                                      'MaxValue':1e5*np.ones(2),
                                      'Length': 2,  'Type':'float',
                               'Comment':''}, 
]
correction_variables = [
{'Name': 'Curvature Correction',  'Unit':'N/A',  'Value':-1.0*np.ones(3), 'Key' : 'poly', 
                                      'MinValue':-1e15*np.ones(3),
                                      'MaxValue':1e15*np.ones(3),
                                      'Length': 3,  'Type':'float',
                               'Comment':''}, 
{'Name': 'Removed FFT region',  'Unit':'N/A',  'Value':-1e9*np.ones(2), 'Key' : 'ffts', 
                                      'MinValue':-1e15*np.ones(2),
                                      'MaxValue':1e15*np.ones(2),
                                      'Length': 2,  'Type':'float',
                               'Comment':''}, 
{'Name': 'Reduction factor',  'Unit':'N/A',  'Value':1.0, 'Key' : 'redfac', 
                                      'MinValue':0.25,
                                      'MaxValue':10.0,
                                      'Length': 1,  'Type':'float',
                               'Comment':''}, 
]

single_vars = [
        ["Input File", ["Single", "IFname"], "Andor SIF file to be processed.", ''],
        ]
        
curvature_vars = [
        ["Input File(s)", ["Curvature", "IFname"], "Andor SIF file(s) to be processed.", ''],
        ["Lower vertical cutoff [channel]", ["Curvature", "LCut"], "First pixel column to be included in the 1D profile", '0'],
        ["Upper vertical cutoff [channel]", ["Curvature", "HCut"], "Last pixel column to be included in the 1D profile", '2048'],
        ["Elastic line position [channel]", ["Curvature", "ELine"], "Initial guess of the elastic line position for fitting.\n"+
                                                                    "ALWAYS use two elements 'min,max' here to limit the range.", ''],
        ["Background per pixel", ["Curvature", "BkgPP"], "This value will be subtracted from each pixel. Will be guessed if left empty", '900'],
        ["Background limit (percentile)", ["Curvature", "BkgPercentile"], "All the points with values below this percentile will be considered background.", '75'],
        ["Cosmic ray correction factor", ["Curvature", "CRay"], "For each pixel row, only values within THIS_NUMBER * StdDev from the mean value will be accepted.", '5'],
        ["Pixels per segment for curvature profile", ["Curvature", "Blocksize"], "This is the number of vertical channels that will be summed up to produce one point in the curvature plot.", '16'],
        ["Polynomial coefficients", ["Curvature", "Poly"], "If you already know the curvature parameters, you can put them here directly", 'None'],
        ["Cut FFT regions", ["Curvature", "FFTCuts"], "Regions to be removed from data in the reciprocal space", 'None'],
        ]

server_vars = [
        ["Working Directory", ["Server", "WDir"], "Working directory where the server will wait for new files to appear.", ''],
        ["First number", ["Server", "NLow"], "If specified, processing will start with a file name containing this number", ''],
        ["Last number", ["Server", "NHigh"], "If specified, processing will finish on a file name containing this number", ''],
        ]

params = {}
params['Single'] = {}
params['Profile'] = {}
params['Server'] = {}
params['Curvature'] = {}

for k in single_vars:
    params['Single'][k[1][1]] = ""
    
for k in server_vars:
    params['Server'][k[1][1]] = ""
    
for k in curvature_vars:
    params['Curvature'][k[1][1]] = ""

class ConfigFileWriter(QObject):
    def __init__(self):
        super().__init__()
        self.confpath = os.path.join(expanduser("~"),'.ADLconfig.txt')
        self.all_vars = {}
        self.lock = QMutex()
        self.init_vars()
    def init_vars(self):
        self.lock.lock()
        self.all_vars['Matplotlib_scale'] = mpl_scale
        self.all_vars['Matplotlib_figure_scale'] = mpl_figure_scale
        self.all_vars['Font_scale'] = font_scale
        self.all_vars['Lastdir'] = last_dir_saved
        self.all_vars['MAX_THREADS'] = MAX_THREADS
        self.all_vars['PATH_data'] = paths['data']
        self.all_vars['PATH_corrections'] = paths['corrections']
        self.all_vars['PATH_overview'] = paths['overview']
        self.all_vars['PATH_xasplotting'] = paths['xasplotting']
        self.all_vars['PATH_postprocessing'] = paths['postprocessing']
        self.all_vars['PATH_variableviewer'] = paths['variableviewer'] 
        self.all_vars['PATH_beamline'] = paths['beamline']
        self.all_vars['PATH_fitting'] = paths['fitting']
        self.lock.unlock()
    @pyqtSlot(object)
    def update_vars(self, newdict):
        self.lock.lock()
        for kk in newdict.keys():
            self.all_vars[kk] = newdict[kk]
        self.lock.unlock()
        self.writeout()
    @pyqtSlot()
    def writeout(self):
        self.lock.lock()
        dump = open(self.confpath, 'w')
        for kk in self.all_vars.keys():
            line = str(kk)+': '+str(self.all_vars[kk])+'\n'
            dump.write(line)
        dump.close()
        self.lock.unlock()

class BetterDict():
    def __init__(self):
        params = {}
        params['Single'] = {}
        params['Profile'] = {}
        params['Server'] = {}
        for k in single_vars:
            params['Single'][k[1][1]] = ""
        for k in server_vars:
            params['Server'][k[1][1]] = ""
        self.dict = params
    def take_new_values(self, newvalues, input_keys):
        for kk in input_keys:
            self.dict[kk[0]][kk[1]] = copy.deepcopy(newvalues[kk[0]][kk[1]])

class QHLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

class InterfaceTabbed:
    def __init__(self, master, size, param,  appinstance = None, pathdict = {}):
        self.master = master
        paths = pathdict
        # self.master.resize(size[0], size[1]) 
        self.params = param
        self.temp_path = last_dir_saved
        self.max_threads = MAX_THREADS
        self.input_fields = {}
        self.input_keys = []
        self.profile_keys = []
        self.shortnames = []
        self.current_array = None
        self.current_data = None
        self.added_header = None
        self.current_file = ""
        self.overview_dir = paths['overview']
        self.temp_name = ""
        self.current_curvature_2ddata = None
        self.fouriered_data = None
        self.conf_writer = ConfigFileWriter()
        self.tabbar = QTabWidget(self.master)
        self.tabbar.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
        # self.tabbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabbar.setMinimumSize(QSize(int(size[0]/3), int(size[1]/3)))
        logtab = self.LogTab()
        self.logger("This is ADLER, Augmented Data Loading, Evaluation and Reduction.")
        self.logger("<b>You are using ADLER version " + str(ADLER_VERSION_STRING)+"</b>")
        self.logger("When in doubt about a part of the gui, refer to the <b>hovertext</b>.")
        self.logger("You could also try reading the <b>manual</b>.")
        self.logger("The configuration file .ADLconfig.txt is stored in your home directory.")
        # self.logger("The PyQt libraries are loaded from "+ str(testQt.__file__))
        # self.logger("Your PyQt version is "+ str(testQt.PYQT_VERSION_STR))
        # self.logger("Your Qt version is "+ str(testQt.QT_VERSION_STR))
        viewertab = LogviewerTab(self.tabbar, self.MakeCanvas(self.tabbar), self.log, startpath = paths['variableviewer'])
        beamlinetab = BeamlineTab(self.tabbar, self.MakeCanvas(self.tabbar), self.log, startpath = paths['beamline'])
        self.STab = SingleTab(self.tabbar, self.MakeCanvas(self.tabbar), self.log,  mthreads = self.max_threads, 
                                        startpath = paths['data'],  logplotter = viewertab,  app = appinstance)
        self.CTab = CorrectionsTab(self.tabbar, self.MakeCanvas(self.tabbar), self.log,  mthreads = self.max_threads, 
                                        startpath = paths['corrections'],  app = appinstance)
        self.PTab = PostprocessingTab(self.tabbar, self.MakeCanvas(self.tabbar), self.log,  mthreads = self.max_threads, 
                                        startpath = paths['postprocessing'],  app = appinstance)
        self.XTab = XASplottingTab(self.tabbar, self.MakeCanvas(self.tabbar), self.log,  mthreads = self.max_threads, 
                                        startpath = paths['xasplotting'],  app = appinstance)
        self.FitTab = FittingTab(self.tabbar, self.MakeCanvas(self.tabbar), self.log,  mthreads = self.max_threads, 
                                        startpath = paths['fitting'],  app = appinstance)
        filetab = self.MakeFileTab()
        self.tabbar.addTab(logtab, 'ADLER Log')
        self.tabbar.addTab(filetab, 'Overview/File Selector')
        self.tabbar.addTab(self.STab.base, 'RIXS Data Reduction')
        self.tabbar.addTab(self.PTab.base, 'Compare RIXS Curves')
        self.tabbar.addTab(self.XTab.base, 'Compare XAS Curves')
        self.tabbar.addTab(viewertab.base,  '.XAS File Viewer')
        self.tabbar.addTab(self.FitTab.base,  'Flexible Fitting')
        self.tabbar.addTab(beamlinetab.base,  'Beamline Commissioning')
        self.tabbar.addTab(self.CTab.base, 'RIXS Data Corrections')
        for atab in [self.STab, self.CTab, self.PTab, self.XTab, self.FitTab, viewertab, beamlinetab]:
            atab.conf_update.connect(self.conf_writer.update_vars)
        # self.CorrectionsTab()
        # self.master.setMinimumSize(QtCore.QSize(size[0], size[1]))
        # self.master.resize(size[0], size[1])
        self.master.setCentralWidget(self.tabbar)
        self.master.setFont(GlobFont)
        # self.master.resized.connect(self.on_resize)
        self.load_last_params()
        # self.save_last_params()
        self.logger("MAX_THREADS set to " + str(MAX_THREADS))
        self.logger("This value can be adjusted manually in ~/.ADLconfig.txt")
        self.confthread = QThread()
        if appinstance is not None:
            appinstance.aboutToQuit.connect(self.confthread.quit)
            # appinstance.aboutToQuit.connect(self.cleanup)
        self.master.destroyed.connect(self.confthread.quit)
        self.conf_writer.moveToThread(self.confthread)
        self.confthread.start()
    def on_resize(self):
        self.master.resize(self.master.sizeHint())
    def MakeFileTab(self):
        tab1 = QWidget(self.tabbar)
        # here comes the new layout
        topbar = QWidget(tab1)
        toplayout = QHBoxLayout(topbar)
        label1 = QLabel("Current directory", topbar)
        toplayout.addWidget(label1)
        bottombar = QWidget(tab1)
        bottomlayout = QHBoxLayout(bottombar)
        label2 = QLabel("Filter", bottombar)
        bottomlayout.addWidget(label2)
        # now the right side: file list
        self.dirline2 = QLineEdit("", parent = tab1)
        filterline = QLineEdit("", parent = tab1)
        self.tabview = TableView(tab1)
        self.sortfilter = QSortFilterProxyModel(tab1)
        self.buttonbar2 = QWidget(tab1)
        self.buttonlay2 = QHBoxLayout(self.buttonbar2)        
        button_list= [
        ['Pick Directory', self.PickFileDir, 'Select a directory containing PEAXIS data files.', 
            None, 'Files'], # 0,
        ['Load Files', self.PassSelectedFiles, 'Load the selected files into the RIXS Data Reduction tab.', 
            None, 'Files'], # 0,
            ]
        finished_buttons = []
        for bl in button_list:
            temp = self.MakeButton(self.buttonbar2, bl[0],  bl[1],  bl[2])
            if bl[3]:
                temp.setStyleSheet(bl[3])
            finished_buttons.append(temp)
            # self.buttonlay2.addWidget(temp) 
        # a tab for drag-and-drop assignment of files
        self.rightlayout = QVBoxLayout(tab1)
        toplayout.addWidget(self.dirline2)
        toplayout.addWidget(finished_buttons[0])
        bottomlayout.addWidget(filterline)
        bottomlayout.addWidget(finished_buttons[1])
        self.rightlayout.addWidget(topbar)
        self.rightlayout.addWidget(self.tabview)    
        self.rightlayout.addWidget(bottombar)           
        self.filemodel = PeaxisDataModel(tab1, ['name', 'Photon energy', 'Measuring time', 
                                                                    '2 theta',  'r1', 'MPTR-XPOS', 
                                                                    'Slit height', 'Temperature', 'Q'
                                                                    ])
        self.sortfilter.setSourceModel(self.filemodel)
        self.tabview.setModel(self.sortfilter)
        # self.tabview.setModel(self.filemodel)
        self.tabview.resizeColumnsToContents()
        self.tabview.setSortingEnabled(True)
        self.tabview.setDragEnabled(True)
        self.tabview.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.tabview.horizontalHeader().setSectionsMovable(True)
        self.tabview.horizontalHeader().setDragEnabled(True)
        self.tabview.horizontalHeader().setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.tabview.verticalHeader().setSectionsMovable(True)
        self.tabview.verticalHeader().setDragEnabled(True)
        self.tabview.verticalHeader().setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.tabview.show()
        self.tabview.release_items.connect(self.STab.load_file_indirect)
        filterline.textChanged.connect(self.sortfilter.setFilterRegularExpression)
        return tab1
    def LogTab(self):
        base = QWidget()
        base.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
        base_layout = QVBoxLayout()
        # layout = QFormLayout()
        box = LogBox(base)
        self.log = box
        box.setReadOnly(True)
        base_layout.addWidget(box)
        base.setLayout(base_layout)
        return base
        # self.tabbar.addTab(base, 'Log')
    def PickFileDir(self):
        result = QFileDialog.getExistingDirectory(self.master,
                        "Set the source folder",
                        self.overview_dir,
                        QFileDialog.Option.ShowDirsOnly
                        )
        # print(result)
        if result == "":
            return None
        else:
            self.filemodel.cleanup()
            self.filemodel.scanDir(result)
            self.dirline2.setText(result)
            self.overview_dir = result
            self.STab.set_extpath(result)
            self.conf_writer.update_vars({'PATH_overview': self.overview_dir})
            self.tabview.resizeColumnsToContents()
    def PassSelectedFiles(self):
        self.tabview.triggerList()
        self.tabbar.setCurrentIndex(2)
        self.tabview.resizeColumnsToContents()
    def save_last_params(self, lastfunction = None):
        try:
            source = open(os.path.join(expanduser("~"),'.ADLconfig.txt'), 'w')
        except:
            return None
        else:
            source.write('Lastdir: '+str(self.temp_path) + '\n')
            source.write('Lastfile: '+str(self.temp_name) + '\n')
            for kk in self.input_keys:
                source.write(" ".join([str(u) for u in [kk[0], kk[1], self.params[kk[0]][kk[1]]]]) + '\n')
            if not lastfunction == None:
                source.write('Last function called: ' + str(lastfunction) + '\n')
            source.write('Matplotlib_scale: ' + str(mpl_scale) + '\n')
            source.write('Matplotlib_figure_scale: ' + str(mpl_figure_scale) + '\n')
            source.write('Font_scale: ' + str(font_scale) + '\n')
            source.write('MAX_THREADS: ' + str(self.max_threads) + '\n')
            source.close()
    def load_last_params(self):
        try:
            source = open(os.path.join(expanduser("~"),'.ADLconfig.txt'), 'r')
        except:
            return None
        else:
            for line in source:
                toks = line.split()
                if len(toks) > 1:
                    if toks[0] == 'Lastdir:':
                        try:
                            self.temp_path = toks[1]
                        except:
                            pass
            source.close()
    def read_inputs(self, newparams, newfields = None):
        if newfields is None:
            newfields = self.input_fields
        temp_params = copy.deepcopy(newparams)
        for kk in self.input_keys:
            try:
                field = newfields[kk[0]][kk[1]]
            except:
                continue
            val = field.text()
            if type(val) == type([]):
                if len(val) == 1:
                    val = val[0]
            try:
                temp_params[kk[0]][kk[1]] = str(val)
            except:
                continue
        return temp_params
    def update_params(self, temp_params, newfields = None):
        if newfields is None:
            newfields = self.input_fields
        for kk in self.input_keys:
            val = temp_params[kk[0]][kk[1]]
            field = newfields[kk[0]][kk[1]]
            field.setText(str(val).strip('[])(\'\'""'))
        self.params = temp_params
        # print(self.canvasref.size())
    def logger(self, message):
        now = time.gmtime()
        timestamp = ("-".join([str(tx) for tx in [now.tm_mday, now.tm_mon, now.tm_year]])
                     + ',' + ":".join([str(ty) for ty in [now.tm_hour, now.tm_min, now.tm_sec]]) + '| ')
        self.log.setReadOnly(False)
        self.log.append(timestamp + message)
        self.log.setReadOnly(True)
    def MakeCanvas(self, parent):
        mdpi, winch, hinch = 75, 9.0*mpl_figure_scale, 7.0*mpl_figure_scale
        canvas = QWidget(parent)
        layout = QVBoxLayout(canvas)
        figure = mpl.figure(figsize = [winch, hinch], dpi=mdpi )#, frameon = False)
        figAgg = FigureCanvasQTAgg(figure)
        figAgg.setParent(canvas)
        figAgg.setSizePolicy(QSizePolicy.Policy.Minimum,QSizePolicy.Policy.Minimum)
        figAgg.updateGeometry()
        toolbar = NavigationToolbar2QTAgg(figAgg, canvas)
        toolbar.update()
        layout.addWidget(figAgg)
        layout.addWidget(toolbar)
        return canvas, figure, layout
    def MakeCanvasExtended(self, parent):
        mdpi, winch, hinch = 75, 9.0*mpl_figure_scale, 7.0*mpl_figure_scale
        canvas = QWidget(parent)
        layout = QVBoxLayout(canvas)
        figure = mpl.figure(figsize = [winch, hinch], dpi=mdpi )#, frameon = False)
        figAgg = FigureCanvasQTAgg(figure)
        figAgg.setParent(canvas)
        figAgg.setSizePolicy(QSizePolicy.Policy.Minimum,QSizePolicy.Policy.Minimum)
        figAgg.updateGeometry()
        toolbar = NavigationToolbar2QTAgg(figAgg, canvas)
        toolbar.update()
        layout.addWidget(figAgg)
        layout.addWidget(toolbar)
        figure2 = mpl.figure(figsize = [winch, hinch], dpi=mdpi )#, frameon = False)
        figAgg2 = FigureCanvasQTAgg(figure2)
        figAgg2.setParent(parent)
        figAgg2.setSizePolicy(QSizePolicy.Policy.Minimum,QSizePolicy.Policy.Minimum)
        figAgg2.updateGeometry()
        return canvas, figure, layout, figure2, figAgg2
    def MakeButton(self, parent, text, function, tooltip = ""):
        button = QPushButton(text, parent)
        if tooltip:
            button.setToolTip(tooltip)
        button.clicked.connect(function)
        # button.setSizePolicy(QSizePolicy.Maximum,QSizePolicy.Maximum)
        # button.setMinimumSize(QtCore.QSize(24, 10))
        # button.setMaximumSize(QtCore.QSize(500, 250))
        button.setFont(GlobFont)
        return button

def startGUI(some_args, pdict = None):
    app = QApplication(some_args)
    root = QMainWindow()
    root.setFont(GlobFont)
    root.setWindowTitle("ADLER 4.0: Augmented Data Loading, Evaluation and Reduction.")
    my_locale = QLocale(QLocale.Language.English, QLocale.Country.Ireland)
    # my_locale = QLocale(QLocale.Language.German, QLocale.Country.Germany)
    QLocale.setDefault(my_locale)
    # interface = InterfaceEmbedded(root, 800, params)    
    interface = InterfaceTabbed(root, (800, 700), params,  appinstance = app, pathdict = pdict) 
    QMetaObject.connectSlotsByName(root)
    root.show()
    app.exec()

if __name__ == '__main__':
    startGUI(sys.argv, pdict = paths)
