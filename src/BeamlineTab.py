
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
This file contains the ADLER Beamline Commissioning Tab.
It is a part of the GUI that is useful mainly to the instrument scientists.
"""

import os
import time
import copy
from os.path import expanduser

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot,  QSize
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QSizePolicy, QWidget, QTableWidget, QFormLayout, QFileDialog,   \
                                                QPushButton,  QVBoxLayout, QHBoxLayout, QTableWidgetItem, \
                                                QScrollArea
# from PyQt5 import sip
from VariablesGUI import VarBox
from ExtendGUI import AdlerTab
from ADLERcalc.ioUtils import load_only_logs, load_lise_logs
from ADLERcalc.fitUtils import fit_neon_gas, fit_n2_gas, fit_edge_profile, gauss_denum
from ADLERplot.Plotter import Plotter

mpl_scale = 1.0
mpl_figure_scale = 1.0
font_scale = 1.0
last_dir_saved = expanduser("~")
MAX_THREADS = 1

GlobFont = QFont('Sans Serif', int(12*font_scale))

oldval = 0.0

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

#### GUI part

plotting_variables = [
{'Name': 'Mutliplot offset maximum',  'Unit':'rel. intensity',  'Value':0.2,   'Key' : 'offmax', 
                               'MinValue':0.0,  'MaxValue':10.0,  'Length': 1,  'Type':'float',
                               'Comment':'The highest absolute value of the curve offset which can be set using the built-in slider in the plot.'},  
{'Name': 'Legend position',  'Unit':'N/A',  'Value':0,   'Key' : 'legpos', 
                               'MinValue':0,  'MaxValue':10,  'Length': 1,  'Type':'int',
                               'Comment':'The position of the legend in the plot. The non-negative values follow the \nmatplotlib definition, and negative values make the legend appear below the plot.'},  
]

fitting_variables = [
{'Name': 'Fixed Lorentz FWHM for N2',  'Unit':'meV',  'Value':120,   'Key' : 'n2width', 
                               'MinValue':0.0,  'MaxValue':1e7,  'Length': 1,  'Type':'float',
                               'Comment':'This part of the Voigt profile will be fixed in the fitting.'},  
{'Name': 'Fixed Lorentz FWHM for Neon',  'Unit':'meV',  'Value':254,   'Key' : 'neonwidth', 
                               'MinValue':0,  'MaxValue':1e7,  'Length': 1,  'Type':'float',
                               'Comment':'This part of the Voigt profile will be fixed in the fitting.'},  
{'Name': 'Number of linear functions for Edge',  'Unit':'N/A',  'Value':0,   'Key' : 'nlines', 
                               'MinValue':0,  'MaxValue':2,  'Length': 1,  'Type':'int',
                               'Comment':'The number of linear functions to be included in the fitting of the edge in the spectrum. \nIf set to 2, it is assumed that the slope of the linear background is different before and after the edge position.'},  
]

tabnames = ['Name', 'Length', 'Xlimits', 'Plot it?', 'Set as X', 'Set as Y']
class LogList(QObject):
    gotvals = pyqtSignal()
    def __init__(self, master,  nrows =1,  ncolumns = len(tabnames)):
        super().__init__(master)
        self.master = master
        self.Names = []
        self.Lengths = []
        self.Xmin = []
        self.Xmax = []
        self.Xunits = []
        self.useit = []
        self.arrays = []
        self.datcount = 0
        self.whichisX = -1
        self.whichisY = -1
        self.busy = True
        self.table = QTableWidget(nrows, ncolumns, self.master)
        self.table.setSizePolicy(QSizePolicy.Policy.MinimumExpanding,QSizePolicy.Policy.MinimumExpanding)
        for n in range(self.table.columnCount()):
            self.table.setItem(0, n, QTableWidgetItem(tabnames[n]))
        self.table.cellChanged.connect(self.update_values)
        self.gotvals.connect(self.redraw_table)
    def add_row(self, name, darray):
        self.busy = True
        rowitems = []
        dlen = len(darray)
        xmin = darray.min()
        xmax = darray.max()
        for nc in range(self.table.columnCount()):
            if nc == 0:
                try:
                    self.Names.append(name)
                except:
                    self.Names.append('?')
                rowitems.append(QTableWidgetItem(str(self.Names[-1])))
            elif nc == 1:
                try:
                    self.Lengths.append(int(dlen))
                except:
                    self.Lengths.append(-1)
                rowitems.append(QTableWidgetItem(str(self.Lengths[-1])))
            elif nc == 2:
                self.Xmin.append(xmin)
                self.Xmax.append(xmax)
                rowitems.append(QTableWidgetItem(','.join([str(xx) for xx in [xmin,  xmax]])))
            elif nc ==3:
                chkBoxItem = QTableWidgetItem()
                chkBoxItem.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                chkBoxItem.setCheckState(Qt.CheckState.Unchecked)
                rowitems.append(chkBoxItem)
                self.useit.append(0)
            elif nc ==4:
                chkBoxItem = QTableWidgetItem()
                chkBoxItem.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                chkBoxItem.setCheckState(Qt.CheckState.Unchecked)
                rowitems.append(chkBoxItem)
            elif nc ==5:
                chkBoxItem = QTableWidgetItem()
                chkBoxItem.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                chkBoxItem.setCheckState(Qt.CheckState.Unchecked)
                rowitems.append(chkBoxItem)
        lastrow = self.table.rowCount()
        self.table.insertRow(lastrow)
        for n, ri in enumerate(rowitems):
            self.table.setItem(lastrow, n, ri)
        self.datcount += 1
        self.arrays.append(darray)
        self.busy = False
    def clear_table(self):
        self.Names = []
        self.Lengths = []
        self.Xmin = []
        self.Xmax = []
        self.Xunits = []
        self.useit = []
        self.arrays = []
        self.datcount = 0
        self.whichisX = -1
        self.whichisY = -1
        for nr in range(1, self.table.rowCount())[::-1]:
            self.table.removeRow(nr)
        self.gotvals.emit()
    def return_values(self):
        xarray = []
        yarray = []
        darrays = []
        for n, bl in enumerate(self.useit):
            if n == self.whichisX:
                xarray = (self.Names[n],  self.arrays[n])
            elif n == self.whichisY:
                yarray = (self.Names[n],  self.arrays[n])
            elif bl:
                darrays.append((self.Names[n],  self.arrays[n]))
        return xarray, yarray, darrays
    @pyqtSlot(int, int)
    def update_values(self,  row =0,  column=0):
        if self.busy:
            return None
        if row > 0:
            if column > 2:
                bval = self.table.item(row, column).checkState()
            else:
                return None
            if column == 3:
                self.useit[row-1] = not self.useit[row-1]
            elif column == 4:
                self.whichisX = row -1
            elif column ==5:
                self.whichisY = row -1
        else:
            return None
        self.gotvals.emit()
    @pyqtSlot()
    def redraw_table(self):
        self.table.blockSignals(True)
        for nr in range(1,  self.table.rowCount()):
            for nc in range(3, 6):
#                if nc ==3:
#                    if self.useit[nr-1]:
#                        self.table.item(nr, 3).setCheckState(Checked)
#                    else:
#                        self.table.item(nr, 3).setCheckState(Unchecked)
#                elif nc == 4:
#                    if nr == (self.whichisX + 1):
#                        self.table.item(nr, 4).setCheckState(Checked)
#                    else:
#                        self.table.item(nr, 4).setCheckState(Unchecked)
#                elif nc == 5:
#                    if nr == (self.whichisY + 1):
#                        self.table.item(nr, 5).setCheckState(Checked)
#                    else:
#                        self.table.item(nr, 5).setCheckState(Unchecked)
                if nc ==3:
                    if self.useit[nr-1]:
                        self.table.item(nr, 3).setCheckState(Qt.CheckState.Checked)
                    else:
                        self.table.item(nr, 3).setCheckState(Qt.CheckState.Unchecked)
                elif nc == 4:
                    if nr == (self.whichisX + 1):
                        self.table.item(nr, 4).setCheckState(Qt.CheckState.Checked)
                    else:
                        self.table.item(nr, 4).setCheckState(Qt.CheckState.Unchecked)
                elif nc == 5:
                    if nr == (self.whichisY + 1):
                        self.table.item(nr, 5).setCheckState(Qt.CheckState.Checked)
                    else:
                        self.table.item(nr, 5).setCheckState(Qt.CheckState.Unchecked)
        self.table.blockSignals(False)

class BeamlineTab(AdlerTab):
    def __init__(self, master,  canvas,  log,  startpath = None):
        super().__init__(master)
        self.master = master
        # self.progbar = None
        self.canvas, self.figure, self.clayout = canvas
        self.params = [(plotting_variables,  "Plotting"), (fitting_variables,  "Fitting")]
        self.boxes = self.make_layout()
        self.boxes[1].values_changed.connect(self.update_fitpars)
        self.parnames = []
        self.pardict = {}
        self.log = log
        self.filelist = None
        self.fwhm_n2, self.fwhm_neon,  self.nlines_edge = None, None, None
        self.currentpath = startpath
        self.plotter = Plotter(figure = self.figure)
        self.update_fitpars()
        # self.curve_list.gotvals.connect(self.core.take_table_values)
        # self.flip_buttons()
    @pyqtSlot()
    def update_fitpars(self):
        names,  vdict = self.boxes[1].returnValues()
        self.fwhm_n2 = vdict['n2width'][0]
        self.fwhm_neon = vdict['neonwidth'][0]
        self.nlines_edge = vdict['nlines'][0]
    def make_layout(self):
        col1 = "background-color:rgb(30,180,20)"
        col2 = "background-color:rgb(0,210,210)"
        col3 = "background-color:rgb(50,150,250)"
        #
        button_list= [
        ['Plot 1D', self.plot_logs, 'Plot the highlighted curves as a function of X.', 
            col1, 'Plotting'], # 0
        ['Plot Grid', self.plot_logmap, 'Plot the first highlighted dataset as a function of X and Y.', 
            col1, 'Plotting'], # 1
        ['Clear List', self.clear_list, 'Remove all the data from the list.', 
            col1, 'Data Handling'], # 2
        ['Load LISE files', self.load_LISE, 'Load one or more LISE files.', 
            '', 'Data Handling'], # 2
        ['Load XAS files', self.load_logs, 'Load one or more XAS files.', 
            '', 'Data Handling'], # 2
        ['Fit Neon', self.fit_neon, 'Fit the position of the Ne peak for beamline energy calibration.', 
            col3, 'Fitting'], # 0
        ['Fit Nitrogen', self.fit_nitrogen, 'Fit the width of the N2 peaks for the beamline resolution test.', 
            col3, 'Fitting'], # 1
        ['Fit Edge', self.fit_edge, 'Fit the curve with a step function.', 
            col3, 'Fitting'], # 1
        ]
        self.button_list = []
        # base = QWidget(self.master)
        base=self.base
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
        button_base.setMinimumHeight(40*len(button_dict.keys()))
        button_base.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        button_layout = QFormLayout(button_base)
        button_layout.setVerticalSpacing(2)
        button_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.curve_list = LogList(base)
        boxes_layout.addWidget(button_base)
        boxes_layout.addWidget(self.curve_list.table)
        # self.progbar = QProgressBar(base)
        boxes = []
        for el in self.params:
            temp = VarBox(boxes_base, el[0],  el[1])
            boxes.append(temp)
            boxes_layout.addWidget(temp.base)
            # temp.values_changed.connect(self.read_inputs)
        boxes_layout.addWidget(self.progbar)
        # structure of vars: label, dictionary keys, tooltip
        self.active_buttons = np.zeros(len(button_list)).astype(np.int)
        self.active_buttons[0:2] = 1
        for k in button_dict.keys():
            bbase = QWidget(button_base)
            blayout = QHBoxLayout(bbase)
            for but in button_dict[k]:
                blayout.addWidget(but)
            button_layout.addRow(k,  bbase)
        self.button_base = button_base
        self.boxes_base = boxes_base
        return boxes
    def takeData(self,  vardict):
        try:
            nvals = len(vardict.keys())
        except:
            self.logger("No variable logs loaded.")
        else:
            if nvals > 0:
                for kk in vardict.keys():
                    self.curve_list.add_row(kk, vardict[kk])
            else:
                self.logger("No variable logs loaded.")
    def logger(self, message):
        now = time.gmtime()
        timestamp = ( "BeamlineTab "
                     + "-".join([str(tx) for tx in [now.tm_mday, now.tm_mon, now.tm_year]])
                     + ',' + ":".join([str(ty) for ty in [now.tm_hour, now.tm_min, now.tm_sec]]) + '| ')
        self.log.setReadOnly(False)
        self.log.append(timestamp + message)
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
    def plot_logs(self):
        x, y, d = self.curve_list.return_values()
        if len(d) == 0:
            self.logger("No free data sets left for plotting.")
            return None
        if len(x) == 0:
            self.logger("No dedicated X axis chosen. Plotting against row numbers.")
            nox = True
        else:
            nox = False
        datsets = []
        axlabels = []
        for ds in d:
            if nox:
                xax = np.arange(len(ds[1]))
                xlab = '# points'
            else:
                xax = x[1][:min(len(x[1]),  len(ds[1]))]
                xlab = x[0]
            datset = np.column_stack([xax, ds[1][:len(xax)]])
            axlabels.append([xlab, ds[0]])
            datsets.append(datset)
        self.plotter.plot1D_grid(datsets, fig = self.figure, text = '', label_override = axlabels)
    def fit_neon(self):
        x, y, d = self.curve_list.return_values()
        if len(x) == 0 or len(y) == 0:
            self.logger("Grid mapping requires both X and Y to be set.")
            return None
        xlab = x[0]
        ylab = y[0]
        xpoints =x[1]
        ypoints = y[1]
        try:
            profile, gauss_sigma, lorentz_gamma,  peak_pos,  GOF = fit_neon_gas(np.column_stack([xpoints, ypoints]),
                                     fixed_gamma = self.fwhm_neon / 1000.0 / 2.0)
        except:
            profile, gauss_sigma, lorentz_gamma,  peak_pos,  GOF = np.zeros(xpoints.shape),  -1.0, -1.0, [], 1e9
        res_text = 'Gauss FWHM = '+str(round(gauss_sigma*1000.0*gauss_denum, 4))+' meV\n'+\
                  'Lorentz FWHM = '+str(round(lorentz_gamma, 4)*2000.0) + ' meV\n'+\
                  'Peak positions = ' + ", ".join([str(x) for x in np.round(peak_pos, 3)]) + '\n'+\
                  'Fit Quality = ' + str(round(1.0/GOF, 3))
        self.logger('Neon fitting results:\n'+res_text)
        self.plotter.plot1D([np.column_stack([xpoints, ypoints]), np.column_stack([xpoints, profile])], outFile = "", fig = self.figure,
                  text = res_text, 
                  label_override = [xlab, ylab], curve_labels= ['Data', 'Fit'], 
                  legend_pos = 0)
    def fit_nitrogen(self):
        x, y, d = self.curve_list.return_values()
        if len(x) == 0 or len(y) == 0:
            self.logger("Grid mapping requires both X and Y to be set.")
            return None
        xlab = x[0]
        ylab = y[0]
        xpoints =x[1]
        ypoints = y[1]
        try:
            profile, gauss_sigma, lorentz_gamma,  peak_pos,  GOF = fit_n2_gas(np.column_stack([xpoints, ypoints]),
                         fixed_gamma = self.fwhm_n2 / 1000.0 / 2.0)
        except:
            profile, gauss_sigma, lorentz_gamma, peak_pos,  GOF = np.zeros(xpoints.shape),  -1.0, -1.0, [],  1e9
        res_text = 'Gauss FWHM = '+str(round(gauss_sigma*1000.0*gauss_denum, 4))+' meV\n'+\
                  'Lorentz $\gamma$ = '+str(round(lorentz_gamma, 4)*2000.0) + ' meV\n'+\
                  'Peak positions = ' + ", ".join([str(x) for x in np.round(peak_pos, 3)]) + '\n'+\
                  'Fit Quality = ' + str(round(1.0/GOF, 3))
        self.logger('Nitrogen fitting results:\n'+res_text)
        self.plotter.plot1D([np.column_stack([xpoints, ypoints]), np.column_stack([xpoints, profile])], outFile = "", fig = self.figure,
                  text = res_text, 
                  label_override = [xlab, ylab], curve_labels= ['Data', 'Fit'], 
                  legend_pos = 0)
    def fit_edge(self):
        x, y, d = self.curve_list.return_values()
        if len(x) == 0 or len(y) == 0:
            self.logger("Grid mapping requires both X and Y to be set.")
            return None
        xlab = x[0]
        ylab = y[0]
        xpoints =x[1]
        ypoints = y[1]
        try:
            profile, edge_pos, edge_width,  GOF = fit_edge_profile(np.column_stack([xpoints, ypoints]), nlines = self.nlines_edge)
        except:
            profile, edge_pos, edge_width,  GOF = np.zeros(xpoints.shape),  -1.0, -1.0, 1e9
        self.plotter.plot1D([np.column_stack([xpoints, ypoints]), np.column_stack([xpoints, profile])], outFile = "", fig = self.figure,
                  text = 'Edge position = '+str(round(edge_pos, 4))+'\n'+'Edge width = '+str(round(edge_width, 4)) + '\n'+
                  'Fit Quality = ' + str(round(1.0/GOF, 3)), 
                  label_override = [xlab, ylab], curve_labels= ['Data', 'Fit'], 
                  legend_pos = 0)
    def plot_logmap(self):
        x, y, d = self.curve_list.return_values()
        if len(x) == 0 or len(y) == 0:
            self.logger("Grid mapping requires both X and Y to be set.")
            return None
        if len(d) == 0:
            self.logger("Grid mapping requires another data set other than X and Y.")
            return None
        xlab = x[0]
        ylab = y[0]
        dlab = d[0][0]
        xpoints =x[1]
        ypoints = y[1]
        data = d[0][1]
        xlen,  ylen = len(xpoints),  len(ypoints)
        gridlen = 1+2*int(round(max(xlen, ylen)**0.5))
        grid = np.zeros((gridlen, gridlen))
        norm = np.zeros((gridlen, gridlen))
        xlims = np.linspace(xpoints.min(),  xpoints.max(),  gridlen+1)[1:]
        ylims = np.linspace(ypoints.min(),  ypoints.max(),  gridlen+1)[1:]
        for n in np.arange(len(data)):
            xind = np.argmax(xpoints[n] < xlims)
            yind = np.argmax(ypoints[n] < ylims)
            grid[xind, yind] += data[n]
            norm[xind, yind] += 1
        grid /= norm
        grid = np.nan_to_num(grid)
        self.plotter.plot2D_sliders(grid.T, [(ypoints[0], ypoints[-1]), (xpoints[0], xpoints[-1])], fig = self.figure, text = dlab, interp = 'none', 
                               labels = [xlab,ylab])
    def clear_list(self):
        self.curve_list.clear_table()
    def load_logs(self):        
        result, ftype = QFileDialog.getOpenFileNames(self.master,
           'Load PEAXIS log files (.XAS files):', self.currentpath,
           'PEAXIS variable log (*.xas);;All files (*.*)')
        if len(result) > 0:
            self.curve_list.clear_table()
            templog = load_only_logs(result)
            self.takeData(templog)
            newpath, shortname = os.path.split(result[0])
            self.conf_update.emit({'PATH_beamline': newpath})
            self.currentpath = newpath
    def load_LISE(self):        
        result, ftype = QFileDialog.getOpenFileNames(self.master,
           'Load LISE scan files (.txt files):', self.currentpath,
           'LISE scan (*.txt);;All files (*.*)')
        if len(result) > 0:
            self.curve_list.clear_table()
            templog = load_lise_logs(result)
            self.takeData(templog)
            newpath, shortname = os.path.split(result[0])
            self.conf_update.emit({'PATH_beamline': newpath})
            self.currentpath = newpath
