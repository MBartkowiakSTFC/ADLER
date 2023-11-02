
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
The ADLER tab for fitting arbitrary curves.
It is still considered work in progress.
"""

import os
import time
from os.path import expanduser

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot,  QSize,  QThread
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QSizePolicy, QWidget, QFileDialog, QTableWidget, QFormLayout,   \
                                                QPushButton,  QVBoxLayout, QHBoxLayout, QTableWidgetItem, \
                                                QLineEdit, QSpinBox, QLabel, QDockWidget, QGridLayout, \
                                                QDialog, QTextEdit, QScrollArea
# from PyQt6 import sip
from VariablesGUI import VarBox
from ADLERcalc.FitCore import FitCore
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

import matplotlib.pyplot as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar2QTAgg
from matplotlib.widgets import Slider


GlobFont = QFont('Sans Serif', int(12*font_scale))

oldval = 0.0

#### GUI part

loading_variables = [
{'Name': 'Curve cutoff',  'Unit':'?',  'Value':np.array([-1e5, 1e5]),  'Key' : 'cuts', 
                               'MinValue':-1e15*np.array([1.0, 1.0]),
                               'MaxValue':1e15*np.array([1.0, 1.0]),  'Length': 2,  'Type':'float',
                               'Comment':'Each curve plotted in this tab will be truncated to the x limits specified here.'}, 
{'Name': 'Reduction factor',  'Unit':'N/A',  'Value':1.0, 'Key' : 'redfac', 
                                      'MinValue':0.25,
                                      'MaxValue':10.0,
                                      'Length': 1,  'Type':'float',
                               'Comment':'Higher values of reduction factor lead to coarser binning of the data, \nwhich can be useful for plotting noisy data sets which do not require high resolution.'}, 
]
fit_variables = [
{'Name': 'Max. number of peaks',  'Unit':'',  'Value':1,  'Key' : 'maxpeaks', 
                               'MinValue':0,  'MaxValue':1e5,  'Length': 1,  'Type':'int',
                               'Comment':'The maximum number of peaks used in automatic fitting. \nThe fitting will be performed in iterations, from 0 peaks to the number specified here.'},  
{'Name': 'Background polynomial order',  'Unit':'',  'Value':1,  'Key' : 'polyorder', 
                               'MinValue':0,  'MaxValue':1e2,  'Length': 1,  'Type':'int',
                               'Comment':'0=constant, 1=linear, 2=quadratic, etc.'}, 
{'Name': 'Fixed Lorentzian width',  'Unit':'x axis units',  'Value':-1.0,  'Key' : 'fixedL', 
                               'MinValue':-1.0,  'MaxValue':1e9,  'Length': 1,  'Type':'float',
                               'Comment':'If positive, it will fix the Lorentzian width in the Voigt profile to the value specified here.'}, 
{'Name': 'Fixed Gaussian width',  'Unit':'x axis units',  'Value':-1.0,  'Key' : 'fixedG', 
                               'MinValue':-1.0,  'MaxValue':1e9,  'Length': 1,  'Type':'float',
                               'Comment':'If positive, it will fix the Gaussian width in the Voigt profile to the value specified here.'}, 
{'Name': 'Add edge to background',  'Unit':'yes/no',  'Value': 0,  'Key' : 'useedge', 
                               'MinValue':0,  'MaxValue':1,  'Length': 1,  'Type':'int',
                               'Comment':'If non-zero, it will fit a step function on top of the polynomial background.'}, 
{'Name': 'Overshoot penalty',  'Unit':'N/A',  'Value':1.0,  'Key' : 'penalty', 
                               'MinValue':0.0,  'MaxValue':1e15,  'Length': 1,  'Type':'float',
                               'Comment':'It may improve the stability of the iterative fitting to keep the fitted functions from assuming values higher than the data points. \nThis parameter scales the penalty for the fit curve going over the data points. The default value means normal least-square fitting with no asymmetry.'}
]
plotting_variables = [
{'Name': 'Mutliplot offset maximum',  'Unit':'rel. intensity',  'Value':0.2,   'Key' : 'offmax', 
                               'MinValue':0.0,  'MaxValue':10.0,  'Length': 1,  'Type':'float',
                               'Comment':'The highest absolute value of the curve offset which can be set using the built-in slider in the plot.'},  
{'Name': 'Legend position',  'Unit':'N/A',  'Value':0,   'Key' : 'legpos', 
                               'MinValue':0,  'MaxValue':10,  'Length': 1,  'Type':'int',
                               'Comment':'The position of the legend in the plot. The non-negative values \nfollow the matplotlib definition, and negative values make the legend appear below the plot.'},   
]

tabnames = ['A', 'Centre', 'Lorentz width', 'Gauss width']
class ProfileList(QObject):
    gotvals = pyqtSignal()
    needanupdate = pyqtSignal()
    def __init__(self, master,  nrows =1,  ncolumns = len(tabnames)):
        super().__init__(master)
        self.master = master
        self.Ei = []
        self.Xmin = []
        self.Xmax = []
        self.Xunits = []
        self.useit = []
        self.useit2 = []
        self.busy = True
        self.table = QTableWidget(nrows, ncolumns, self.master)
        self.table.setSizePolicy(QSizePolicy.Policy.MinimumExpanding,QSizePolicy.Policy.MinimumExpanding)
        for n in range(self.table.columnCount()):
            self.table.setItem(0, n, QTableWidgetItem(tabnames[n]))
        self.table.cellChanged.connect(self.update_values)
        self.table.cellChanged.connect(self.update_ticks)
        # self.needanupdate.connect(self.redraw_table)
    def add_row(self, data):
        self.busy = True
        self.table.blockSignals(True)
        rowitems = []
        lastnit = len(data)
        for nit, dud in enumerate(data):
            d = str(dud)
            rowitems.append(QTableWidgetItem(str(d).strip("()[]'")))
        lastrow = self.table.rowCount()
        self.table.insertRow(lastrow)
        for n, ri in enumerate(rowitems):
            self.table.setItem(lastrow, n, ri)
        # for nit in range(lastnit, self.table.columnCount()):
        #     self.table.setItem(lastrow, nit, QTableWidgetItem(str("")))
        self.busy = False
        self.table.blockSignals(False)
        self.needanupdate.emit()
    @pyqtSlot()
    def clear_table(self):
        self.Ei = []
        self.Xmin = []
        self.Xmax = []
        self.Xunits = []
        self.useit = []
        self.useit2 = []
        for nr in range(1, self.table.rowCount())[::-1]:
            self.table.removeRow(nr)
        self.gotvals.emit()
    def return_values(self):
        final = []
        for nr in range(len(self.useit)):
            if self.useit[nr] or self.useit2[nr]:
                rowdata = [nr]
                rowdata += [self.Xmin[nr],  self.Xmax[nr], self.useit[nr], self.useit2[nr]]
                final.append(rowdata)
        return final
    @pyqtSlot(int, int)
    def update_values(self,  row =0,  column=0):
        if self.busy:
            return None
        self.busy = True
        for nr in range(1,  self.table.rowCount()):
            for nc in range(0, 4):
                if nc == 1:
                    try:
                        vals = [float(tok.strip("()[]'")) for tok in self.table.item(nr, nc).text().split(',')]
                    except:
                        continue
                    self.Xmin[nr-1] = vals[0]
                    self.Xmax[nr-1] = vals[1]
                # elif nc ==4:
                    # # tempdebug = self.table.item(nr, 4).checkState()
                    # # self.useit[nr-1] = (self.table.item(nr, 4).checkState() == Qt.Qt.Checked)
                    # self.useit[nr-1] = not self.useit[nr-1]
        self.busy = False
        self.needanupdate.emit()
        self.gotvals.emit()
    @pyqtSlot(int, int)
    def update_ticks(self,  row =0,  column=0):
        if self.busy:
            return None
        self.busy = True
        for nr in [row]:
            for nc in [column]:
                if nc ==2:
                    # tempdebug = self.table.item(nr, 4).checkState()
                    # self.useit[nr-1] = (self.table.item(nr, 4).checkState() == Qt.Qt.Checked)
                    self.useit[nr-1] = not self.useit[nr-1]
                if nc ==3:
                    # tempdebug = self.table.item(nr, 4).checkState()
                    # self.useit[nr-1] = (self.table.item(nr, 4).checkState() == Qt.Qt.Checked)
                    self.useit2[nr-1] = not self.useit2[nr-1]
        self.busy = False
        self.needanupdate.emit()
        self.gotvals.emit()
    @pyqtSlot()
    def redraw_table(self):
        if self.busy:
            return None
        self.busy = True
        self.table.blockSignals(True)
        for nr in range(1,  self.table.rowCount()):
            for nc in range(1, 3):
                if nc == 1:
                    temp = ",".join([str(x) for x in [self.Xmin[nr-1], self.Xmax[nr-1]]])
                    self.table.item(nr, nc).setText(temp)
                elif nc ==2:
                    if self.useit[nr-1]:
                        self.table.item(nr, 2).setCheckState(Qt.CheckState.Checked)
                    else:
                        self.table.item(nr, 2).setCheckState(Qt.CheckState.Unchecked)
                elif nc ==3:
                    if self.useit2[nr-1]:
                        self.table.item(nr, 3).setCheckState(Qt.CheckState.Checked)
                    else:
                        self.table.item(nr, 3).setCheckState(Qt.CheckState.Unchecked)
        self.busy = False
        self.table.blockSignals(False)
    def assign_fitparams(self, fitparams):
        nums, width,  widtherr,  area,  areaerr = fitparams
        for en,  realnum in enumerate(nums):
            nr = realnum + 1
            for nc in range(5, 9):
                if nc == 5:
                    self.table.item(nr, nc).setText(str(width[en]))
                if nc == 6:
                    self.table.item(nr, nc).setText(str(widtherr[en]))
                if nc == 7:
                    self.table.item(nr, nc).setText(str(area[en]))
                if nc == 8:
                    self.table.item(nr, nc).setText(str(areaerr[en]))

class CurveDialog(QDialog):
    values_ready = pyqtSignal(object)
    def __init__(self, parent, filename, appinstance = None):
        super().__init__(parent.base)
        self.fname = filename
        self.layout = QHBoxLayout()
        rightpanel = QWidget(self)
        rightlayout = QVBoxLayout(rightpanel)
        self.button = QPushButton('Accept Data', self)
        self.button.clicked.connect(self.send_values_back)
        self.button.clicked.connect(self.accept)
        self.button.setDefault(True)
        self.nobutton = QPushButton('Cancel', self)
        buttonbar = QWidget(rightpanel)
        buttonlayout = QHBoxLayout(buttonbar)
        buttonlayout.addWidget(self.nobutton)
        buttonlayout.addWidget(self.button)
        # self.nobutton.clicked.connect(self.send_nothing_back)
        self.nobutton.clicked.connect(self.reject)
        formbase = QWidget(self)
        formlay = QFormLayout(formbase)
        self.commentfield = QLineEdit("#", formbase)
        self.separatorfield = QLineEdit(",", formbase)
        self.spinx = QSpinBox(formbase)
        self.spiny = QSpinBox(formbase)
        self.spine = QSpinBox(formbase)
        self.spinx.setValue(1)
        self.spiny.setValue(2)
        self.spine.setValue(-1)
        for s in [self.spinx, self.spiny, self.spine]:
            s.valueChanged.connect(self.new_columns)
            s.valueChanged.connect(self.update_plot)
        for s in [self.commentfield, self.separatorfield]:
            s.textChanged.connect(self.new_characters)
            s.textChanged.connect(self.update_preview)
            # s.textChanged.connect(self.update_plot)
        self.xcol = 0
        self.ycol = 1
        self.ecol = -1
        self.comm = '#'
        self.sepp = ','
        formlay.addRow("Comment character", self.commentfield)
        formlay.addRow("Separator character", self.separatorfield)
        formlay.addRow("X column", self.spinx)
        formlay.addRow("Y column", self.spiny)
        formlay.addRow("Error column", self.spine)
        self.headerview = QTextEdit(self)
        self.dataview = QTextEdit(self)
        rightlayout.addWidget(formbase)
        rightlayout.addWidget(self.headerview)
        rightlayout.addWidget(self.dataview)
        rightlayout.addWidget(buttonbar)
        c, f, l = self.MakeCanvas(self)
        self.layout.addWidget(c)
        self.fig = f
        self.plotter = Plotter(figure = self.fig)
        self.layout.addWidget(rightpanel)
        # self.layout.addWidget(self.button)
        self.setLayout(self.layout)
        self.setWindowTitle("Select a curve for fitting with ADLER")
        self.new_columns()
        self.new_characters()
        self.update_preview()
        self.update_plot()
    def MakeCanvas(self, parent):
        mdpi, winch, hinch = 75, 6.0*mpl_figure_scale, 4.5*mpl_figure_scale
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
    @pyqtSlot()
    def new_columns(self):
        self.xcol = self.spinx.value() -1
        self.ycol = self.spiny.value() -1
        self.ecol = self.spine.value() -1
    @pyqtSlot()
    def new_characters(self):
        self.comm = self.commentfield.text()
        self.sepp = self.separatorfield.text()
        # temp = self.separatorfield.text()
        # if len(temp) > 0:
        #     self.sepp = temp
    @pyqtSlot()
    def update_preview(self):
        source = open(self.fname, 'r')
        headlines,  datalines = [], []
        for line in source:
            toks = line.split()
            if len(toks) < 1:
                continue
            if self.comm in toks[0]:
                headlines.append(line.strip('\n'))
            else:
                datalines.append(line.split(self.comm)[0].strip('\n'))
        source.close()
        self.headerview.clear()
        self.headerview.insertPlainText('\n'.join(headlines))
        self.dataview.clear()
        self.dataview.insertPlainText('\n'.join(datalines))
        self.textdata = datalines
        self.update_plot()
    @pyqtSlot()
    def update_plot(self):
        data = []
        xarr, yarr, earr = None, None, None
        for line in self.textdata:
            if len(self.sepp) > 0:
                toks = line.split(self.sepp)
            else:
                toks = line.split()
            try:
                nums = [float(x) for x in toks]
            except:
                # print(len)
                continue
            data.append(nums)
        if len(data) < 1:
            self.thecurve = None
            self.fig.clear()
            self.fig.canvas.draw()
            print("Fitting Dialog: not enough data lines! Length = ",  len(data))
            return None
        data = np.array(data)
        try:
            ncol = data.shape[1]
        except:
            print("Fitting Dialog: strange data shape: ", data.shape)
            self.thecurve = None
            self.fig.clear()
            self.fig.canvas.draw()
            return None
        if self.xcol < ncol:
            xarr = data[:, self.xcol]
        else:
            xarr = None
        if self.ycol < ncol:
            yarr = data[:, self.ycol]
        else:
            yarr = None
        if self.ecol >= 0:
            if self.ecol < ncol:
                earr = data[:, self.ecol]
            else:
                earr = None
        if xarr is None or yarr is None:
            print("Fitting Dialog: x or y arrays are empty.")
            self.thecurve = None
            self.fig.clear()
            self.fig.canvas.draw()
            return None
        else:
            if earr is None:
                pic = [np.column_stack([xarr, yarr])]
            else:
                pic = [np.column_stack([xarr, yarr, earr])]
            self.thecurve = pic[0]
            self.plotter.plot1D(pic,  fig=self.fig)
    def send_values_back(self):
        pdict = {}
        pdict['separator'] =  self.sepp
        pdict['comment'] = self.comm
        pdict['xcolumn'] = self.xcol
        pdict['ycolumn'] = self.ycol
        pdict['ecolumn'] = self.ecol
        self.values_ready.emit(pdict)
        
        
class FittingTab(AdlerTab):
    for_loading = pyqtSignal(object)
    clear = pyqtSignal()
    start_seq_fit = pyqtSignal()
    start_glob_fit = pyqtSignal()
    def __init__(self, master,  canvas,  log,  mthreads = 1, startpath = None,  app = None):
        super().__init__(master)
        self.master = master
        # self.progbar = None
        self.log = log
        self.canvas, self.figure, self.clayout = canvas
        self.plotter = Plotter(figure = self.figure)
        self.params = [(loading_variables, "File Loading"),
                               (plotting_variables,  "Plotting"),
                               (fit_variables,  "Fitting")]
        self.profile_list = ProfileList(self.base)
        self.core = FitCore(None,  None,  self.log,  max_threads = mthreads, 
                                        table = self.profile_list,  startpath = startpath, 
                                        progress_bar = self.progbar)
        self.currentpath = startpath
        self.boxes = self.make_layout()
        self.core.assign_boxes(self.boxes)
        self.parnames = []
        self.pardict = {}
        self.filelist = None
        # self.destroyed.connect(self.cleanup)
        self.profile_list.gotvals.connect(self.core.take_table_values)
        self.core.cleared.connect(self.profile_list.clear_table)
        self.core.fileparams.connect(self.finish_loading)
        # self.core.finished_fitting.connect(self.showfits)
        self.core.finished_overplot.connect(self.showmulti)
        self.core.finished_rixsmap.connect(self.showrixs)
        self.core.finished_fitting.connect(self.summary)
        # self.core.finished_merge.connect(self.add_mergedcurve)
        self.fitnum_spinbox.valueChanged.connect(self.plot_specific)
        # self.core.finished_filter.connect(self.add_filteredcurves)
        self.flip_buttons()
        #
        self.for_loading.connect(self.core.load_profiles)
        self.clear.connect(self.core.clear_profiles)
        self.start_seq_fit.connect(self.core.sequential_fit)
        self.start_glob_fit.connect(self.core.global_fit)
        #
        self.core.cleared.connect(self.flip_buttons)
        self.core.finished_fitting.connect(self.flip_buttons)
        self.core.finished_overplot.connect(self.flip_buttons)
        self.core.finished_rixsmap.connect(self.flip_buttons)
        self.core.finished_merge.connect(self.flip_buttons)
        #
        self.corethread = QThread()
        if app is not None:
            app.aboutToQuit.connect(self.corethread.quit)
            app.aboutToQuit.connect(self.cleanup)
        self.base.destroyed.connect(self.corethread.quit)
        self.core.moveToThread(self.corethread)
        self.corethread.start()
    @pyqtSlot()
    def cleanup(self):
        self.corethread.quit()
        self.corethread.wait()
    def make_layout(self):
        col1 = "background-color:rgb(30,180,20)"
        col2 = "background-color:rgb(0,210,210)"
        col3 = "background-color:rgb(50,150,250)"
        #
        button_list= [
        ['Load a Profile', self.load_profile_button, 'Pick the 1D profiles to be loaded.', 
            '', 'Files'], # 0
        ['Clear Results', self.clear_profile_button, 'Remove all profiles from the list.', 
            '', 'Files'], # 1
        ['Sequential Fit', self.sequential_fit, 'Fit the curve using a iterative algorithm.', 
            col1, 'Fitting'], # 2
        # ['Global Fit', self.global_fit, 'Fit the curve using a global minimum search algorithm.', 
        #     col1, 'Fitting'], # 3
        ['Show breakdown', self.core.return_fitpars, 'Show the fitting parameters in a table.', 
            col2, 'Results'], # 4
        ['Save fitting results', self.save_fit_results, 'Save the fit results to a file.', 
            '', 'Results'], # 5
        ]
        self.button_list = []
        # base = QWidget(self.master)
        base = self.base
        base.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        # file name display
        topbar = QWidget(base)
        topbar_layout = QHBoxLayout(topbar)
        toplabel = QLabel("File name: ", topbar)
        fnamefield = QLineEdit("",  topbar)
        topbar_layout.addWidget(toplabel)
        topbar_layout.addWidget(fnamefield)
        self.fname_edit = fnamefield
        # file name display finished
        base_layout = QHBoxLayout(base)
        base_layout.addWidget(self.canvas)
        boxes_base = QWidget(base)
        boxes_layout = QVBoxLayout(boxes_base)
        scroll = QScrollArea(widgetResizable=True)
        scroll.setWidget(boxes_base)
        base_layout.addWidget(scroll)
        # base_layout.addWidget(boxes_base)
        boxes_layout.addWidget(topbar)
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
        for k in button_dict.keys():
            bbase = QWidget(button_base)
            blayout = QHBoxLayout(bbase)
            for but in button_dict[k]:
                blayout.addWidget(but)
            button_layout.addRow(k,  bbase)
        boxes_layout.addWidget(button_base)
        innerbox = QWidget(button_base)
        innerlayout = QGridLayout(innerbox)
        boxes = []
        for num, el in enumerate(self.params):
            temp = VarBox(boxes_base, el[0],  el[1])
            boxes.append(temp)
            if num <2:
                innerlayout.addWidget(temp.base,  num, 0)
            else:
                innerlayout.addWidget(temp.base,  num-2,  1, 2,  1)
            # temp.values_changed.connect(self.read_inputs)
        boxes_layout.addWidget(innerbox)
        boxes_layout.addWidget(self.progbar)
        # still need the spinbox
        spinbar = QWidget(base)
        spinbar_layout = QHBoxLayout(spinbar)
        spinlabel = QLabel("Currently plotting fit number: ", spinbar)
        spinnumb = QSpinBox(spinbar)
        spinbar_layout.addWidget(spinlabel)
        spinbar_layout.addWidget(spinnumb)
        self.fitnum_spinbox = spinnumb
        # spinbox display finished
        boxes_layout.addWidget(spinbar)
        boxes_layout.addWidget(self.profile_list.table)
        # background fit display
        midbar = QWidget(base)
        midbar_layout = QHBoxLayout(midbar)
        midlabel = QLabel("Polynomial background parameters: ", midbar)
        polyfield = QLineEdit("",  midbar)
        midbar_layout.addWidget(midlabel)
        midbar_layout.addWidget(polyfield)
        self.polyfit_edit = polyfield
        #
        lowbar = QWidget(base)
        lowbar_layout = QHBoxLayout(lowbar)
        lowlabel = QLabel("Step function parameters: ", lowbar)
        edgefield = QLineEdit("",  lowbar)
        lowbar_layout.addWidget(lowlabel)
        lowbar_layout.addWidget(edgefield)
        self.edgefit_edit = edgefield
        # background fit display finished
        boxes_layout.addWidget(midbar)
        boxes_layout.addWidget(lowbar)
        # self.progbar = QProgressBar(base)
        # structure of vars: label, dictionary keys, tooltip
        self.active_buttons = np.zeros(len(button_list)).astype(int)
        self.active_buttons[0:2] = 1
        self.button_base = button_base
        self.boxes_base = boxes_base
        return boxes
    def sequential_fit(self):
        self.block_interface()
        self.start_seq_fit.emit()
        # obj, thread = self.thread_locknload(self.core.sequential_fit)
        # thread.start()
        # self.core.sequential_fit()
    def global_fit(self):
        self.block_interface()
        self.start_glob_fit.emit()
        # obj, thread = self.thread_locknload(self.core.global_fit)
        # thread.start()
        # self.core.global_fit()
    def logger(self, message):
        now = time.gmtime()
        timestamp = ("-".join([str(tx) for tx in [now.tm_mday, now.tm_mon, now.tm_year]])
                     + ',' + ":".join([str(ty) for ty in [now.tm_hour, now.tm_min, now.tm_sec]]) + '| ')
        self.log.setReadOnly(False)
        self.log.append("Fitting :" + timestamp + message)
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
    @pyqtSlot()
    def freeze_inputs(self):
        self.fitnum_spinbox.setReadOnly(True)
    @pyqtSlot()
    def unfreeze_inputs(self):
        self.fitnum_spinbox.setReadOnly(False)
    @pyqtSlot()
    def freeze_outputs(self):
        self.fname_edit.setReadOnly(True)
        self.polyfit_edit.setReadOnly(True)
        self.edgefit_edit.setReadOnly(True)
    @pyqtSlot()
    def unfreeze_outputs(self):
        self.fname_edit.setReadOnly(False)
        self.polyfit_edit.setReadOnly(False)
        self.edgefit_edit.setReadOnly(False)
    def load_profile_button(self):
        result, ftype = QFileDialog.getOpenFileName(self.master, 'Load data from the Andor camera file:', self.currentpath,
               'ADLER 1D curve (*.txt);;All files (*.*)')
        if len(result) > 0:
            newpath, shortname = os.path.split(result)
            self.conf_update.emit({'PATH_fitting': newpath})
            self.currentpath = newpath
            dial = CurveDialog(self, result)
            dial.values_ready.connect(self.core.new_loadparams)
            worked = dial.exec()
            if worked == QDialog.DialogCode.Accepted:
                self.block_interface()
                self.active_buttons[:] = 1
                self.for_loading.emit(result)
            else:
                return None
    @pyqtSlot(object)
    def finish_loading(self,  parlist):
        fullname, shortname, profile = parlist
        self.unfreeze_outputs()
        self.fname_edit.setText(fullname)
        self.freeze_outputs()
        self.plotter.plot1D([profile], fig = self.figure, text = "Choose the fitting parameters.", 
                      label_override = ['X axis',  'Y axis'], curve_labels = [shortname])
        self.flip_buttons()
    def clear_profile_button(self):
        self.block_interface()
        self.active_buttons[:] = 0
        self.active_buttons[0:2] = 1
        self.figure.clear()
        self.figure.canvas.draw_idle()
        self.unfreeze_outputs()
        self.fname_edit.setText("")
        self.polyfit_edit.setText("")
        self.edgefit_edit.setText("")
        self.freeze_outputs()
        self.clear.emit()
    def reload_file(self):
        if self.filelist is not None:
            if len(self.filelist) > 1:
                self.core.preprocess_manyfiles(self.filelist)
                profs = [self.core.summed_rawprofile,  self.core.summed_adjusted_rawprofile]
                self.plotter.plot1D(profs, fig = self.figure, text = "Pick the better profile!", 
                      label_override = ['Channels',  'Counts'], curve_labels = ['Simple Merge',  'Shifted Merge'])
                self.active_buttons[0:3] = 1
                self.active_buttons[3:13] = 0
                self.active_buttons[9:11] = 1
                self.flip_buttons()
            elif len(self.filelist) > 0:
                self.core.process_manyfiles(self.filelist)
                self.plotter.plot2D_sliders(self.core.data2D, self.core.plotax, fig = self.figure)
                self.active_buttons[0:13] = 1
                self.active_buttons[9:11] = 0
                self.flip_buttons()
    def autoprocess_file_button(self):
        profi, back, peak, fitpars, text = self.core.process_file(guess = True)
        if profi is None:
            self.logger('No data available to be processed.')
            return None
        elif back is None or peak is None or text is None:
            self.plotter.plot1D([profi], fig = self.figure, text = "", 
                  label_override = ['Channels',  'Counts'], curve_labels = ['Data'])
        else:
            self.plotter.plot1D([profi,  back,  peak], fig = self.figure, text = text, 
                  label_override = ['Channels',  'Counts'], curve_labels = ['Data',  'Background', 'Fit'])
        self.flip_buttons()
    def process_file_button(self):
        profi, back, peak, fitpars, text = self.core.process_file()
        if profi is None:
            self.logger('No data available to be processed.')
            return None
        elif back is None or peak is None or text is None:
            self.plotter.plot1D([profi], fig = self.figure, text = "", 
                  label_override = ['Channels',  'Counts'], curve_labels = ['Data'])
        else:
            self.plotter.plot1D([profi,  back,  peak], fig = self.figure, text = text, 
                  label_override = ['Channels',  'Counts'], curve_labels = ['Data',  'Background', 'Fit'])
        self.flip_buttons()
    def merge_profiles(self):
        result = self.core.manual_merge()
        if result is not None:
            self.plotter.plot1D([self.core.merged_curve], fig = self.figure, text = "Manually merged profiles", curve_labels = ['Merged'] )
    def save_merged_curve(self):
        result, ftype = QFileDialog.getSaveFileName(self.master, 'Save the merged profile to a text file:', self.currentpath,
               'ADLER 1D output file (*.txt);;All files (*.*)')
        if result is None:
            self.logger("No file name specified; the curve has not been saved.")
        else:
            newpath, shortname = os.path.split(result)
            self.conf_update.emit({'PATH_xasplotting': newpath})
            self.currentpath = newpath
            retcode = self.core.save_merged_profile(result)
            if retcode is not None:
                self.logger("The merged curve has been saved to " + str(result))
    def save_fit_results(self):
        result = QFileDialog.getExistingDirectory(self.master, 'Save the selected profiles to text files:',
                                                                              self.currentpath)
        if result is None:
            self.logger("No file name specified; the curve has not been saved.")
        else:
            result = result + '/'
            newpath, shortname = os.path.split(result)
            self.conf_update.emit({'PATH_xasplotting': newpath})
            self.currentpath = newpath
            retcode = self.core.save_ticked_profiles(newpath)
            if retcode is not None:
                self.logger("The curves have been saved to " + str(result))
    def rixs_map(self):
        obj, thread = self.thread_locknload(self.core.rixsmap)
        thread.finished.connect(self.showrixs)
        thread.start()
    @pyqtSlot()
    def showrixs(self):
        if self.core.rixs_worked:
            self.plotter.plot2D_sliders(self.core.map2D[0], self.core.map2Dplotax, fig = self.figure)
        else:
            self.logger('The RIXS map has NOT been prepared.')
    def overplot(self):
        obj, thread = self.thread_locknload(self.core.multiplot)
        thread.finished.connect(self.showmulti)
        thread.start()
    @pyqtSlot()
    def showmulti(self):
        if self.core.overplotworked:
            curves = self.core.mplot_curves
            rawcurves = self.core.mplot_raw_curves
            labels = self.core.mplot_labels
            text = ""
            plotlabs = self.core.mplot_override
            self.plotter.plot1D_sliders(curves, rawdata = rawcurves, 
                  fig = self.figure, text = text, legend_pos = self.core.legpos, 
                  label_override = plotlabs, curve_labels = labels, max_offset = self.core.offmax)
    @pyqtSlot()
    def showmerged(self):
        if self.core.overplotworked:
            curves = self.core.mplot_curves
            rawcurves = self.core.mplot_raw_curves
            labels = self.core.mplot_labels
            text = ""
            plotlabs = self.core.mplot_override
            self.plotter.plot1D_merged(curves, rawdata = rawcurves, 
                  fig = self.figure, text = text, legend_pos = self.core.legpos, 
                  label_override = plotlabs, curve_labels = labels, max_offset = self.core.offmax)
    def autofit(self):
        obj, thread = self.thread_locknload(self.core.autofit_many)
        thread.finished.connect(self.showfits)
        thread.start()
    def multifit(self):
        obj, thread = self.thread_locknload(self.core.fit_many)
        thread.finished.connect(self.showfits)
        thread.start()
    @pyqtSlot()
    def plot_specific(self):
        if self.core.fitsworked:
            self.profile_list.clear_table()
            num = self.fitnum_spinbox.value()
            if num in self.core.fit_results.keys():
                curve,  pars = self.core.fit_results[num]
                curves = [self.core.profile, curve]
                labels = ['data', 'fit']
                text = "Fitting with " + str(num) + " peaks"
                plotlabs = self.core.mplot_override
                peaks = self.core.mplot_fits
                # fitparams = self.core.mplot_fitparams
                # self.profile_list.assign_fitparams(fitparams)
                self.unfreeze_outputs()
                for pn in range(num):
                    self.profile_list.add_row(pars[pn : 4*num : num])
                self.polyfit_edit.setText(", ".join([str(x) for x in pars[4*num:]]))
                self.freeze_outputs()
                self.plotter.plot1D_withfits(curves, peaks,  fig = self.figure, text = text, legend_pos = self.core.legpos, 
                  label_override = plotlabs, curve_labels = labels, max_offset = self.core.offmax)
    @pyqtSlot()
    def showfits(self):
        if self.core.fitsworked:
            curves = self.core.mplot_curves
            labels = self.core.mplot_labels
            text = ""
            plotlabs = self.core.mplot_override
            peaks = self.core.mplot_fits
            fitparams = self.core.mplot_fitparams
            self.profile_list.assign_fitparams(fitparams)
            self.plotter.plot1D_withfits(curves, peaks,  fig = self.figure, text = text, legend_pos = self.core.legpos, 
                  label_override = plotlabs, curve_labels = labels, max_offset = self.core.offmax)
    @pyqtSlot()
    def summary(self):
        if self.core.fitsworked:
            results = self.core.fit_results
            fsummary = self.core.fit_summary
            numfits = len(results)
            self.fitnum_spinbox.setValue(0)
            self.fitnum_spinbox.setMaximum(numfits)
            xs, vals = [], []
            for kk in fsummary.keys():
                vals.append(fsummary[kk])
                xs.append(int(kk))
            pics = []
            vals = np.array(vals)
            for nn in range(vals.shape[1]):
                pics.append(np.column_stack([xs, vals[:, nn]]))
            self.plotter.plot1D_grid(pics, outFile = "", fig = self.figure, text = '', 
            label_override = [["Peak number", "Cost function"], ["Peak number", "R$^{2}$"], ["Peak number", "$\chi^{2}$"]], 
                    legend_pos = 0)
        
