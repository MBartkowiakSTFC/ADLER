
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
This tab of the ADLER GUI allows the users to load 1D curves,
and compare them quickly.
"""

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot,  QSize,  QThread, QModelIndex, QSortFilterProxyModel
from PyQt6.QtCore import Qt
# from PyQt6.QtCore import Qt.ItemIsEnabled as ItemIsEnabled
# from PyQt6.QtCore import Qt.Checked as Qt.Checked
# from PyQt6.QtCore import Qt.UnQt.Checked as UnChecked
from PyQt6.QtGui import QFont, QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QFrame, QSizePolicy, QWidget, QFileDialog, QTableView, QFormLayout,   \
                                                QPushButton,  QVBoxLayout, QHBoxLayout, QComboBox, \
                                                QDockWidget, QScrollArea, QSplitter, QStackedWidget, \
                                                QAbstractItemView, QMenu, QApplication
# from PyQt6 import sip
from VariablesGUI import VarBox, CheckGroup
from ADLERcalc import unit_to_int, int_to_unit, RixsMeasurement
from ADLERcalc import NewSimpleCore as SimpleCore
from ExtendGUI import AdlerTab


import numpy as np
import os
import time
import copy
from os.path import expanduser

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
# from matplotlib.backends import qt_compat
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar2QTAgg
from matplotlib.widgets import Slider

# this is a Windows thing
# ctypes.windll.kernel32.SetDllDirectoryW('.')

# simple mathematical functions are defined here

GlobFont = QFont('Sans Serif', int(12*font_scale))

oldval = 0.0

#### plotting part

def plot2D(pic, ax, outFile = "", fig = None, text = ''):
    if fig == None:
        fig = mpl.figure(figsize = [12.0, 8.0], dpi=75, frameon = False)
        trigger = True
    else:
        fig.clear()
        trigger = False
    labels = ['Counts','Counts']
    symbolcount = 0
    handles = []
    ptlabels = []
    print(pic.shape, pic.min(), pic.max())
    axes = fig.add_subplot(111)
    axlabels = ['Pixels (horizontal)', 'Pixels (vertical)']
    topval = np.nan_to_num(pic).max()
    if topval == 0.0:
        topval = 1.0
    xxx = axes.imshow(np.nan_to_num(pic)[::-1,:], extent = [ax[1][0], ax[1][-1],
                                        ax[0][0], ax[0][-1]], interpolation = 'none',
                                        cmap = mpl.get_cmap('OrRd'), aspect = 'auto',
                                        vmin = np.percentile(pic, 20), vmax = np.percentile(pic, 90)
                                        # vmin = 1e-3, vmax = 1.0
                                        )
    cb = mpl.colorbar(xxx, ax = xxx.axes, format = '%.1e', pad = 0.02)
    cb.set_label(labels[0])
    # cb.set_clim(-1, 2.0)
    # xxx.autoscale()
    # axes.contour(np.nan_to_num(pic), # [1e-3*np.nan_to_num(pic).max()], 
    #                            extent = [ax[0][0], ax[0][-1], ax[1][0], ax[1][-1]],
    #                            aspect = 'auto', linestyles = 'solid', linewidths = 1.0)
    axes.grid(True)
    axes.set_xlabel(axlabels[0])
    axes.set_ylabel(axlabels[1])
    box = axes.get_position()
    axes.set_position([box.x0, box.y0 + box.height * 0.2,
             box.width, box.height * 0.8])
    tpos_x = axes.get_xlim()[0]
    ty1, ty2 = axes.get_ylim()
    tpos_y = ty2 + 0.05 * (ty2-ty1)
    axtextf = fig.add_axes([0.20, 0.11, 0.10, 0.01], frameon = False) # , axisbg = '0.9')
    axtextf.set_yticks([])
    axtextf.set_xticks([])
    axtextf.set_title(text)
    if not outFile:
        if trigger:
            mpl.show()
        else:
            fig.canvas.draw()
    else:
        fig.canvas.draw()
        mpl.savefig(outFile, bbox_inches = 'tight')
        mpl.close()

def plot1D(pic, outFile = "", fig = None, text = '', label_override = ["", ""], curve_labels= [], 
                  legend_pos = 0):
    if fig == None:
        fig = mpl.figure(figsize = [12.0, 8.0], dpi=75, frameon = False)
        trigger = True
    else:
        fig.clear()
        trigger = False
    labels = ['Counts','Counts']
    symbolcount = 0
    handles = []
    ptlabels = []
    axes = fig.add_subplot(111)
    axlabels = ['Pixels (vertical)', labels[0]]
    # topval = np.nan_to_num(pic).max()
    # if topval == 0.0:
    #     topval = 1.0
    # xxx = axes.plot(pic[:,0], pic[:,1], '-')
    for xp, p in enumerate(pic):
        if len(p[0]) > 2:
            axes.errorbar(p[:,0], p[:,1], yerr = p[:,2], fmt='-s')
        else:
            if len(curve_labels) == len(pic):
                axes.plot(p[:,0], p[:,1], '-', label = curve_labels[xp])
            else:
                axes.plot(p[:,0], p[:,1], '-')
    axes.grid(True)
    if label_override[0]:
        axes.set_xlabel(label_override[0])
    else:
        axes.set_xlabel(axlabels[0])
    if label_override[1]:
        axes.set_ylabel(label_override[1])
    else:
        axes.set_ylabel(axlabels[1])
    box = axes.get_position()
    axes.set_position([box.x0, box.y0 + box.height * 0.2,
             box.width, box.height * 0.8])
    tpos_x = axes.get_xlim()[0]
    ty1, ty2 = axes.get_ylim()
    if len(p[0]) > 2:
        temp_ylim = np.array([ty1, ty2])
        if ty1 < 0:
            temp_ylim[0] = 0
        if ty2 > 2048:
            temp_ylim[1] = 2048
        axes.set_ylim(temp_ylim)
    tpos_y = ty2 + 0.05 * (ty2-ty1)
    axtextf = fig.add_axes([0.40, 0.01, 0.20, 0.01], frameon = False) # , axisbg = '0.9')
    axtextf.set_yticks([])
    axtextf.set_xticks([])
    axtextf.set_title(text)
    # fig.add_axes(axtextf)
    if len(curve_labels) == len(pic):
        axes.legend(loc=legend_pos)
    if not outFile:
        if trigger:
            mpl.show()
        else:
            fig.canvas.draw()
    else:
        mpl.savefig(outFile, bbox_inches = 'tight')
        mpl.close()

def plot1D_sliders(pic, outFile = "", fig = None, text = '', label_override = ["", ""],
                            curve_labels= [],  max_offset= 0.2,  legend_pos = 0):
    if fig == None:
        fig = mpl.figure(figsize = [12.0, 8.0], dpi=75, frameon = False)
        trigger = True
    else:
        fig.clear()
        trigger = False
    # if len(curve_labels) > 5:
    #     figsize_hor, figsize_vert = fig.get_size_inches()
    #     fig.set_size_inches((figsize_hor, figsize_vert + (len(curve_labels)-5)*0.2))
    figsize_hor, figsize_vert = fig.get_size_inches()
    labels = ['Counts','Counts']
    symbolcount = 0
    handles = []
    ptlabels = []
    axes = fig.add_subplot(111)
    mainpos = axes.get_position()
    mainpos.y0 = 0.25       # for example 0.2, choose your value
    # mainpos.ymax = 1.0
    mainpos.y1 = 0.99
    axes.set_position(mainpos)
    axlabels = ['Pixels (vertical)', labels[0]]
    for n, l in enumerate(curve_labels):
        temp = l.split('eV')
        try:
            tempval = float(temp[0])
        except:
            curve_labels[n] = '-1.0 eV' + temp[1]
    energies = np.array([float(jab.split(' eV')[0]) for jab in curve_labels])
    sequence = np.argsort(energies)
    refs = []
    maxval = 0.0
    minval = 1e9
    # topval = np.nan_to_num(pic).max()
    # if topval == 0.0:
    #     topval = 1.0
    # xxx = axes.plot(pic[:,0], pic[:,1], '-')
    for rn in sequence:
        p = pic[rn]
        l = curve_labels[rn]
        [ref] = axes.plot(p[:,0], p[:,1], '-', label = l)
        refs.append(ref)
        maxval = max(maxval, np.abs(p[10:-10,1]).max())
        minval = min(minval, p[10:-10,1].min())
    span = maxval-minval
    axes.grid(True)
    if label_override[0]:
        axes.set_xlabel(label_override[0])
    else:
        axes.set_xlabel(axlabels[0])
    if label_override[1]:
        axes.set_ylabel(label_override[1])
    else:
        axes.set_ylabel(axlabels[1])
    box = axes.get_position()
    axes.set_position([box.x0, box.y0 + box.height * 0.2,
             box.width, box.height * 0.8])
    axes.set_ylim([minval - 0.1*abs(minval), maxval + 0.1*abs(maxval)])
    axtextf = fig.add_axes([0.40, 0.01, 0.20, 0.01], frameon = False) # , axisbg = '0.9')
    axtextf.set_yticks([])
    axtextf.set_xticks([])
    axtextf.set_title(text)
    if len(curve_labels) == len(pic):
        if legend_pos <0:
            axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                fancybox=True, shadow=True, ncol=1,
                # scatterpoints = 1, numpoints = 1)
                )
        else:
            axes.legend(loc=legend_pos)
    # here we add sliders
    offset_slider_ax  = fig.add_axes([0.25, 0.28, 0.55, 0.03])#, axisbg=axis_color)
    offset_slider = Slider(offset_slider_ax, 'Offset', -max_offset, max_offset, valinit=0.0)
    def sliders_on_changed(val):
        global oldval
        newval = offset_slider.val * span
        for n, r in enumerate(sequence):
            ydata = pic[r][:,1] + n*newval
            refs[n].set_ydata(ydata)
        ty1, ty2 = axes.get_ylim()
        axes.set_ylim([ty1, ty2 + n*(newval-oldval)])
        fig.canvas.draw_idle()
        oldval = copy.deepcopy(newval)
    offset_slider.on_changed(sliders_on_changed)
    if not outFile:
        if trigger:
            mpl.show()
        else:
            fig.canvas.draw()
    else:
        mpl.savefig(outFile, bbox_inches = 'tight')
        mpl.close()

def plot1D_withfits(pic, fit, outFile = "", fig = None, text = '', label_override = ["", ""],
                            curve_labels= [],  max_offset= 0.2,  legend_pos = 0):
    if fig == None:
        fig = mpl.figure(figsize = [12.0, 8.0], dpi=75, frameon = False)
        trigger = True
    else:
        fig.clear()
        trigger = False
    labels = ['Counts','Counts']
    axes = fig.add_subplot(111)
    mainpos = axes.get_position()
    mainpos.y0 = 0.25       # for example 0.2, choose your value
    # mainpos.ymax = 1.0
    mainpos.y1 = 0.99
    axes.set_position(mainpos)
    axlabels = ['Pixels (vertical)', labels[0]]
    for n, l in enumerate(curve_labels):
        temp = l.split('eV')
        try:
            tempval = float(temp[0])
        except:
            curve_labels[n] = '-1.0 eV' + temp[1]
    energies = np.array([float(jab.split(' eV')[0]) for jab in curve_labels])
    sequence = np.argsort(energies)
    refs = []
    peakrefs = []
    maxval = 0.0
    minval = 1e9
    # topval = np.nan_to_num(pic).max()
    # if topval == 0.0:
    #     topval = 1.0
    # xxx = axes.plot(pic[:,0], pic[:,1], '-')
    for rn in sequence:
        p = pic[rn]
        f = fit[rn]
        l = curve_labels[rn]
        [ref] = axes.plot(p[:,0], p[:,1], '-', label = l)
        [peakref] = axes.plot(f[:,0], f[:,1], '--k')
        refs.append(ref)
        peakrefs.append(peakref)
        maxval = max(maxval, np.abs(p[10:-10,1]).max())
        minval = min(minval, np.abs(p[10:-10,1]).min())
    span = maxval-minval
    axes.grid(True)
    if label_override[0]:
        axes.set_xlabel(label_override[0])
    else:
        axes.set_xlabel(axlabels[0])
    if label_override[1]:
        axes.set_ylabel(label_override[1])
    else:
        axes.set_ylabel(axlabels[1])
    box = axes.get_position()
    axes.set_position([box.x0, box.y0 + box.height * 0.2,
             box.width, box.height * 0.8])
    axes.set_ylim([0.9*minval, 1.1*maxval])
    axtextf = fig.add_axes([0.40, 0.01, 0.20, 0.01], frameon = False) # , axisbg = '0.9')
    axtextf.set_yticks([])
    axtextf.set_xticks([])
    axtextf.set_title(text)
    if len(curve_labels) == len(pic):
        if legend_pos <0:
            axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                fancybox=True, shadow=True, ncol=1,
                # scatterpoints = 1, numpoints = 1)
                )
        else:
            axes.legend(loc=legend_pos)
    # here we add sliders
    offset_slider_ax  = fig.add_axes([0.25, 0.28, 0.55, 0.03])#, axisbg=axis_color)
    offset_slider = Slider(offset_slider_ax, 'Offset', -max_offset, max_offset, valinit=0.0)
    def sliders_on_changed(val):
        global oldval
        newval = offset_slider.val * span
        for n, r in enumerate(sequence):
            ydata = pic[r][:,1] + n*newval
            refs[n].set_ydata(ydata)
            ydata2 = fit[r][:,1] + n*newval
            peakrefs[n].set_ydata(ydata2)
        ty1, ty2 = axes.get_ylim()
        axes.set_ylim([ty1, ty2 + n*(newval-oldval)])
        fig.canvas.draw_idle()
        oldval = copy.deepcopy(newval)
    offset_slider.on_changed(sliders_on_changed)
    if not outFile:
        if trigger:
            mpl.show()
        else:
            fig.canvas.draw()
    else:
        mpl.savefig(outFile, bbox_inches = 'tight')
        mpl.close()

def plot2D_sliders(pics, ax, outFile = "", fig = None, text = '', interp = 'none', 
                            axlabels = ['Pixels (horizontal)', 'Pixels (vertical)'],  comap = 'rainbow'): # interp = 'Bessel'):
    if fig == None:
        fig = mpl.figure(figsize = [12.0, 8.0], dpi=75, frameon = False)
        trigger = True
    else:
        fig.clear()
        trigger = False
    labels = ['Counts','Counts']
    symbolcount = 0
    handles = []
    ptlabels = []
    pic = pics
    print(pic.shape, pic.min(), pic.max())
    axes = fig.add_subplot(111)
    # if len(pics) > 1:
    #     axlabels = ['Photon energy [eV]', 'Energy transfer [eV]']
    #     comap = 'rainbow'
    # else:
    #     axlabels = ['Pixels (horizontal)', 'Pixels (vertical)']
    #     comap = 'OrRd'
    topval = np.nan_to_num(pic).max()
    if topval == 0.0:
        topval = 1.0
    abs_maxval = pic.max()
    abs_minval = pic.min()
    curr_maxval, curr_minval = pic.max(), pic.min()
    print(pic.shape, pic.min(), pic.max())
    xxx = axes.imshow(np.nan_to_num(pic)[::-1,:], extent = [ax[1][0], ax[1][-1],
                                        ax[0][0], ax[0][-1]], interpolation = interp,
                                        cmap = mpl.get_cmap(comap), aspect = 'auto',
                                        vmin = np.percentile(pic, 20), vmax = np.percentile(pic, 90)
                                        # vmin = 1e-3, vmax = 1.0
                                        )
    cb = mpl.colorbar(xxx, ax = xxx.axes, format = '%.1e', pad = 0.02)
    cb.set_label(labels[0])
    print(pic.shape, pic.min(), pic.max())
    # cb.set_clim(-1, 2.0)
    # xxx.autoscale()
    # pic2 = pics[1]
    # axes.contour(np.nan_to_num(pic2), [1e-3*np.nan_to_num(pic2).max()], 
    #                            extent = [ax[1][0], ax[1][-1],
    #                                     ax[0][0], ax[0][-1]],
    #                            aspect = 'auto', linestyles = 'solid', linewidths = 1.0)
    axes.grid(True)
    axes.set_xlabel(axlabels[0])
    axes.set_ylabel(axlabels[1])
    box = axes.get_position()
    axes.set_position([box.x0, box.y0 + box.height * 0.2,
             box.width, box.height * 0.8])
    tpos_x = axes.get_xlim()[0]
    ty1, ty2 = axes.get_ylim()
    tpos_y = ty2 + 0.05 * (ty2-ty1)
    # axtextf = mpl.axes([0.20, 0.11, 0.10, 0.01], frameon = False) # , axisbg = '0.9')
    # axtextf.set_yticks([])
    # axtextf.set_xticks([])
    # axtextf.set_title(text)
    # new part: the sliders
    maxval_slider_ax  = fig.add_axes([0.12, 0.12, 0.55, 0.03])#, axisbg=axis_color)
    maxval_slider = Slider(maxval_slider_ax, 'Maxval', 0.0, 100.0, valinit=90.0)
    minval_slider_ax  = fig.add_axes([0.12, 0.04, 0.55, 0.03])#, axisbg=axis_color)
    minval_slider = Slider(minval_slider_ax, 'Minval', 0.0, 100.0, valinit=20.0)
    def sliders_on_changed(val):
        newmax = np.percentile(pic, maxval_slider.val)
        newmin = np.percentile(pic, minval_slider.val)
        if newmax >= newmin:
            xxx.set_clim([newmin, newmax])
            fig.canvas.draw_idle()
    maxval_slider.on_changed(sliders_on_changed)
    minval_slider.on_changed(sliders_on_changed)
    # buttons!
    # reset_button_ax = fig.add_axes([0.55, 0.12, 0.1, 0.04])
    # reset_button = Button(reset_button_ax, 'Reset', hovercolor='0.775')
    # def reset_button_on_clicked(mouse_event):
        # curr_maxval = copy.deepcopy(abs_maxval)
        # curr_minval = copy.deepcopy(abs_minval)
    # reset_button.on_clicked(reset_button_on_clicked)
    # # focus
    # focus_button_ax = fig.add_axes([0.55, 0.04, 0.1, 0.04])
    # focus_button = Button(focus_button_ax, 'Focus', hovercolor='0.775')
    # def focus_button_on_clicked(mouse_event):
        # curr_maxval = curr_maxval * maxval_slider.val
        # curr_minval = curr_minval * minval_slider.val
        # maxval_slider.val = 1.0
        # minval_slider.val = 0.0
    # focus_button.on_clicked(focus_button_on_clicked)
    print(pic.shape, pic.min(), pic.max())
    if not outFile:
        if trigger:
            mpl.show()
        else:
            fig.canvas.draw()
    else:
        fig.canvas.draw()
        mpl.savefig(outFile, bbox_inches = 'tight')
        mpl.close()

#### GUI part

loading_variables = [
{'Name': 'Reduction factor',  'Unit':'N/A',  'Value':1.0, 'Key' : 'redfac', 
                                      'MinValue':0.25,
                                      'MaxValue':10.0,
                                      'Length': 1,  'Type':'float',
                               'Comment':'Higher values of reduction factor lead to coarser binning of the data,\n which can be useful for plotting noisy data sets which do not require high resolution.'}, 
]
line_variables = [
{'Name': 'Elastic line limits',  'Unit':'pixel',  'Value':np.array([-1.0, 1.0]),   'Key' : 'eline', 
                               'MinValue':np.array([-2048, -2048]),  'MaxValue':np.array([2048, 2048]),  'Length': 2,  'Type':'float',
                               'Comment':'The manual fitting of the elastic line will be performed within these x limits.'},   
{'Name': 'Detection limit for BKG',  'Unit':'percentile',  'Value':75,  'Key' : 'bkg_perc', 
                               'MinValue':0.0,  'MaxValue':100.0,  'Length': 1,  'Type':'float',
                               'Comment':'The average of the y values of the curve up to this percentile\n will be used as a fixed, constant background in the elastic line fitting.'},  
{'Name': 'Filter cutoff',  'Unit':'points',  'Value':500,  'Key' : 'cutoff', 
                               'MinValue':0,  'MaxValue':1e7,  'Length': 1,  'Type':'int',
                               'Comment':'If you choose to apply a low-pass filter, \nthe last N values of the Fourier transform will be set to 0.'}, 
]
plotting_variables = [
{'Name': 'Mutliplot offset maximum',  'Unit':'rel. intensity',  'Value':0.2,   'Key' : 'offmax', 
                               'MinValue':0.0,  'MaxValue':10.0,  'Length': 1,  'Type':'float',
                               'Comment':'The highest absolute value of the curve offset which can be set using the built-in slider in the plot.'},  
{'Name': 'Legend position',  'Unit':'N/A',  'Value':-1,   'Key' : 'legpos', 
                               'MinValue':-10,  'MaxValue':10,  'Length': 1,  'Type':'int',
                               'Comment':'The position of the legend in the plot. The non-negative values\n follow the matplotlib definition, and negative values make the legend appear below the plot.'},   
{'Name': 'RixsMap Smearing',  'Unit':'x-axis unit',  'Value':2.0,   'Key' : 'smear', 
                               'MinValue':0.0,  'MaxValue':50.0,  'Length': 1,  'Type':'float',
                               'Comment':'The width of the horizontal smearing of individual spectra in the RIXS map.\n Has to be adjusted based on the spacing between the spectra.\n It is expressed in the units currently set in the combo-box for RIXS map plotting.'},  
]

class BetterTable(QTableView):
    def __init__(self, master, datamodel = None):
        super().__init__(master)
        # self.datamodel = datamodel
    def contextMenuEvent(self, event):
        menu = QMenu(self)
        self.clickedPos = event.pos()
        self.populateMenu(menu)
        menu.exec(event.globalPos())
    def populateMenu(self, menu):
        try:
            temp = self.model().sourceModel()
        except:
            temp = self.model()
        Action = menu.addAction("Copy as Text")
        Action.triggered.connect(temp.textToClipboard)
        Action = menu.addAction("Spreadsheet-friendly copy")
        Action.triggered.connect(temp.excelToClipboard)

tabnames = ['Filename', 'Ei (eV)', 'Temperature (K)', '2 theta (deg)',  'Q (1/A)',  'Use it?', 'FWHM', '+/- dFWHM',  'Int.',  '+/- dInt.',  'Centre',  '+/- dCentre']

class NewPostprocessingTab(AdlerTab):
    for_loading = pyqtSignal(object)
    clear = pyqtSignal()
    def __init__(self, master,  canvas,  log,  mthreads = 1, startpath = None,  app = None):
        super().__init__(master)
        self.master = master
        # self.progbar = None
        self.log = log
        if len(canvas) == 3:
            self.canvas, self.figure, self.clayout = canvas
            self.figure2, self.canvas2 = None, None
        else:
            self.canvas, self.figure, self.clayout, self.figure2, self.canvas2 = canvas
        self.figsize1, self.figsize2 = self.figure.get_size_inches()
        self.params = [(loading_variables, "File Loading"),  (line_variables,  "Elastic Line"),
                               (plotting_variables,  "Plotting")]
        # self.profile_list = ProfileList(self.base)
        self.core = SimpleCore(None,  None,  self.log,  max_threads = mthreads, 
                                        table_headers = tabnames,  startpath = startpath, 
                                        progress_bar = self.progbar)
        normlist = self.core.possible_normalisation_choices()
        self.profile_table = BetterTable(self.master, self.core)
        self.currentpath = startpath
        self.boxes = self.make_layout(normlist)
        self.core.assign_boxes(self.boxes)
        tlist = self.core.possible_rixsmap_axes()
        tlist2 = self.core.possible_rixsmap_axes2()
        xlist = self.core.possible_plot_axes()
        newlist = []
        for old in tlist:
            newlist.append("RIXS map X axis: " + old)
        newlist2 = []
        for old in tlist2:
            newlist2.append("RIXS map Y axis: " + old)
        newlist3 = []
        for old in xlist:
            newlist3.append("1D plot X axis: " + old)
        self.combo.addItems(newlist)
        self.combo2.addItems(newlist2)
        self.combo3.addItems(newlist3)
        self.parnames = []
        self.pardict = {}
        self.filelist = None
        # self.destroyed.connect(self.cleanup)
        self.checkers.new_values.connect(self.core.normalisation_flags)
        # self.profile_list.gotvals.connect(self.core.take_table_values)
        # self.core.cleared.connect(self.profile_list.clear_table)        
        self.profile_table.setSizePolicy(QSizePolicy.Policy.MinimumExpanding,QSizePolicy.Policy.MinimumExpanding)
        # for n in range(self.table.columnCount()):
        #     self.table.setItem(0, n, QTableWidgetItem(tabnames[n]))
        # self.itemChanged.connect(self.update_values)
        topheader = self.profile_table.horizontalHeader()
        # topheader.sectionClicked.connect(self.core.update_ticks)
        # self.table.cellChanged.connect(self.update_ticks)
        self.profile_sorter = QSortFilterProxyModel()
        self.profile_sorter.setSourceModel(self.core)
        self.profile_table.setModel(self.profile_sorter)
        self.profile_table.resizeColumnsToContents()
        self.profile_table.setSortingEnabled(True)
        self.profile_table.setDragEnabled(True)
        self.profile_table.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.profile_table.horizontalHeader().setSectionsMovable(True)
        self.profile_table.horizontalHeader().setDragEnabled(True)
        self.profile_table.horizontalHeader().setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.profile_table.verticalHeader().setSectionsMovable(True)
        self.profile_table.verticalHeader().setDragEnabled(True)
        self.profile_table.verticalHeader().setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.profile_table.show()
        self.core.loaded.connect(self.flip_buttons)
        self.core.loaded.connect(self.profile_table.resizeColumnsToContents)
        self.core.finished_fitting.connect(self.showfits)
        self.core.finished_overplot.connect(self.showmulti)
        self.core.finished_rixsmap.connect(self.showrixs)
        self.core.finished_merge.connect(self.add_mergedcurve)
        self.core.finished_filter.connect(self.add_filteredcurves)
        self.combo.currentIndexChanged.connect(self.core.rixsmap_axis)
        self.combo2.currentIndexChanged.connect(self.core.rixsmap_axis_Y)
        self.combo3.currentIndexChanged.connect(self.core.plot_axis)
        self.flip_buttons()
        #
        self.for_loading.connect(self.core.load_profiles)
        self.clear.connect(self.core.clear_profiles)
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
    def make_layout(self, normlist = []):
        col1 = "background-color:rgb(30,180,20)"
        col2 = "background-color:rgb(0,210,210)"
        col3 = "background-color:rgb(50,150,250)"
        #
        button_list= [
        ['Load Profiles', self.load_profile_button, 'Pick the 1D profiles to be loaded.', 
            col1, 'Files'], # 0
        ['Get Parameters', self.load_params_from_file, 'Read processing parameters from an existing output file.', 
            '', 'Files'], # 1
        ['Clear List', self.clear_profile_button, 'Remove all profiles from the list.', 
            col1, 'Files'], # 2
        ['Save Merged', self.save_merged_curve, 'Save the merge result to a text file.', 
            col1, 'Files'], # 2
        ['Automatic Fit', self.wrapper_autofit_many, 'Try to fit the elastic line automatically.', 
            col1, 'Fitting'], # 3
        ['Fit All', self.wrapper_fit_many, 'Fit the elastic line using the input parameters.', 
            col1, 'Fitting'], # 4
        ['Overplot (scaled)', self.wrapper_multiplot, 'Show the selected profiles in a single plot with an offset.', 
            col2, 'Combining'], # 5
        ['Merge', self.merge_profiles, 'Create a new profile by summing up the selected ones.', 
            col2, 'Combining'], # 6
        ['RIXS Map', self.core.rixsmap, 'Combine the profiles into a 2D RIXS Map.', 
            col3, 'Combining'], # 7
        ['FFT', self.core.fft_curves, 'Calculate the Fourier Transform of the curves', 
            col2, 'Manipulation'], # 7
        ['Filter', self.core.fft_filter, 'Filter the high frequencies from the curve', 
            col2, 'Manipulation'], # 7
        ['Save curves', self.save_ticked_curve, 'Save the selected curves.', 
            col1, 'Manipulation'], # 7
        ]
        self.button_list = []
        # base = QWidget(self.master)
        base = self.base
        base.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        base_layout = QHBoxLayout(base)
        # scar = QScrollArea(base)
        # scar.setWidget(self.canvas)
        # base_layout.addWidget(scar)
        base_layout.addWidget(self.canvas)
        boxes_base = QWidget(base)
        boxes_layout = QVBoxLayout(boxes_base)
        # uberlayout = QStackedWidget(base)
        button_base = QWidget(base)
        # uberlayout.addWidget(button_base)
        # uberlayout.addWidget(self.canvas2)
        # uberlayout.setCurrentIndex(2)
        scroll = QScrollArea(widgetResizable=True)
        scroll.setWidget(boxes_base)
        base_layout.addWidget(scroll)
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
        boxes_layout.addWidget(button_base)
        boxes_layout.addWidget(self.profile_table)
        # self.progbar = QProgressBar(base)
        boxes = []
        self.combo3 = QComboBox(boxes_base)
        boxes_layout.addWidget(self.combo3)
        self.combo = QComboBox(boxes_base)
        boxes_layout.addWidget(self.combo)
        self.combo2 = QComboBox(boxes_base)
        boxes_layout.addWidget(self.combo2)
        temp = []
        for nit in normlist:
            temp.append((nit, 0))
        self.checkers = CheckGroup(self.base,
                              setup_variables= temp, 
                              gname="Divide counts by:", 
                              max_items_per_row = 3)
        boxes_layout.addWidget(self.checkers.base)
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
        # self.splitter.addWidget(self.canvas)
        # self.splitter.addWidget(self.boxes_base)
        return boxes
    @pyqtSlot()
    def wrapper_autofit_many(self):
        # self.profile_list.update_values()
        self.core.autofit_many()
    @pyqtSlot()
    def wrapper_fit_many(self):
        # self.profile_list.update_values()
        self.core.fit_many()
    @pyqtSlot()
    def wrapper_multiplot(self):
        # self.profile_list.update_values()
        self.core.multiplot()
    def save_last_params(self, lastfunction = None):
        try:
            source = open(os.path.join(expanduser("~"),'.ADLERpostprocess.txt'), 'w')
        except:
            return None
        else:
            source.write('Lastdir: '+str(self.core.temp_path) + '\n')
            source.write('Lastfile: '+str(self.temp_name) + '\n')
            for kk in self.input_keys:
                source.write(" ".join([str(u) for u in [kk[0], kk[1], self.params[kk[0]][kk[1]]]]) + '\n')
            if not lastfunction == None:
                source.write('Last function called: ' + str(lastfunction) + '\n')
            source.write('Matplotlib_scale: ' + str(mpl_scale) + '\n')
            source.write('Matplotlib_figure_scale: ' + str(mpl_figure_scale) + '\n')
            source.write('Font_scale: ' + str(font_scale) + '\n')
            source.close()
    def load_last_params(self):
        try:
            source = open(os.path.join(expanduser("~"),'.ADLERpostprocess.txt'), 'r')
        except:
            return None
        else:
            for line in source:
                toks = line.split()
                if len(toks) > 1:
                    if toks[0] == 'Lastdir:':
                        try:
                            self.core.temp_path = toks[1]
                        except:
                            pass
            source.close()
    def load_params_from_file(self):
        result, ftype = QFileDialog.getOpenFileName(self.master, 'Load ADLER parameters from output file header.', self.currentpath,
           'ADLER 1D file (*.txt);;ADLER YAML file (*.yaml);;All files (*.*)')
        if result == None:
            self.logger('No valid file chosen, parameters not loaded.')
        else:
            newpath, shortname = os.path.split(result)
            self.conf_update.emit({'PATH_postprocessing': newpath})
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
                            elif 'BkgPercentile'  in toks[2]:
                                vnames.append('bkg_perc')
                                vdict['bkg_perc'] = [toks[4]]
                            elif 'RedFac'  in toks[2]:
                                vnames.append('redfac')
                                vdict['redfac'] = [toks[4]]
                        elif 'ADLER' in toks[1]:
                            if toks[2] in vnames:
                                vdict[toks[2]] = toks[4:]
                            else:
                                vnames.append(toks[2])
                                vdict[toks[2]] = toks[4:]
                source.close()
                for b in self.boxes:
                    b.takeValues(vnames,  vdict)
    def logger(self, message):
        now = time.gmtime()
        timestamp = ("-".join([str(tx) for tx in [now.tm_mday, now.tm_mon, now.tm_year]])
                     + ',' + ":".join([str(ty) for ty in [now.tm_hour, now.tm_min, now.tm_sec]]) + '| ')
        self.log.setReadOnly(False)
        self.log.append("Postprocessing :" + timestamp + message)
        self.log.setReadOnly(True)
    def MakeCanvas(self, parent):
        mdpi, winch, hinch = 75, 9.0*mpl_figure_scale, 7.0*mpl_figure_scale
        canvas = QWidget(parent)
        layout = QVBoxLayout(canvas)
        figure = mpl.figure(figsize = [winch, hinch], dpi=mdpi )#, frameon = False)
        figAgg = FigureCanvasQTAgg(figure)
        figAgg.setParent(canvas)
        figAgg.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        figAgg.updateGeometry()
        toolbar = NavigationToolbar2QTAgg(figAgg, canvas)
        toolbar.update()
        layout.addWidget(figAgg)
        layout.addWidget(toolbar)
        return canvas, figure, layout
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
    def load_profile_button(self):
        result, ftype = QFileDialog.getOpenFileNames(self.master, 'Load data from the Andor camera file:', self.currentpath,
               'ADLER 1D output file (*.txt);;ADLER YAML file (*.yaml);;All files (*.*)')
        if len(result) > 0:
            newpath, shortname = os.path.split(result[0])
            self.conf_update.emit({'PATH_postprocessing': newpath})
            self.currentpath = newpath
            self.block_interface()
            self.active_buttons[:] = 1
            self.for_loading.emit(result)
    @pyqtSlot()
    def add_mergedcurve(self):
        # newcurve = self.core.merged_curve.copy()
        # xmin, xmax = round(newcurve[0, 0], 2),  round(newcurve[-1, 0], 2)
        # self.profile_list.add_row(["Merged data", self.core.merged_energy,
        #                                 self.core.merged_temperature,  self.core.merged_2theta, self.core.merged_q])
        # self.profile_list.update_values()
        # # self.profile_list.redraw_table()
        self.flip_buttons()
    @pyqtSlot()
    def add_filteredcurves(self):
        for num, curve in enumerate(self.core.filter_curves):
            newcurve = curve.copy()
            xmin, xmax = round(newcurve[0, 0], 2),  round(newcurve[-1, 0], 2)
            # self.profile_list.add_row([self.core.filter_labels[num], self.core.filter_energies[num],
            #                                 self.core.filter_temperatures[num], self.core.filter_2thetas[num], self.core.filter_qs[num]])
        # self.profile_list.update_values()
        # # self.profile_list.redraw_table()
        self.flip_buttons()
    def clear_profile_button(self):
        self.block_interface()
        self.active_buttons[:] = 0
        self.active_buttons[0:2] = 1
        self.clear.emit()
    def autoprocess_file_button(self):
        profi, back, peak, fitpars, text = self.core.process_file(guess = True)
        if profi is None:
            self.logger('No data available to be processed.')
            return None
        elif back is None or peak is None or text is None:
            plot1D([profi], fig = self.figure, text = "", 
                  label_override = ['Channels',  'Counts'], curve_labels = ['Data'])
        else:
            plot1D([profi,  back,  peak], fig = self.figure, text = text, 
                  label_override = ['Channels',  'Counts'], curve_labels = ['Data',  'Background', 'Fit'])
        self.flip_buttons()
    def process_file_button(self):
        profi, back, peak, fitpars, text = self.core.process_file()
        if profi is None:
            self.logger('No data available to be processed.')
            return None
        elif back is None or peak is None or text is None:
            plot1D([profi], fig = self.figure, text = "", 
                  label_override = ['Channels',  'Counts'], curve_labels = ['Data'])
        else:
            plot1D([profi,  back,  peak], fig = self.figure, text = text, 
                  label_override = ['Channels',  'Counts'], curve_labels = ['Data',  'Background', 'Fit'])
        self.flip_buttons()
    def merge_profiles(self):
        result = self.core.manual_merge()
        if result is not None:
            plot1D([self.core.merged_curve], fig = self.figure, text = "Manually merged profiles", 
                  label_override = self.core.merged_units, curve_labels = ['Merged'] )
    def save_merged_curve(self):
        result, ftype = QFileDialog.getSaveFileName(self.master, 'Save the merged profile to a text file:', self.currentpath,
               'ADLER 1D output file (*.txt);;All files (*.*)')
        if len(result) < 1:
            self.logger("No file name specified; the curve has not been saved.")
        else:
            newpath, shortname = os.path.split(result)
            self.conf_update.emit({'PATH_postprocessing': newpath})
            self.currentpath = newpath
            retcode = self.core.save_merged_profile(result)
            if retcode is not None:
                self.logger("The merged curve has been saved to " + str(result))
    def save_ticked_curve(self):
        result = QFileDialog.getExistingDirectory(self.master,
                                    'Save the selected profiles to text files:', self.currentpath)
        if len(result) < 1:
            self.logger("No file name specified; the curve has not been saved.")
        else:
            result = result + '/'
            newpath, shortname = os.path.split(result)
            self.conf_update.emit({'PATH_postprocessing': newpath})
            self.currentpath = newpath
            retcode = self.core.save_ticked_profiles(newpath)
            if retcode is not None:
                self.logger("The curves have been saved to " + str(result))
    def rixs_map(self):
        # self.profile_list.update_values()
        obj, thread = self.thread_locknload(self.core.rixsmap)
        thread.finished.connect(self.showrixs)
        thread.start()
    @pyqtSlot()
    def showrixs(self):
        # self.profile_list.update_values()
        # self.figure.set_size_inches(self.figsize1, self.figsize2)
        if self.core.rixs_worked:
            plot2D_sliders(self.core.map2D[0], self.core.map2Dplotax, fig = self.figure, 
            axlabels = [self.core.rixs_axis_label, 'Energy transfer [eV]'], 
            comap = 'rainbow')
        else:
            self.logger('The RIXS map has NOT been prepared.')
    def overplot(self):
        # self.profile_list.update_values()
        obj, thread = self.thread_locknload(self.core.multiplot)
        thread.finished.connect(self.showmulti)
        thread.start()
    @pyqtSlot()
    def showmulti(self):
        # self.profile_list.update_values()
        # self.figure.set_size_inches(self.figsize1, self.figsize2)
        if self.core.overplotworked:
            curves = self.core.mplot_curves
            labels = self.core.mplot_labels
            text = ""
            plotlabs = self.core.mplot_override
            plot1D_sliders(curves, fig = self.figure, text = text, legend_pos = self.core.legpos, 
                  label_override = plotlabs, curve_labels = labels, max_offset = self.core.offmax)
    def autofit(self):
        # self.profile_list.update_values()
        obj, thread = self.thread_locknload(self.core.autofit_many)
        thread.finished.connect(self.showfits)
        thread.start()
    def multifit(self):
        # self.profile_list.update_values()
        obj, thread = self.thread_locknload(self.core.fit_many)
        thread.finished.connect(self.showfits)
        thread.start()
    @pyqtSlot()
    def showfits(self):
        # self.figure.set_size_inches(self.figsize1, self.figsize2)
        if self.core.fitsworked:
            curves = self.core.mplot_curves
            labels = self.core.mplot_labels
            text = ""
            plotlabs = self.core.mplot_override
            peaks = self.core.mplot_fits
            fitparams = self.core.mplot_fitparams
            # self.profile_list.assign_fitparams(fitparams)
            plot1D_withfits(curves, peaks,  fig = self.figure, text = text, legend_pos = self.core.legpos, 
                  label_override = plotlabs, curve_labels = labels, max_offset = self.core.offmax)
        
