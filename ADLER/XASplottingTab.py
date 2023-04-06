
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
This file contains the ADLER tab for plotting and comparing
the XAS spectra measured on PEAXIS via CHaOS.
"""

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot,  QSize,  QThread
from PyQt6.QtCore import Qt
# from PyQt6.QtCore import Qt.ItemIsEnabled as ItemIsEnabled
# from PyQt6.QtCore import Qt.Checked as Qt.Checked
# from PyQt6.QtCore import Qt.UnQt.Checked as UnChecked
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QSizePolicy, QWidget, QFileDialog, QTableWidget, QFormLayout,   \
                                                QPushButton,  QVBoxLayout, QHBoxLayout, QTableWidgetItem, QScrollArea, \
                                                QComboBox
# from PyQt6 import sip
from VariablesGUI import VarBox
from ADLERcalc import XasCore, unit_to_int, int_to_unit
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
    mainpos = axes.get_position()
    mainpos.y0 = 0.25       # for example 0.2, choose your value
    # mainpos.ymax = 1.0
    mainpos.y1 = 0.99
    axes.set_position(mainpos)
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
    mainpos = axes.get_position()
    mainpos.y0 = 0.25       # for example 0.2, choose your value
    # mainpos.ymax = 1.0
    mainpos.y1 = 0.99
    axes.set_position(mainpos)
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
                            curve_labels= [],  max_offset= 0.2,  legend_pos = 0, 
                            rawdata = []):
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
    mainpos = axes.get_position()
    mainpos.y0 = 0.25       # for example 0.2, choose your value
    # mainpos.ymax = 1.0
    mainpos.y1 = 0.95
    axes.set_position(mainpos)
    axlabels = ['Pixels (vertical)', labels[0]]
    # for n, l in enumerate(curve_labels):
    #     temp = l.split('eV')
    #     try:
    #         tempval = float(temp[0])
    #     except:
    #         curve_labels[n] = '-1.0 eV' + temp[1]
    refs = []
    refs2 = []
    refs3 = []
    refs4 = []
    refs5 = []
    maxval = 0.0
    minval = 1e9
    lenref3 = 0
    # topval = np.nan_to_num(pic).max()
    # if topval == 0.0:
    #     topval = 1.0
    # xxx = axes.plot(pic[:,0], pic[:,1], '-')
    for rn in range(len(pic)):
        p = pic[rn]
        l = curve_labels[rn]
        [ref] = axes.plot(p[:,0], p[:,1], '-', label = l)
        tempcolour = ref._color
        [ref2,  dummy1,  dummy2] = axes.errorbar(p[:,0], p[:,1], yerr = p[:,2],  color=tempcolour)
        # dupua = axes.errorbar(p[:,0], p[:,1], yerr = p[:,2])
        if rn <= len(rawdata)-1:
            pr = rawdata[rn]
            [ref3] = axes.plot(pr[:,0], pr[:,1], '.', color=tempcolour)
            refs3.append(ref3)
            lenref3 += 1
        else:
            pr = None
        refs.append(ref)
        refs2.append(ref2)
        refs4.append(dummy1)
        refs5.append(dummy2)
        if len(p) > 0:
            maxval = max(maxval, np.abs(p[:,1]).max())
            minval = min(minval, p[:,1].min())
            if pr is not None:
                maxval = max(maxval, np.abs(pr[:,1]).max())
                minval = min(minval, pr[:,1].min())
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
        for n, r in enumerate(range(len(pic))):
            ydata = pic[r][:,1] + n*newval
            raw_ydata = rawdata[r][:,1] + n*newval
            refs[n].set_ydata(ydata)
            refs2[n].set_ydata(ydata)
            if n < lenref3:
                refs3[n].set_ydata(raw_ydata)
            refs5[n][0].set_segments([np.array([[x, yt], [x, yb]])
                                for x, yt, yb in zip(pic[r][:, 0], ydata + pic[r][:, 2], ydata - pic[r][:, 2])])
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

def plot1D_merged(pic, outFile = "", fig = None, text = '', label_override = ["", ""],
                            curve_labels= [],  max_offset= 0.2,  legend_pos = 0, 
                            rawdata = []):
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
    mainpos = axes.get_position()
    mainpos.y0 = 0.25       # for example 0.2, choose your value
    # mainpos.ymax = 1.0
    mainpos.y1 = 0.95
    axes.set_position(mainpos)
    axlabels = ['Pixels (vertical)', labels[0]]
    # for n, l in enumerate(curve_labels):
    #     temp = l.split('eV')
    #     try:
    #         tempval = float(temp[0])
    #     except:
    #         curve_labels[n] = '-1.0 eV' + temp[1]
    maxval = 0.0
    minval = 1e9
    # topval = np.nan_to_num(pic).max()
    # if topval == 0.0:
    #     topval = 1.0
    # xxx = axes.plot(pic[:,0], pic[:,1], '-')
    for rn in range(len(pic)):
        p = pic[rn]
        l = curve_labels[rn]
        [ref] = axes.plot(p[:,0], p[:,1], 'k-', label = 'Merged data')
        tempcolour = ref._color
        [ref2,  dummy1,  dummy2] = axes.errorbar(p[:,0], p[:,1], yerr = p[:,2],  color=tempcolour)
        maxval = max(maxval, np.abs(p[:,1]).max())
        minval = min(minval, p[:,1].min())
        # dupua = axes.errorbar(p[:,0], p[:,1], yerr = p[:,2])
    for rn in range(len(rawdata)):
        if len(rawdata) > 0:
            pr = rawdata[rn]
            [ref3] = axes.plot(pr[:,0], pr[:,1], '.', label = curve_labels[rn+1])
        maxval = max(maxval, np.abs(pr[:,1]).max())
        minval = min(minval, pr[:,1].min())
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
        axes.legend(loc=legend_pos)
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
    mainpos.y1 = 0.95
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

def plot2D_sliders(pics, ax, outFile = "", fig = None, text = '', interp = 'none'): # interp = 'Bessel'):
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
    mainpos = axes.get_position()
    mainpos.y0 = 0.25       # for example 0.2, choose your value
    # mainpos.ymax = 1.0
    mainpos.y1 = 0.99
    axes.set_position(mainpos)
    if len(pics) > 1:
        axlabels = ['Photon energy [eV]', 'Energy transfer [eV]']
        comap = 'rainbow'
    else:
        axlabels = ['Pixels (horizontal)', 'Pixels (vertical)']
        comap = 'OrRd'
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
{'Name': 'Curve cutoff',  'Unit':'eV',  'Value':np.array([0.0, 1200.0]),  'Key' : 'cuts', 
                               'MinValue':-100.0*np.array([1, 1]),
                               'MaxValue':np.array([12000.0, 12000.0]),  'Length': 2,  'Type':'float',
                               'Comment':'Each curve plotted in this tab will be truncated to the x limits specified here.'}, 
{'Name': 'Reduction factor',  'Unit':'N/A',  'Value':1.0, 'Key' : 'redfac', 
                                      'MinValue':0.25,
                                      'MaxValue':10.0,
                                      'Length': 1,  'Type':'float',
                               'Comment':'Higher values of reduction factor lead to coarser binning of the data,\n which can be useful for plotting noisy data sets which do not require high resolution.'}, 
# {'Name': 'Bin size',  'Unit':'N/A',  'Value':-1.0, 'Key' : 'binsize', 
#                                       'MinValue':-10.0,
#                                       'MaxValue':10.0,
#                                       'Length': 1,  'Type':'float',
#                                'Comment':'This re-binnig option is not being used at the moment'}, 
]
line_variables = [
{'Name': 'Detection limit for BKG',  'Unit':'percentile',  'Value':75,  'Key' : 'bkg_perc', 
                               'MinValue':0.0,  'MaxValue':100.0,  'Length': 1,  'Type':'float',
                               'Comment':''},  
{'Name': 'Filter cutoff',  'Unit':'points',  'Value':10,  'Key' : 'cutoff', 
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
# {'Name': 'RixsMap Smearing',  'Unit':'meV',  'Value':2.0,   'Key' : 'smear', 
#                                'MinValue':0.0,  'MaxValue':50.0,  'Length': 1,  'Type':'float',
#                                'Comment':''},  
]

tabnames = ['Filename', 'Xlimits', 'Use TEY?', 'Use TPY?']
class ProfileList(QObject):
    gotvals = pyqtSignal()
    needanupdate = pyqtSignal()
    def __init__(self, master,  nrows =1,  ncolumns = len(tabnames)):
        super().__init__(master)
        self.master = master
        self.Ei = []
        self.names = []
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
        self.needanupdate.connect(self.redraw_table)
    def add_row(self, data):
        self.busy = True
        self.table.blockSignals(True)
        rowitems = []
        lastnit = len(data)
        for nit, dud in enumerate(data):
            d = str(dud)
            rowitems.append(QTableWidgetItem(str(d).strip("()[]'")))
            if nit == 0:
                self.names.append(d)
            elif nit == 1:
                try:
                    vals = [float(tok.strip("()[]'")) for tok in d.split(',')]
                except:
                    vals = [-1.0, -1.0]
                self.Xmin.append(vals[0])
                self.Xmax.append(vals[1])
        chkBoxItem = QTableWidgetItem()
        chkBoxItem.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        chkBoxItem.setCheckState(Qt.CheckState.Checked)
        rowitems.append(chkBoxItem)
        self.useit.append(True)
        chkBoxItem2 = QTableWidgetItem()
        chkBoxItem2.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        chkBoxItem2.setCheckState(Qt.CheckState.Checked)
        rowitems.append(chkBoxItem2)
        self.useit2.append(True)
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
        self.names = []
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
                rowdata += [self.Xmin[nr],  self.Xmax[nr], self.useit[nr], self.useit2[nr], self.names[nr]]
                final.append(rowdata)
        return final
    @pyqtSlot(int, int)
    def update_values(self,  row =0,  column=0):
        if self.busy:
            return None
        self.busy = True
        for nr in range(1,  self.table.rowCount()):
            for nc in range(0, 4):
                if nc == 0:
                    try:
                        name = self.table.item(nr, nc).text()
                    except:
                        name = "The Nameless One"
                    self.names[nr-1] = name
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
            for nc in range(0, 3):
                if nc == 0:
                    temp = self.names[nr-1]
                    self.table.item(nr, nc).setText(temp)
                elif nc == 1:
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

class QHLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

class XASplottingTab(AdlerTab):
    for_loading = pyqtSignal(object)
    clear = pyqtSignal()
    def __init__(self, master,  canvas,  log,  mthreads = 1, startpath = None,  app = None):
        super().__init__(master)
        self.master = master
        # self.progbar = None
        self.log = log
        self.canvas, self.figure, self.clayout = canvas
        self.params = [(loading_variables, "File Loading"),  (line_variables,  "Data correction"),
                               (plotting_variables,  "Plotting")]
        self.profile_list = ProfileList(self.base)
        self.core = XasCore(None,  None,  self.log,  max_threads = mthreads, 
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
        self.core.finished_fitting.connect(self.showfits)
        self.core.finished_overplot.connect(self.showmulti)
        self.core.finished_rixsmap.connect(self.showrixs)
        self.core.finished_merge.connect(self.add_mergedcurve)
        self.core.finished_filter.connect(self.add_filteredcurves)
        self.core.finished_flux.connect(self.add_fluxcurves)
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
        self.combo.currentIndexChanged.connect(self.core.setInterpolation)
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
        ['Load Profiles', self.load_profile_button, 'Pick the 1D profiles to be loaded.', 
            '', 'Files'], # 0
        ['Clear List', self.clear_profile_button, 'Remove all profiles from the list.', 
            '', 'Files'], # 1
        ['Merge datasets', self.core.many_as_one, 'Plot the raw data from separate files.', 
            col1, 'Single Scan'], # 2
        # ['Merge', self.merge_profiles, 'Combine the data from separate files.', 
        #     col1, 'Single Scan'], # 3
        ['Save Merged', self.save_merged_curve, 'Save the merge result to a text file.', 
            '', 'Single Scan'], # 4
        ['Plot absolute', self.core.absoluteplot, 'Compare the absolute values of current.', 
            col1, 'Compare Scans'], # 5
        ['Plot scaled', self.core.multiplot, 'Compare the shapes of the curves.', 
            col1, 'Compare Scans'], # 6
        ['FFT', self.core.fft_curves, 'Calculate the Fourier Transform of the curves', 
            col2, 'Manipulation'], # 8
        ['Filter', self.core.fft_filter, 'Filter the high frequencies from the curve', 
            col2, 'Manipulation'], # 9
        ['Flux correction', self.core.flux_correction, 'Divide the XAS curve by the incoming photon flux.', 
            col1, 'Manipulation'], # 9
        ['Save curves', self.save_ticked_curve, 'Save the selected curves.', 
            '', 'Manipulation'], # 10
        ]
        self.button_list = []
        # base = QWidget(self.master)
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
        boxes_layout.addWidget(self.profile_list.table)
        self.combo = QComboBox(boxes_base)
        templist = []
        for l in self.core.interp_kinds:
            templist.append("Interpolation of missing Flux Correction regions: " +l)
        self.combo.addItems(templist)
        boxes_layout.addWidget(self.combo)
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
    def on_resize(self):
        self.master.resize(self.master.sizeHint())
    def background_launch(self,  core_function,  args =[]):
        self.block_interface()
        # self.core.thread_start(core_function,  args)
        core_function(args)
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
           'ADLER 1D file (*.txt);;ADLER 1D file, server mode (*.asc);;All files (*.*)')
        if result == None:
            self.logger('No valid file chosen, parameters not loaded.')
        else:
            newpath, shortname = os.path.split(result)
            self.conf_update.emit({'PATH_xasplotting': newpath})
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
        self.log.append("XAS plotting :" + timestamp + message)
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
               'PEAXIS XAS file (*.xas);;ADLER 1D curve (*.txt);;All files (*.*)')
        if len(result) > 0:
            newpath, shortname = os.path.split(result[0])
            self.conf_update.emit({'PATH_xasplotting': newpath})
            self.currentpath = newpath
            self.block_interface()
            self.active_buttons[:] = 1
            self.for_loading.emit(result)
    @pyqtSlot(object)
    def finish_loading(self,  parlist):
        snames, tey_profiles, tpy_profiles, raw_tey_profiles, raw_tpy_profiles = parlist
        for n in range(len(snames)):
            if len(tey_profiles[n]) == 0:
                xmin = round(raw_tey_profiles[n][0, 0], 2)
                xmax = round(raw_tey_profiles[n][-1, 0], 2)
            else:
                xmin = round(tey_profiles[n][0, 0], 2)
                xmax = round(tey_profiles[n][-1, 0], 2)
            self.profile_list.add_row([snames[n], [xmin, xmax]])
        self.profile_list.update_values()
        self.profile_list.redraw_table()
        self.flip_buttons()
    @pyqtSlot()
    def add_mergedcurve(self):
        newcurve = self.core.merged_curve.copy()
        xmin, xmax = round(newcurve[0, 0], 2),  round(newcurve[-1, 0], 2)
        self.profile_list.add_row(["Merged data", [xmin, xmax]])
        self.profile_list.update_values()
        self.profile_list.redraw_table()
        self.showmerged()
        self.flip_buttons()
    @pyqtSlot()
    def add_filteredcurves(self):
        for num, curve in enumerate(self.core.filter_curves):
            newcurve = curve[0].copy()
            xmin, xmax = round(newcurve[0, 0], 2),  round(newcurve[-1, 0], 2)
            newcurve = curve[1].copy()
            xmin, xmax = (max(round(newcurve[0, 0], 2),xmin), 
                                  min(round(newcurve[-1, 0], 2), xmax) )
            self.profile_list.add_row([self.core.filter_labels[num], [xmin, xmax]])
        self.profile_list.update_values()
        self.profile_list.redraw_table()
        self.flip_buttons()
    @pyqtSlot()
    def add_fluxcurves(self):
        for num, curve in enumerate(self.core.flux_curves):
            newcurve = curve[0].copy()
            xmin, xmax = round(newcurve[0, 0], 2),  round(newcurve[-1, 0], 2)
            newcurve = curve[1].copy()
            xmin, xmax = (max(round(newcurve[0, 0], 2),xmin), 
                                  min(round(newcurve[-1, 0], 2), xmax) )
            self.profile_list.add_row([self.core.flux_labels[num], [xmin, xmax]])
        self.profile_list.update_values()
        self.profile_list.redraw_table()
        self.flip_buttons()
    def clear_profile_button(self):
        self.block_interface()
        self.active_buttons[:] = 0
        self.active_buttons[0:2] = 1
        self.clear.emit()
    def merge_files_simple(self):
        if len(self.filelist) > 0:
            self.core.justload_manyfiles(self.filelist)
            plot2D_sliders(self.core.data2D, self.core.plotax, fig = self.figure)
            self.active_buttons[0:9] = 1
            self.active_buttons[9:11] = 0
            self.active_buttons[11:14] = 1
            self.flip_buttons()
    def merge_files_offsets(self):
        if len(self.filelist) > 0:
            self.core.finalise_manyfiles()
            plot2D_sliders(self.core.data2D, self.core.plotax, fig = self.figure)
            self.active_buttons[0:9] = 1
            self.active_buttons[9:11] = 0
            self.active_buttons[11:14] = 1
            self.flip_buttons()
    def reload_file(self):
        if self.filelist is not None:
            if len(self.filelist) > 1:
                self.core.preprocess_manyfiles(self.filelist)
                profs = [self.core.summed_rawprofile,  self.core.summed_adjusted_rawprofile]
                plot1D(profs, fig = self.figure, text = "Pick the better profile!", 
                      label_override = ['Channels',  'Counts'], curve_labels = ['Simple Merge',  'Shifted Merge'])
                self.active_buttons[0:3] = 1
                self.active_buttons[3:14] = 0
                self.active_buttons[9:11] = 1
                self.flip_buttons()
            elif len(self.filelist) > 0:
                self.core.process_manyfiles(self.filelist)
                plot2D_sliders(self.core.data2D, self.core.plotax, fig = self.figure)
                self.active_buttons[0:14] = 1
                self.active_buttons[9:11] = 0
                self.flip_buttons()
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
            plot1D([self.core.merged_curve], fig = self.figure, text = "Manually merged profiles", curve_labels = ['Merged'] )
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
    def save_ticked_curve(self):
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
            plot2D_sliders(self.core.map2D[0], self.core.map2Dplotax, fig = self.figure)
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
            plot1D_sliders(curves, rawdata = rawcurves, 
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
            plot1D_merged(curves, rawdata = rawcurves, 
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
    def showfits(self):
        if self.core.fitsworked:
            curves = self.core.mplot_curves
            labels = self.core.mplot_labels
            text = ""
            plotlabs = self.core.mplot_override
            peaks = self.core.mplot_fits
            fitparams = self.core.mplot_fitparams
            self.profile_list.assign_fitparams(fitparams)
            plot1D_withfits(curves, peaks,  fig = self.figure, text = text, legend_pos = self.core.legpos, 
                  label_override = plotlabs, curve_labels = labels, max_offset = self.core.offmax)
        
