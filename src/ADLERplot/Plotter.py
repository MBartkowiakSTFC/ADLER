
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

import copy

import numpy as np
import matplotlib.pyplot as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar2QTAgg
from matplotlib.widgets import Slider


class Plotter:
    """This class provides matplotlib-based plotting services.
    It will normally be associated with one specific tab of the
    ADLER gui, and a specific canvas. This way the behaviour of
    the plotter can be customised for each tab.
    """

    def __init__(self, *args, figure = None, **kwargs):

        self.figure = figure
        comap = kwargs.get('colourmap', 'OrRd')
        self.colourmap = mpl.get_cmap(comap)
        self.labels1d = kwargs.get('labels1d', ['Counts', 'Counts'])
        self.labels2d = kwargs.get('labels2d', ['Pixels', 'Pixels'])

    def plot1D(self, pic, outFile = "", fig = None, text = '',
                    label_override = ["", ""], curve_labels= [], 
                    legend_pos = 0):
        if self.figure == None:
            self.figure = mpl.figure(figsize = [12.0, 8.0], dpi=75, frameon = False)
            trigger = True
        else:
            self.figure.clear()
            trigger = False
        labels = self.labels1d
        symbolcount = 0
        handles = []
        ptlabels = []
        axes = self.figure.add_subplot(111)
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
        axtextf = self.figure.add_axes([0.40, 0.01, 0.20, 0.01], frameon = False) # , axisbg = '0.9')
        axtextf.set_yticks([])
        axtextf.set_xticks([])
        axtextf.set_title(text)
        # fig.add_axes(axtextf)
        if len(curve_labels) == len(pic):
            if legend_pos <0:
                axes.legend(loc='upper center', bbox_to_anchor=(0.25, -0.14),
                    fancybox=True, shadow=True, ncol=1,
                    # scatterpoints = 1, numpoints = 1)
                    )
            else:
                axes.legend(loc=legend_pos)
        if not outFile:
            if trigger:
                mpl.show()
            else:
                self.figure.canvas.draw()
        else:
            mpl.savefig(outFile, bbox_inches = 'tight')
            mpl.close()

    def plot1D_sliders(self, pic, outFile = "", fig = None, text = '', label_override = ["", ""],
                                curve_labels= [],  max_offset= 0.2,  legend_pos = 0):
        if self.figure == None:
            self.figure = mpl.figure(figsize = [12.0, 8.0], dpi=75, frameon = False)
            trigger = True
        else:
            self.figure.clear()
            trigger = False
        labels = self.labels1d
        symbolcount = 0
        handles = []
        ptlabels = []
        axes = self.figure.add_subplot(111)
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
        for rn in sequence:
            p = pic[rn]
            l = curve_labels[rn]
            [ref] = axes.plot(p[:,0], p[:,1], '-', label = l)
            refs.append(ref)
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
        axtextf = self.figure.add_axes([0.40, 0.01, 0.20, 0.01], frameon = False) # , axisbg = '0.9')
        axtextf.set_yticks([])
        axtextf.set_xticks([])
        axtextf.set_title(text)
        if len(curve_labels) == len(pic):
            axes.legend(loc=legend_pos)
        # here we add sliders
        offset_slider_ax  = self.figure.add_axes([0.25, 0.15, 0.55, 0.03])#, axisbg=axis_color)
        offset_slider = Slider(offset_slider_ax, 'Offset', 0.0, max_offset, valinit=0.0)
        def sliders_on_changed(val):
            global oldval
            newval = offset_slider.val * span
            for n, r in enumerate(sequence):
                ydata = pic[r][:,1] + n*newval
                refs[n].set_ydata(ydata)
            ty1, ty2 = axes.get_ylim()
            axes.set_ylim([ty1, ty2 + n*(newval-oldval)])
            self.figure.canvas.draw_idle()
            oldval = copy.deepcopy(newval)
        offset_slider.on_changed(sliders_on_changed)
        if not outFile:
            if trigger:
                mpl.show()
            else:
                self.figure.canvas.draw()
        else:
            mpl.savefig(outFile, bbox_inches = 'tight')
            mpl.close()

    def plot1D_withfits(self, pic, fit, outFile = "", fig = None, text = '', label_override = ["", ""],
                                curve_labels= [],  max_offset= 0.2,  legend_pos = 0):
        if self.figure == None:
            self.figure = mpl.figure(figsize = [12.0, 8.0], dpi=75, frameon = False)
            trigger = True
        else:
            self.figure.clear()
            trigger = False
        labels = self.labels1d
        axes = self.figure.add_subplot(111)
        mainpos = axes.get_position()
        mainpos.y0 = 0.25
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
        axtextf = self.figure.add_axes([0.40, 0.01, 0.20, 0.01], frameon = False) # , axisbg = '0.9')
        axtextf.set_yticks([])
        axtextf.set_xticks([])
        axtextf.set_title(text)
        if len(curve_labels) == len(pic):
            if legend_pos <0:
                axes.legend(loc='upper center', bbox_to_anchor=(0.25, -0.14),
                    fancybox=True, shadow=True, ncol=1,
                    # scatterpoints = 1, numpoints = 1)
                    )
            else:
                axes.legend(loc=legend_pos)
        # here we add sliders
        offset_slider_ax  = self.figure.add_axes([0.25, 0.15, 0.55, 0.03])#, axisbg=axis_color)
        offset_slider = Slider(offset_slider_ax, 'Offset', 0.0, max_offset, valinit=0.0)
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
            self.figure.canvas.draw_idle()
            oldval = copy.deepcopy(newval)
        offset_slider.on_changed(sliders_on_changed)
        if not outFile:
            if trigger:
                mpl.show()
            else:
                self.figure.canvas.draw()
        else:
            mpl.savefig(outFile, bbox_inches = 'tight')
            mpl.close()

    @classmethod
    def gridtype(nplots):
        if nplots ==1:
            return (111, )
        elif nplots ==2:
            return (211, 212)
        elif nplots ==3:
            return (311,  312,  313)
        elif nplots ==4:
            return (221,  222,  223,  224)
        elif nplots ==5:
            return (321,  322,  323,  324,  325)
        elif nplots ==6:
            return (321,  322,  323,  324,  325,  326)
        elif nplots ==7:
            return (421,  422,  423,  424,  425,  426,  427)
        elif nplots ==8:
            return (421,  422,  423,  424,  425,  426,  427,  428)
        elif nplots > 8:
            return (331, 332, 333, 334, 335, 336, 337, 338, 339)
        else:
            return []

    def plot1D_grid(self, pic, outFile = "", fig = None, text = '', label_override = ["", ""],
                        legend_pos = 0):
        if self.figure == None:
            self.figure = mpl.figure(figsize = [12.0, 8.0], dpi=75, frameon = False)
            trigger = True
        else:
            self.figure.clear()
            trigger = False
        grids = self.gridtype(len(pic))
        for ncur, grid in enumerate(grids):
            axes = self.figure.add_subplot(grid)
            axes.grid(True)
            xlab, ylab = label_override[ncur]
            axes.set_xlabel(xlab)
            axes.set_ylabel(ylab)
            axes.plot(pic[ncur][:, 0],  pic[ncur][:, 1])
        if not outFile:
            if trigger:
                mpl.show()
            else:
                self.figure.canvas.draw()
        else:
            mpl.savefig(outFile, bbox_inches = 'tight')
            mpl.close()

    def plot2D_sliders(self, pics, ax, outFile = "", fig = None, text = '', interp = 'none', 
                                labels = ['','']): # interp = 'Bessel'):
        if self.figure == None:
            self.figure = mpl.figure(figsize = [12.0, 8.0], dpi=75, frameon = False)
            trigger = True
        else:
            self.figure.clear()
            trigger = False
        symbolcount = 0
        handles = []
        ptlabels = []
        pic = pics
        print(pic.shape, pic.min(), pic.max())
        axes = self.figure.add_subplot(111)
        mainpos = axes.get_position()
        mainpos.y0 = 0.25       # for example 0.2, choose your value
        # mainpos.ymax = 1.0
        mainpos.y1 = 0.99
        axes.set_position(mainpos)
        axlabels = self.labels2d
        comap = 'rainbow'
        topval = np.nan_to_num(pic).max()
        if topval == 0.0:
            topval = 1.0
        abs_maxval = pic.max()
        abs_minval = pic.min()
        curr_maxval, curr_minval = pic.max(), pic.min()
        print(pic.shape, pic.min(), pic.max())
        xxx = axes.imshow(np.nan_to_num(pic)[::-1,:], extent = [ax[1][0], ax[1][-1],
                                            ax[0][0], ax[0][-1]], interpolation = interp,
                                            cmap = self.colourmap, aspect = 'auto',
                                            vmin = np.percentile(pic, 20), vmax = np.percentile(pic, 90)
                                            # vmin = 1e-3, vmax = 1.0
                                            )
        cb = mpl.colorbar(xxx, ax = xxx.axes, format = '%.1e', pad = 0.02)
        cb.set_label(text)
        print(pic.shape, pic.min(), pic.max())
        axes.grid(True)
        axes.set_xlabel(axlabels[0])
        axes.set_ylabel(axlabels[1])
        box = axes.get_position()
        axes.set_position([box.x0, box.y0 + box.height * 0.2,
                box.width, box.height * 0.8])
        tpos_x = axes.get_xlim()[0]
        ty1, ty2 = axes.get_ylim()
        tpos_y = ty2 + 0.05 * (ty2-ty1)
        maxval_slider_ax  = self.figure.add_axes([0.12, 0.12, 0.55, 0.03])#, axisbg=axis_color)
        maxval_slider = Slider(maxval_slider_ax, 'Maxval', 0.0, 100.0, valinit=90.0)
        minval_slider_ax  = self.figure.add_axes([0.12, 0.04, 0.55, 0.03])#, axisbg=axis_color)
        minval_slider = Slider(minval_slider_ax, 'Minval', 0.0, 100.0, valinit=20.0)
        def sliders_on_changed(val):
            newmax = np.percentile(pic, maxval_slider.val)
            newmin = np.percentile(pic, minval_slider.val)
            if newmax >= newmin:
                xxx.set_clim([newmin, newmax])
                self.figure.canvas.draw_idle()
        maxval_slider.on_changed(sliders_on_changed)
        minval_slider.on_changed(sliders_on_changed)
        print(pic.shape, pic.min(), pic.max())
        if not outFile:
            if trigger:
                mpl.show()
            else:
                self.figure.canvas.draw()
        else:
            self.figure.canvas.draw()
            mpl.savefig(outFile, bbox_inches = 'tight')
            mpl.close()

    def plot1D_merged(self, pic, outFile = "", fig = None, text = '', label_override = ["", ""],
                                curve_labels= [],  max_offset= 0.2,  legend_pos = 0, 
                                rawdata = []):
        if self.figure == None:
            self.figure = mpl.figure(figsize = [12.0, 8.0], dpi=75, frameon = False)
            trigger = True
        else:
            self.figure.clear()
            trigger = False
        labels = ['Counts','Counts']
        symbolcount = 0
        handles = []
        ptlabels = []
        axes = self.figure.add_subplot(111)
        mainpos = axes.get_position()
        mainpos.y0 = 0.25       # for example 0.2, choose your value
        # mainpos.ymax = 1.0
        mainpos.y1 = 0.99
        axes.set_position(mainpos)
        axlabels = ['Pixels (vertical)', labels[0]]
        maxval = 0.0
        minval = 1e9
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
        axtextf = self.figure.add_axes([0.40, 0.01, 0.20, 0.01], frameon = False) # , axisbg = '0.9')
        axtextf.set_yticks([])
        axtextf.set_xticks([])
        axtextf.set_title(text)
        if len(curve_labels) == len(pic):
            axes.legend(loc=legend_pos)
        if not outFile:
            if trigger:
                mpl.show()
            else:
                self.figure.canvas.draw()
        else:
            mpl.savefig(outFile, bbox_inches = 'tight')
            mpl.close()
