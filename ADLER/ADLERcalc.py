
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
The part of the ADLER code responsible for
the handling of the files and processing the data.
"""

import math
import numpy as np
import os
import time
import sys
import gzip
import h5py
from os.path import expanduser
import copy
from collections import defaultdict
from numba import jit, prange
from scipy.sparse import csc_array

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

has_voigt = True
try:
    from scipy.special import voigt_profile
except:
    from scipy.special import wofz
    has_voigt = False
from scipy.optimize import leastsq, shgo, minimize
from scipy.interpolate import interp1d
from scipy.fftpack import rfft, irfft, fftfreq
from astropy.io import fits as fits_module

# import ctypes

from PyQt6.QtCore import QThread, QObject, pyqtSignal, pyqtSlot, QMutex, QDate, QTime
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import  QApplication
from ExtendGUI import CustomThreadpool
from DataHandling import DataEntry, DataGroup, RixsMeasurement
# this is a Windows thing
# ctypes.windll.kernel32.SetDllDirectoryW('.')

# simple mathematical functions are defined here
try:
    rand_mt = np.random.Generator(np.random.MT19937(np.random.SeedSequence(31337)))
except:
    rand_mt = np.random.Generator(np.random.Philox4x3210(31337))

gauss_denum = 2.0 * (2.0 * math.log(2.0))**0.5
gauss_const = 1/((2.0*math.pi)**0.5)

oldval = 0.0

# from numba import jit, prange

# has_voigt = False
# from scipy.special import wofz

if has_voigt:
    def my_voigt(xarr, A, xc, wG, wL):
        x_span = xarr.max()-xarr.min()
        temp_x = np.linspace(-2*x_span, 2*x_span, 8*len(xarr))
        temp_y = voigt_profile(temp_x, wG, wL) *abs(A)
        temp_func = interp1d(temp_x + xc, temp_y, bounds_error = False, fill_value = 0.0)
        return temp_func(xarr)
    def fast_voigt(xarr, A, xc, wG, wL):
        temp_x = xarr - xc
        temp_y = voigt_profile(temp_x, wG, wL)
        temp_y *= abs(A) / temp_y.max()
        return temp_y
    def asym_voigt(xarr, A, xc, wG, wL, R):
        temp_x = xarr - xc
        temp_y = np.zeros(temp_x.shape)
        widths_left = np.array([wG, wL])
        widths_orig = widths_left.copy()
        widths_right = np.array([wG*2**R, wL*2**R])
        widths_scale = 2*widths_orig/(widths_left+widths_right)
        widths_left *= widths_scale
        widths_right *= widths_scale
        # print("wG", wG)
        # print("wL", wL)
        # print("widths left: ",  widths_left)
        # print("widths right: ",  widths_right)
        left = np.where(temp_x <= 0)
        right = np.where(temp_x > 0)
        templeft = np.ones(1)
        if len(left[0]) >0:
            templeft = voigt_profile(temp_x[left], widths_left[0], widths_left[1])
            temp_y[left] += templeft
        if len(right[0]) >0:
            tempright = voigt_profile(temp_x[right], widths_right[0], widths_right[1])
            tempright *= templeft.max()/ tempright.max()
            temp_y[right] += tempright
        temp_y *= abs(A) / temp_y.max()
        return temp_y
else:
    def my_voigt(xarr, A_in, xc_in, wG_in, wL_in):
        A = np.abs(A_in)
        xc = xc_in
        wG = np.abs(wG_in)
        wL = np.abs(wL_in)
        x_span = xarr.max()-xarr.min()
        temp_x = np.linspace(-2*x_span, 2*x_span, 8*len(xarr))
        # temp_y = voigt_profile(temp_x, wG, wL) *A
        temp_y = A * np.real(wofz((temp_x + 1j*wL)/(wG*np.sqrt(2))) ) / (wG * np.sqrt(2*np.pi))
        temp_func = interp1d(temp_x + xc, temp_y, bounds_error = False, fill_value = 0.0)
        return temp_func(xarr)
    def fast_voigt(xarr, A_in, xc_in, wG_in, wL_in):
        A = np.abs(A_in)
        xc = xc_in
        wG = np.abs(wG_in)
        wL = np.abs(wL_in)
        temp_x = xarr-xc
        # temp_y = voigt_profile(temp_x, wG, wL) *A
        temp_y = np.real(wofz((temp_x + 1j*wL)/(wG*np.sqrt(2))) ) / (wG * np.sqrt(2*np.pi))
        temp_y *= A / temp_y.max()
        return temp_y
    def asym_voigt(xarr, A_in, xc_in, wG_in, wL_in, R):
        A = np.abs(A_in)
        xc = xc_in
        wG = np.abs(wG_in)
        wL = np.abs(wL_in)
        temp_x = xarr - xc
        temp_y = np.zeros(temp_x.shape)
        widths_left = np.array([wG, wL])
        widths_orig = widths_left.copy()
        widths_right = np.array([wG*2**R, wL*2**R])
        widths_scale = 2*widths_orig/(widths_left+widths_right)
        widths_left *= widths_scale
        widths_right *= widths_scale
        # print("wG", wG)
        # print("wL", wL)
        # print("widths left: ",  widths_left)
        # print("widths right: ",  widths_right)
        left = np.where(temp_x <= 0)
        right = np.where(temp_x > 0)
        templeft = np.ones(1)
        if len(left[0]) >0:
            templeft = np.real(wofz((temp_x[left] + 1j*widths_left[1])/(widths_left[0]*np.sqrt(2))) ) / (widths_left[0] * np.sqrt(2*np.pi))
            temp_y[left] += templeft
        if len(right[0]) >0:
            tempright = np.real(wofz((temp_x[right] + 1j*widths_right[1])/(widths_right[0]*np.sqrt(2))) ) / (widths_right[0] * np.sqrt(2*np.pi))
            tempright *= templeft.max()/ tempright.max()
            temp_y[right] += tempright
        temp_y *= abs(A) / temp_y.max()
        return temp_y

def loadtext_wrapper(fname):
    result = []
    try:
        source = open(fname, 'rb')
    except:
        return None
    else:
        for n, line in enumerate(source):
            try:
                textline = line.decode('utf-8')
            except:
                try:
                    textline = line.decode('cp1252')
                except:
                    continue
                else:
                    result.append(textline.strip('\n'))
            else:
                result.append(textline.strip('\n'))
        source.close()
        return result

def compress_array(inp_array):
    data = inp_array.copy()
    shape = data.shape
    dtype = data.dtype.str
    bdata = data.tobytes()
    compdata = gzip.compress(bdata)
    return compdata, shape, dtype

def decompress_array(compdata, shape, dtype):
    bdata = gzip.decompress(compdata)
    linedata = np.frombuffer(bdata, dtype=dtype)
    truedata = linedata.reshape(shape)
    return truedata

def store_array(fname, compdata, shape, dtype, maxdim=4, dimlen=5, dtypelen=4):
    ndim = len(shape)
    header = (maxdim*dimlen+dtypelen)*'0'
    for n in np.arange(ndim):
        tnum = shape[n]
        text = str(tnum).zfill(5)
        header[n*dimlen:(n+1)*dimlen] = text
    header[maxdim*dimlen:maxdim*dimlen+dtypelen] = dtype.zfill(dtypelen)
    dump = open(fname, 'wb')
    dump.write(header.encode('ascii'))
    dump.write(compdata)
    dump.close()

def restore_array(fname, maxdim=4, dimlen=5, dtypelen=4,  buffsize = 8192):
    source = open(fname, 'rb')
    header = source.read(maxdim*dimlen+dtypelen)
    compdata = b''
    buffer = b'999'
    while buffer:
        buffer = source.read(buffsize)
        compdata += buffer
    source.close()
    shape = []
    for x in np.arange(maxdim):
        val = int(header[x*dimlen:(x+1)*dimlen].decode('ascii'))
        if val>0:
            shape.append(val)
    dtype= header[maxdim*dimlen:maxdim*dimlen+dtypelen].decode('ascii').strip('0')
    the_array = decompress_array(compdata, shape, dtype)
    return the_array


# print(sys._MEIPASS)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class Worker(QObject):
    finished = pyqtSignal()
    intReady = pyqtSignal(int)
    runPeriodically = None
    keeprunnig = True
    waittime = 1
    def set_end(self):
        self.keeprunning = False
    def set_waittime(self, wtime):
        self.waittime = wtime
    def assignFunction(self, some_function):
        self.runPeriodically = some_function
    def assignArgs(self, args):
        self.func_args = args
    @pyqtSlot()
    def updater(self): # A slot takes no params
        while self.keeprunnig:
            time.sleep(self.waittime)
            if self.runPeriodically is not None:
                self.runPeriodically()
        self.finished.emit()
    @pyqtSlot()
    def run_once(self): # A slot takes no params
        try:
            nargs = len(self.func_args)
        except:
            nargs = 0
        if self.runPeriodically is not None:
            if nargs == 0:
                self.runPeriodically()
            elif nargs == 1:
                self.runPeriodically(self.func_args[0])
            elif nargs == 2:
                self.runPeriodically(self.func_args[0],  self.func_args[1])
        self.finished.emit()
    @pyqtSlot()
    def run_repeated(self): # A slot takes no params
        try:
            nargs = len(self.func_args)
        except:
            nargs = 0
        if self.runPeriodically is not None:
            for n in range(nargs):
                self.runPeriodically(self.func_args[n])
                self.intReady.emit(n)
        self.finished.emit()

def gaussian(a, FWHM, centre, array):
    """
    gaussian(a, FWHM, centre, arguments_array)
    Takes an array of x values and returns a
    gaussian peak with heighth a with FWHM and centre
    defined by input parameters.
    """
    c = FWHM / gauss_denum
    return  a * np.exp( - (array - centre)**2 / (2 * c**2))

def polynomial(args, inarray):
    """
    An n-th order polynomial.
    """
    xs = np.array(inarray)
    ys = np.zeros(xs.shape)
    for n, a in enumerate(args):
        ys += a*xs**n
    return ys

def fit_polynomial(args, array):
    newfit = polynomial(args, array[:,0])
    if array.shape[-1] > 2 :
        return (newfit - array[:,1])/(array[:,2]+np.abs(0.00001*array[:,1].mean()))
    else:
        return (newfit - array[:,1])

def fit_gaussian(args, array):
    a, FWHM, centre, zeroline = args[0], args[1], args[2], args[3]
    newvals = gaussian(a, FWHM, centre, array[:,0]) + zeroline
    return newvals - array[:,1]

def fit_gaussian_fixedbase(args, array, base):
    a, FWHM, centre = args[0], args[1], args[2]
    newvals = gaussian(a, FWHM, centre, array[:,0]) + base
    return newvals - array[:,1]

def elastic_line(profile, background, fixwidth = None, olimits = (None, None)):
    """
    As an initial guess, this function will assume that the 
    elastic line is at the point with the highest intensity.
    Otherwise, an override for the peak centre can be provided.
    """
    chi2 = -1.0
    if len(background) > 1:
        zline, zerr = background[:,1].mean(), background[:,1].std()
    else:
        zline = profile[:, 1].min()
        zerr = abs(profile[:, 1].max() - profile[:, 1].min())**0.5
    temp = profile.copy()
    temp[:,1] -= zline
    maxval = temp[:,1].max()
    nlimits = [0, -1]
    if olimits[0] is not None and olimits[1] is not None:
        boolarr1 = temp[:, 0] > olimits[0]
        if boolarr1.any() and not boolarr1.all():
            nlimits[0] = max(np.argmax(boolarr1) -1, 0)
        else:
            nlimits[0] = 0
        boolarr2 = temp[:, 0] > olimits[1]
        if boolarr2.any() and not boolarr2.all():
            nlimits[1] = min(np.argmax(boolarr2), len(temp)-1)
        else:
            nlimits[1] = len(temp)-1
        if abs(nlimits[1]-nlimits[0]) < 5:
            nlimits[1] += 1
        if abs(nlimits[1]-nlimits[0]) < 5:
            nlimits[0] -= 1
        if abs(nlimits[1]-nlimits[0]) < 5:
            nlimits[1] += 1
        if abs(nlimits[1]-nlimits[0]) < 5:
            nlimits[0] -= 1
        # centre = (temp[nlimits[0],0] + temp[nlimits[1],0])/2
        nmaxval = temp[nlimits[0]:nlimits[1],1].max()
        centre = temp[nlimits[0]:nlimits[1],0][np.where(temp[nlimits[0]:nlimits[1],1] == nmaxval)][-1]
    else:
        centre = temp[:,0][np.where(temp[:,1] == maxval)][-1]
    tempcrit = np.abs(temp[:,0]-centre)
    crit = tempcrit.min()
    cindex, = np.where(tempcrit == crit)
    if len(cindex) == 1:
        cindex = int(cindex)
    else:
        cindex = int(len(temp)/2)
    table = np.zeros((90,2))
    table[:,0] = np.arange(len(table)) + 1
    for x in np.arange(len(table)):
        try:
            table[x,1] = temp[cindex - table[x,0],1] + temp[cindex + table[x,0],1]
        except IndexError:
            break
    criterion = np.abs(table[:,1] - maxval)
    fwhm = int(table[:,0][np.where(criterion == criterion.min())].mean())
    step = np.array([fwhm])[0]
    if olimits[0] is not None and olimits[1] is not None:
        subset = profile[nlimits[0]:nlimits[1]]
    else:
        subset = profile[-step + cindex: step + cindex]
        counter, running = 0, 1
        while (len(subset) < 350) and running:
            step +=1
            subset = profile[-step + cindex: step + cindex]
            if ((subset[:3,1].mean() + subset[-3:,1].mean() - 2*zline)/zerr)**2 < 2.0:
                counter += 1
            if counter > 8:
                running = 0
            # print(((subset[:3,1].mean() + subset[-3:,1].mean() - 2*zline)/zerr)**2)
    if fixwidth is not None:
        subset = profile[-int(fixwidth) + cindex: int(fixwidth) + cindex]
        maxval = subset[:,1].max()
        zline = subset[:,1].min()
        fwhm = subset[:,0].std()/2.0
        centre = subset[:,0].mean()
        if len(subset) < 5:
            print("Not enough points in the curve.")
            return None, None, None
        pfit, pcov, infodict, errmsg, success = leastsq(fit_gaussian, np.array([maxval, fwhm, centre, zline]), args=(subset,), full_output=1)
        if pcov is not None:
            s_sq = (fit_gaussian(pfit, subset)**2).sum()/(len(subset)-len(pfit))
            a, FWHM, centre, zeroline = pfit
            tpeak = gaussian(a, FWHM, centre, subset[:,0]) + zeroline
            chi2 = chisquared(tpeak, subset,  zeroline)
            pcov = pcov * s_sq
    else:
        if len(subset) < 5:
            print("Not enough points in the curve.")
            return None, None,  None
        pfit, pcov, infodict, errmsg, success = leastsq(fit_gaussian_fixedbase, np.array([maxval, fwhm, centre]), args=(subset, zline), full_output=1)
        if pcov is not None:
            s_sq = (fit_gaussian_fixedbase(pfit, subset, zline)**2).sum()/(len(subset)-len(pfit))
            a, FWHM, centre = pfit
            tpeak = gaussian(a, FWHM, centre, subset[:,0]) + zline
            chi2 = chisquared(tpeak, subset, zline)
            pcov = pcov * s_sq
    error = [] 
    for i in range(len(pfit)):
        try:
            error.append(np.absolute(pcov[i][i])**0.5)
        except:
            error.append( 0.00 )
    optimised = [pfit, np.array(error)]
    a,b,c = optimised[0][0], optimised[0][1], optimised[0][2]
    newx =  np.linspace(subset[0,0], subset[-1,0], len(subset)*10)
    if fixwidth is not None:
        peak = gaussian(a,b,c, newx) + optimised[0][3]
    else:
        peak = gaussian(a,b,c, newx) + zline
    return optimised, np.column_stack([newx, peak]),  chi2

def elastic_line_anyx(profile, background, fixwidth = None, olimits = (None, None), init_fwhm = 0.05):
    """
    As an initial guess, this function will assume that the 
    elastic line is at the point with the highest intensity.
    Otherwise, an override for the peak centre can be provided.
    """
    chi2 = -1.0
    if len(background) > 1:
        zline, zerr = background[:,1].mean(), background[:,1].std()
    else:
        zline = profile[:, 1].min()
        zerr = abs(profile[:, 1].max() - profile[:, 1].min())**0.5
    temp = profile.copy()
    temp[:,1] -= zline
    maxval = temp[:,1].max()
    if olimits[0] is not None and olimits[1] is not None:
        # centre = ((temp[:, 0][np.where(crit1==min1)] + temp[:, 0][np.where(crit2==min2)])/2)[0]
        if olimits[0] <= temp[:, 0].min():
            if olimits[1] >= temp[:, 0].max():
                subset = profile
            else:
                subset = profile[:np.argmax(temp[:, 0] >= olimits[1])]
        else:
            if olimits[1] >= temp[:, 0].max():
                subset = profile[np.argmax(temp[:, 0] >= olimits[0]):]
            else:
                subset = profile[np.argmax(temp[:, 0] >= olimits[0]):np.argmax(temp[:, 0] >= olimits[1])]
        if len(subset) < 2:
            return None, None,  None
        centre = subset[:,0][np.where(subset[:,1] == subset[:, 1].max())][-1]
    else:
        subset = None
        centre = temp[:,0][np.where(temp[:,1] == maxval)][-1]
    tempcrit = np.abs(temp[:,0]-centre)
    crit = tempcrit.min()
    cindex, = np.where(tempcrit == crit)
    if len(cindex) == 1:
        cindex = int(cindex)
    else:
        cindex = int(cindex.mean())
    table = np.zeros((90,2))
    table[:,0] = np.arange(len(table)) + 1
    for x in np.arange(len(table)):
        try:
            table[x,1] = temp[cindex - table[x,0],1] + temp[cindex + table[x,0],1]
        except IndexError:
            break
    criterion = np.abs(table[:,1] - maxval)
    fwhm = int(table[:,0][np.where(criterion == criterion.min())].mean())
    step = np.array([fwhm])[0]
    if subset is None:
        subset = profile[-step + cindex: step + cindex]
        counter, running = 0, 1
        while (len(subset) < 350) and running:
            step +=1
            subset = profile[-step + cindex: step + cindex]
            if ((subset[:3,1].mean() + subset[-3:,1].mean() - 2*zline)/zerr)**2 < 2.0:
                counter += 1
            if counter > 8:
                running = 0
            # print(((subset[:3,1].mean() + subset[-3:,1].mean() - 2*zline)/zerr)**2)
    if fixwidth is not None:
        subset = profile[-int(fixwidth) + cindex: int(fixwidth) + cindex]
        maxval = subset[:,1].max()
        zline = subset[:,1].min()
        fwhm = subset[:,0].std()/2.0
        centre = subset[:,0].mean()
        if len(subset) < 5:
            print("Not enough points in the curve.")
            return None, None,  None
        pfit, pcov, infodict, errmsg, success = leastsq(fit_gaussian, np.array([maxval, fwhm, centre, zline]), args=(subset,), full_output=1)
        if pcov is not None:
            s_sq = (fit_gaussian(pfit, subset)**2).sum()/(len(subset)-len(pfit))
            a, FWHM, centre, zeroline = pfit
            tpeak = gaussian(a, FWHM, centre, subset[:,0]) + zeroline
            chi2 = chisquared(tpeak, subset,  zeroline)
            pcov = pcov * s_sq
    else:
        if len(subset) < 5:
            print("Not enough points in the curve.")
            return None, None,  None
        pfit, pcov, infodict, errmsg, success = leastsq(fit_gaussian_fixedbase, np.array([maxval, init_fwhm, centre]), args=(subset, zline), full_output=1)
        if pcov is not None:
            s_sq = (fit_gaussian_fixedbase(pfit, subset, zline)**2).sum()/(len(subset)-len(pfit))
            a, FWHM, centre = pfit
            tpeak = gaussian(a, FWHM, centre, subset[:,0]) + zline
            chi2 = chisquared(tpeak, subset, zline)
            pcov = pcov * s_sq
    error = [] 
    for i in range(len(pfit)):
        try:
            error.append(np.absolute(pcov[i][i])**0.5)
        except:
            error.append( 0.00 )
    optimised = [pfit, np.array(error)]
    a,b,c = optimised[0][0], optimised[0][1], optimised[0][2]
    newx =  np.linspace(subset[0,0], subset[-1,0], len(subset)*10)
    if fixwidth is not None:
        peak = gaussian(a,b,c, newx) + optimised[0][3]
    else:
        peak = gaussian(a,b,c, newx) + zline
    return optimised, np.column_stack([newx, peak]),  chi2

# def discrete_rebin(data, offset = 0.0, padding = 6, nsubdiv = 100):
    # startlen = len(data) + 2*padding
    # newdata = np.zeros(nsubdiv*startlen)
    # for n, i in enumerate(data):
        # newdata[int((padding+offset+n)*nsubdiv):int((padding+offset+n+1)*nsubdiv)] = i/nsubdiv
    # target = np.zeros(len(data))
    # for n in range(len(data)):
        # target[n] = newdata[int((padding+n)*nsubdiv):int((padding+n+1)*nsubdiv)].sum()
    # return target
    
def simplify_number_range(flist):
    numbs = []
    prefs = []
    for generic, fname in enumerate(flist):
        fpath, shortname = os.path.split(fname)
        shortname = ".".join(shortname.split('.')[:-1])
        #for bad_start in ['Merged_', 'BestMerge_']:
        #    shortname = shortname.lstrip(bad_start)
        #for bad_end in ['_1D_deltaE', '_1D']:
        #    shortname = shortname.rstrip(bad_end)
        if 'Merged_' in shortname[:7]:
            shortname = shortname[7:]
        elif 'BestMerge_'in shortname[:10]:
            shortname = shortname[10:]
        if '_1D' in shortname[-3:]:
            shortname = shortname[:-3]
        elif '_1D_deltaE' in shortname[-10:]:
            shortname = shortname[:-10]
        for n in range(len(shortname)):
            try:
                fnumber = int(shortname[n:])
            except:
                continue
            else:
                prefix = shortname[:n]
                break
        try:
            fnumber
        except:
            fnumber = generic+1
            prefix = shortname
        numbs.append(fnumber)
        prefs.append(prefix)
    prefs = np.array(prefs)
    numbs = np.array(numbs).astype(np.int)
    unique_names = np.unique(prefs)
    textsegs = []
    for un in unique_names:
        cname = un
        aaa = numbs[np.where(prefs == un)]
        subnums = np.sort(aaa)
        ranges = []
        for n,  val in enumerate(subnums):
            if n == 0:
                beg = val
                end = val
            else:
                if val == (subnums[n-1] + 1):
                    end = val
                else:
                    ranges.append((beg,  end))
                    beg = val
                    end = val
            if n==(len(subnums)-1):
                ranges.append((beg,  end))
        for rr in ranges:
            if rr[0] == rr[1]:
                textsegs.append(cname+str(rr[0]))
            else:
                textsegs.append(cname+str(rr[0])+'-'+str(rr[1]))
    return textsegs
    
def unit_to_int(text):
    if 'hanne' in text:
        return 0
    elif 'ransfe' in text:
        return 1
    elif 'nergy' in text:
        return 2
    else:
        return -1

def int_to_unit(num):
    if num == 0:
        return "Detector channels"
    elif num ==1:
        return "Energy transfer [eV]"
    elif num ==2:
        return "Energy [eV]"
    else:
        return "Unknown"
    
def discrete_rebin(data, offset = 0.0, padding = 6, nsubdiv = 100):
    startlen = len(data) # + 2*padding
    newdata = np.ones((startlen, nsubdiv)) / nsubdiv
    newdata = (newdata.T * data).T
    fulllen = startlen*nsubdiv
    newdata = newdata.reshape(fulllen)
    target = np.zeros(len(data))
    firstone, lastone = int((offset)*nsubdiv), int((offset+startlen)*nsubdiv)
    if firstone < 0:
        startind = 0
    else:
        startind = firstone
    newdata = newdata[startind:lastone+1]
    templen = len(newdata)
    missing = fulllen - templen
    if firstone < 0:
        newdata = np.concatenate([np.zeros(missing), newdata])
    elif lastone >= fulllen:
        newdata = np.concatenate([newdata, np.zeros(missing)])
    newdata = newdata.reshape((startlen, nsubdiv))
    target = newdata.sum(1)
    # for n in range(len(data)):
    #     target[n] = newdata[int((offset+n)*nsubdiv):int((offset+n+1)*nsubdiv)].sum()
    return target
    
def continuous_rebin(indata, offset = 0.0):
    if len(indata.shape) == 1:
        data = np.column_stack([np.arange(len(indata)), indata])
        returnYonly = True
    else:
        data = indata
        returnYonly = False
    newdata = data.copy()
    newdata[:, 1:] = 0.0
    newdata[:, 0] -= offset
    the_object = MergeCurves(data, newdata, tpool = None, pbar = None, mthreads = 1)
    the_object.runit()
    target = the_object.postprocess()
    if returnYonly:
        target = target[:, 1]
    return target

def merge2curves_old(source, target):
    """
    This function will treat two curves with different binning as histograms,
    and merge the data proportionally to the bin overlap.
    The input curves have to be sorted in ascending order along the x axis.
    """
    xs1, ys1 = source[:,0], source[:,1]
    xs2, ys2 = target[:,0], target[:,1]
    step1 = xs1[1:] - xs1[:-1]
    step1 = np.concatenate([step1[:1], step1, step1[-1:]])
    lowlim1 = xs1 - 0.5*step1[:-1]
    highlim1 = xs1 + 0.5*step1[1:]
    # print('step1: ', step1.min(), step1.max(), step1.mean())
    # now we have low limits, high limits and middle for each bin, plus the bin size
    step2 = xs2[1:] - xs2[:-1]
    step2 = np.concatenate([step2[:1], step2, step2[-1:]])
    lowlim2 = xs2 - 0.5*step2[:-1]
    highlim2 = xs2 + 0.5*step2[1:]
    # print('step2: ', step2.min(), step2.max(), step2.mean())
    # now we can begin
    newy = ys2.copy()
    lowest_n = 0
    highest_n = len(ys2)-1
    lowest_m=0
    highest_m=len(ys1)-1
    # acc_n = []
    for n in np.arange(len(xs2)):
        m1 = np.argmax(highlim1 > lowlim2[n])
        m2 = np.argmax(lowlim1 > highlim2[n])
        if highlim1[0] > lowlim2[n]:
            m1 = 0
        elif highlim1[-1] < lowlim2[n]:
            m1 = len(highlim1) -1
        elif highlim1[-2] < lowlim2[n]:
            m1 = len(highlim1) -2
        if lowlim1[0] > highlim2[n]:
            m2 = 0
        elif lowlim1[-1] < highlim2[n]:
            m2 = len(lowlim1)-1
        elif lowlim1[-2] < highlim2[n]:
            m2 = len(lowlim1)-2
        # now m1 is the first bin in the source curve that overlaps with the current bin
        # and m2 is the first bin after that that doesn't
        for m in np.arange(m1-2,m2+2):
            if (m > highest_m) or (m < lowest_m):
                continue
            if (highlim1[m] < lowlim2[n]) or (lowlim1[m] > highlim2[n]):
                continue
            if lowlim1[m] <= lowlim2[n]:
                if highlim1[m] > highlim2[n]:
                    newy[n] += ys1[m]*(highlim2[n]-lowlim2[n])/step1[m]
                else:
                    newy[n] += ys1[m]*(highlim1[m]-lowlim2[n])/step1[m]
            else:
                if highlim1[m] > highlim2[n]:
                    newy[n] += ys1[m]*(highlim2[n]-lowlim1[m])/step1[m]
                else:
                    newy[n] += ys1[m]
            # factor = (highlim1[m] - max(lowlim2[n],lowlim1[m])/step1[m]
            # altfactor = (min(highlim2[n],highlim1[m]) - lowlim1[m])/step1[m]
    # acc_n = np.array(acc_n)
    # print('acc_n: ', acc_n.min(0), acc_n.max(0), acc_n.mean(0))
    return np.column_stack([xs2,newy])

def merge2curves_errors_old(source, target):
    """
    This function will treat two curves with different binning as histograms,
    and merge the data proportionally to the bin overlap.
    The input curves have to be sorted in ascending order along the x axis.
    """
    xs1, ys1, errs1 = source[:,0], source[:,1], source[:,2]**2
    xs2, ys2, errs2 = target[:,0], target[:,1], target[:,2]**2
    # xs1 = xs1[np.where(np.abs(xs1))]
    step1 = xs1[1:] - xs1[:-1]
    print("Steps min, max: ",  step1.min(),  step1.max())
    step1 = np.concatenate([step1[:1], step1, step1[-1:]])
    lowlim1 = xs1 - 0.5*step1[:-1]
    highlim1 = xs1 + 0.5*step1[1:]
    # print('step1: ', step1.min(), step1.max(), step1.mean())
    # now we have low limits, high limits and middle for each bin, plus the bin size
    step2 = xs2[1:] - xs2[:-1]
    step2 = np.concatenate([step2[:1], step2, step2[-1:]])
    lowlim2 = xs2 - 0.5*step2[:-1]
    highlim2 = xs2 + 0.5*step2[1:]
    # print('step2: ', step2.min(), step2.max(), step2.mean())
    # now we can begin
    newy = ys2.copy()
    newerr = errs2.copy()
    lowest_n = 0
    highest_n = len(ys2)-1
    lowest_m=0
    highest_m=len(ys1)-1
    # acc_n = []
    for n in np.arange(len(xs2)):
        m1 = np.argmax(highlim1 > lowlim2[n])
        m2 = np.argmax(lowlim1 > highlim2[n])
        if highlim1[0] > lowlim2[n]:
            m1 = 0
        elif highlim1[-1] < lowlim2[n]:
            m1 = len(highlim1) -1
        elif highlim1[-2] < lowlim2[n]:
            m1 = len(highlim1) -2
        if lowlim1[0] > highlim2[n]:
            m2 = 0
        elif lowlim1[-1] < highlim2[n]:
            m2 = len(lowlim1)-1
        elif lowlim1[-2] < highlim2[n]:
            m2 = len(lowlim1)-2
        # now m1 is the first bin in the source curve that overlaps with the current bin
        # and m2 is the first bin after that that doesn't
        for m in np.arange(m1-2,m2+2):
            if (m > highest_m) or (m < lowest_m):
                continue
            if (highlim1[m] < lowlim2[n]) or (lowlim1[m] > highlim2[n]):
                continue
            if lowlim1[m] <= lowlim2[n]:
                if highlim1[m] > highlim2[n]:
                    frac = (highlim2[n]-lowlim2[n])/step1[m]
                    newy[n] += ys1[m]*frac
                    newerr[n] += errs1[m]*frac
                else:
                    frac = (highlim1[m]-lowlim2[n])/step1[m]
                    newy[n] += ys1[m]*frac
                    newerr[n] += errs1[m]*frac
            else:
                if highlim1[m] > highlim2[n]:
                    frac = (highlim2[n]-lowlim1[m])/step1[m]
                    newy[n] += ys1[m]*frac
                    newerr[n] += errs1[m]*frac
                else:
                    newy[n] += ys1[m]
                    newerr[n] += errs1[m]
            # factor = (highlim1[m] - max(lowlim2[n],lowlim1[m])/step1[m]
            # altfactor = (min(highlim2[n],highlim1[m]) - lowlim1[m])/step1[m]
    # acc_n = np.array(acc_n)
    # print('acc_n: ', acc_n.min(0), acc_n.max(0), acc_n.mean(0))
    return np.column_stack([xs2,newy, np.sqrt(newerr)])
    
def merge2curves(source, target):
    """
    This function will treat two curves with different binning as histograms,
    and merge the data proportionally to the bin overlap.
    The input curves have to be sorted in ascending order along the x axis.
    """
    xs1, ys1 = source[:,0], source[:,1]
    xs2, ys2 = target[:,0], target[:,1]
    step1 = xs1[1:] - xs1[:-1]
    step1 = np.concatenate([step1[:1], step1, step1[-1:]])
    lowlim1 = xs1 - 0.5*step1[:-1]
    highlim1 = xs1 + 0.5*step1[1:]
    # print('step1: ', step1.min(), step1.max(), step1.mean())
    # now we have low limits, high limits and middle for each bin, plus the bin size
    step2 = xs2[1:] - xs2[:-1]
    step2 = np.concatenate([step2[:1], step2, step2[-1:]])
    lowlim2 = xs2 - 0.5*step2[:-1]
    highlim2 = xs2 + 0.5*step2[1:]
    # print('step2: ', step2.min(), step2.max(), step2.mean())
    # now we can begin
    newy = ys2.copy()
    # acc_n = []
    limits1 = np.concatenate([lowlim1,  highlim1[-1:]])
    limits2 = np.concatenate([lowlim2,  highlim2[-1:]])
    overlap = 1.0 - np.abs( ( limits2.reshape((len(limits2), 1)) - limits1.reshape((1, len(limits1))) ) /step1.reshape( (1, len(step1) ) ) )
    temp = np.zeros(overlap.shape)
    temp[np.where(overlap > 0.0)] = overlap[np.where(overlap > 0.0)]
    temp[:, :-1] *= ys1.reshape((1,  len(ys1)))
    temp2 = temp.sum(1)
    newy += temp2[:-1]
    results = np.column_stack([xs2,newy])
    return results

def merge2curves_errors(source, target):
    """
    This function will treat two curves with different binning as histograms,
    and merge the data proportionally to the bin overlap.
    The input curves have to be sorted in ascending order along the x axis.
    """
    xs1, ys1, errs1 = source[:,0], source[:,1], source[:,2]**2
    xs2, ys2, errs2 = target[:,0], target[:,1], target[:,2]**2
    # xs1 = xs1[np.where(np.abs(xs1))]
    step1 = xs1[1:] - xs1[:-1]
    print("Steps min, max: ",  step1.min(),  step1.max())
    step1 = np.concatenate([step1[:1], step1, step1[-1:]])
    lowlim1 = xs1 - 0.5*step1[:-1]
    highlim1 = xs1 + 0.5*step1[1:]
    # print('step1: ', step1.min(), step1.max(), step1.mean())
    # now we have low limits, high limits and middle for each bin, plus the bin size
    step2 = xs2[1:] - xs2[:-1]
    step2 = np.concatenate([step2[:1], step2, step2[-1:]])
    lowlim2 = xs2 - 0.5*step2[:-1]
    highlim2 = xs2 + 0.5*step2[1:]
    # print('step2: ', step2.min(), step2.max(), step2.mean())
    # now we can begin
    newy = ys2.copy()
    newerr = errs2.copy()
    # acc_n = []
    limits1 = np.concatenate([lowlim1,  highlim1[-1:]])
    limits2 = np.concatenate([lowlim2,  highlim2[-1:]])
    overlap = 1.0 - np.abs( ( limits2.reshape((len(limits2), 1)) - limits1.reshape((1, len(limits1))) ) /step1.reshape( (1, len(step1) ) ) )
    temp = np.zeros(overlap.shape)
    temp[np.where(overlap > 0.0)] = overlap[np.where(overlap > 0.0)]
    temp2 = temp.copy()
    temp[:, :-1] *= ys1.reshape((1,  len(ys1)))
    newy += temp[:, :-1].sum(1)
    temp2[:, :-1] *= errs1.reshape((1,  len(errs1)))
    newerr += temp2[:, :-1].sum(1)
    results = np.column_stack([xs2,newy[:-1], np.sqrt(newerr[:-1])])
    return results

def normalising_merge_curves(sources, target):
    """
    This function will treat two curves with different binning as histograms,
    and merge the data proportionally to the bin overlap.
    The input curves have to be sorted in ascending order along the x axis.
    """
    xs2, ys2 = target[:,0], target[:,1]
    # now we have low limits, high limits and middle for each bin, plus the bin size
    step2 = xs2[1:] - xs2[:-1]
    step2 = np.concatenate([step2[:1], step2, step2[-1:]])
    lowlim2 = xs2 - 0.5*step2[:-1]
    highlim2 = xs2 + 0.5*step2[1:]
    newy = ys2.copy()
    newnorm = np.zeros(newy.shape)
    for source in sources:
        xs1, ys1 = source[:,0], source[:,1]
        step1 = xs1[1:] - xs1[:-1]
        step1 = np.concatenate([step1[:1], step1, step1[-1:]])
        lowlim1 = xs1 - 0.5*step1[:-1]
        highlim1 = xs1 + 0.5*step1[1:]
        # print('step1: ', step1.min(), step1.max(), step1.mean())
        # print('step2: ', step2.min(), step2.max(), step2.mean())
        # now we can begin
        lowest_n = 0
        highest_n = len(ys2)-1
        lowest_m=0
        highest_m=len(ys1)-1
        # acc_n = []
        for n in np.arange(len(xs2)):
            m1 = np.argmax(highlim1 > lowlim2[n])
            m2 = np.argmax(lowlim1 > highlim2[n])
            if highlim1[0] > lowlim2[n]:
                m1 = 0
            elif highlim1[-1] < lowlim2[n]:
                m1 = len(highlim1) -1
            elif highlim1[-2] < lowlim2[n]:
                m1 = len(highlim1) -2
            if lowlim1[0] > highlim2[n]:
                m2 = 0
            elif lowlim1[-1] < highlim2[n]:
                m2 = len(lowlim1)-1
            elif lowlim1[-2] < highlim2[n]:
                m2 = len(lowlim1)-2
            # now m1 is the first bin in the source curve that overlaps with the current bin
            # and m2 is the first bin after that that doesn't
            for m in np.arange(m1-2,m2+2):
                if (m > highest_m) or (m < lowest_m):
                    continue
                if (highlim1[m] < lowlim2[n]) or (lowlim1[m] > highlim2[n]):
                    continue
                if lowlim1[m] <= lowlim2[n]:
                    if highlim1[m] > highlim2[n]:
                        frac = (highlim2[n]-lowlim2[n])/step1[m]
                        newy[n] += ys1[m]*frac
                        newnorm[n] += frac
                    else:
                        frac = (highlim1[m]-lowlim2[n])/step1[m]
                        newy[n] += ys1[m]*frac
                        newnorm[n] += frac
                else:
                    if highlim1[m] > highlim2[n]:
                        frac = (highlim2[n]-lowlim1[m])/step1[m]
                        newy[n] += ys1[m]*frac
                        newnorm[n] += frac
                    else:
                        newy[n] += ys1[m]
                        newnorm[n] += 1.0
                # factor = (highlim1[m] - max(lowlim2[n],lowlim1[m])/step1[m]
                # altfactor = (min(highlim2[n],highlim1[m]) - lowlim1[m])/step1[m]
        # acc_n = np.array(acc_n)
        # print('acc_n: ', acc_n.min(0), acc_n.max(0), acc_n.mean(0))
    return np.column_stack([xs2,newy/newnorm])

def quick_match_profiles(data,  to_match, xshift = 0.0,  tpoolin = None, pbarin = None,  maxthreads = 1):
    tempdat = data.copy()
    tempdat[:, 0] -= xshift
    temptarget = to_match.copy()
    temptarget[:, 1] = 0.0
    # rebinned = merge2curves(tempdat,  temptarget)
    # the_object = MergeCurves(tempdat,  temptarget,  tpool = tpoolin,  pbar = pbarin)
    the_object = MergeCurves(tempdat,  temptarget, None,  pbar = pbarin,  mthreads = maxthreads)
    the_object.runit()
    rebinned = the_object.postprocess()
    pfit = leastsq(scaling_fit, [1.0, 0.0], args = (rebinned[10:-10, 1], to_match[10:-10, 1]))
    temp = rebinned[10:-10, 1]*pfit[0][0] + pfit[0][1]
    return (temp - to_match[10:-10, 1])

def scaling_fit(args, data, to_match):
    temp = args[0] * data + args[1]
    return temp - to_match

def shgo_profile_offsets(args,  data,  to_match, tpoolin = None, pbarin = None,  maxthreads = 1):
#    prec30 = np.percentile(to_match[:, 1], 30)
#    prec100 = to_match[:, 1].max()
#    A = np.percentile(data[:, 1], 30)
#    B = data[:, 1].max()
#    N = (A-B)/(prec30-prec100)
#    M = (A+B-N*(prec30 + prec100))/2.0
#    data[:, 1] *= N
#    data[:, 1] += M
    retval = quick_match_profiles(data,  to_match, args[0],  tpoolin, pbarin,  maxthreads)
    return (retval**2).sum()

def profile_offsets(args,  data,  to_match, tpoolin = None, pbarin = None,  maxthreads = 1):
    xshift = args[0]
    yshift = args[1]
    yscale = args[2]
    tempdat = data.copy()
    tempdat[:, 0] -= xshift
    tempdat[:, 1] *= yscale
    tempdat[:, 1] += yshift
    temptarget = to_match.copy()
    temptarget[:, 1] = 0.0
    # rebinned = merge2curves(tempdat,  temptarget)
    # the_object = MergeCurves(tempdat,  temptarget,  tpool = tpoolin,  pbar = pbarin)
    the_object = MergeCurves(tempdat,  temptarget, None,  pbar = pbarin,  mthreads = maxthreads)
    the_object.runit()
    rebinned = the_object.postprocess()
    return (rebinned[10:-10,1] - to_match[10:-10, 1])

#def apply_offset_to_2D_data(array, locut, hicut, offset):
#    """This is a dummy function for now.
#    Later on, it will apply a function to transform the image,
#    so that all the lines are straight instead of curved.
#    """
#    xaxis = np.arange(locut, hicut, 1) + 1 
#    result = np.zeros(array.shape)
#    for n, i in enumerate(xaxis):
#        result[:,n] = discrete_rebin(array[:,n], offset = offset, padding = 6, nsubdiv=100)
#    return result

def apply_offset_to_2D_data(inarray, locut, hicut, offset):
    """This is a dummy function for now.
    Later on, it will apply a function to transform the image,
    so that all the lines are straight instead of curved.
    """
    xaxis = np.arange(locut, hicut, 1) + 1 
    result = np.zeros(inarray.shape)
    for n, i in enumerate(xaxis):
        result[:,n] = discrete_rebin(inarray[:,n], offset = offset)
    return result

def chisquared(peak, data,  zline):
    temp = peak-zline
    toppoint = temp.max()
    crit = np.where(np.abs(temp)/abs(toppoint) > 0.001)
    peakpoints = len(peak[crit].ravel())
    print(peakpoints,  len(peak))
    toppart = peak-data[:, 1]
    values = (toppart[crit])**2
    bottompart = (peak[crit])# **2
    values = values.sum()/bottompart.sum() /peakpoints
    # toppart = peak-data[:, 1]
    # values = (toppart*toppart) / np.abs(peak)
    return values

def curvature_profile(data2D, blocksize = 16, percentile = 70, override = None, olimits = (None, None)):
    width = data2D.shape[1]
    segnums = int(round(width/blocksize))
    segments = [data2D[:,int(n*blocksize):int((n+1)*blocksize)].sum(1) for n in np.arange(segnums)]
    results = []
    for num, seg in enumerate(segments):
        background = np.where(seg<np.percentile(seg, percentile))
        xaxis = np.arange(len(seg))+1
        prof = np.column_stack([xaxis, seg])
        back = prof[background]
        try:
            a,b, c = elastic_line(prof, back, override, olimits = olimits)
        except:
            continue
        else:
            if olimits[0] is None or olimits[1] is None:
                results.append([(num+0.5)*blocksize, a[0][2], a[1][2]])
            elif a[0][2] > olimits[0] and a[0][2] < olimits[1]:
                results.append([(num+0.5)*blocksize, a[0][2], a[1][2]])
    return np.array(results)

#### data processing part

def read_1D_curve(fname, xcol =0,  ycol=1, ecol= -1, comment = '#', sep = ','):
    curve = []
    envals = []
    units = 'Detector channels'
    source = loadtext_wrapper(fname)
    if source is None:
        print("could not read file: ", fname)
        return curve, envals, units
    else:
        for line in source:
            if len(line.split()) > 0:
                if comment in line.split()[0][0]:
                    if 'hoton energy' in line:
                        label = line.split('hoton energy')[-1].strip()
                        for tok in label.split():
                            try:
                                tval = float(tok)
                            except:
                                continue
                            else:
                                envals.append(tok)
                else:
                    if (sep == ' ') or (sep == ''):
                        xy = [float(z) for z in line.split()]
                    else:
                        try:
                            xy = [float(z) for z in line.split(sep)]
                        except:
                            xy = [float(z) for z in line.split()]
                    if ecol < 0:
                        curve.append((xy[xcol], xy[ycol]))
                    else:
                        curve.append((xy[xcol], xy[ycol], xy[ecol]))
        curve = np.array(curve)
        sorting = np.argsort(curve[:,0])
        curve = curve[sorting]
        print(curve.shape)
        # source.close()
        xmin,  xmax = curve[:, 0].min(),  curve[:, 0].max()
        if xmin <0:
            units = "Energy Transfer [eV]"
        elif xmax - xmin > 1000.0:
            units = "Detector channels"
        else:
            units = "Energy [eV]"
        if len(envals) == 0:
            envals = [-1.0]
    return curve, envals, units

def read_1D_curve_extended(fname, xcol =0,  ycol=1, ecol= -1, comment = '#', sep = ','):
    curve = []
    envals = []
    pardict = {'energy': -1.0, 
    'temperature':-1.0, 
    'Q':-1.0, 
    '2theta':-1.0
    }
    units = 'Detector channels'
    source = loadtext_wrapper(fname)
    if source is None:
        print("could not read file: ", fname)
        return curve, envals, units, pardict
    else:
        for line in source:
            if len(line.split()) > 0:
                if comment in line.split()[0][0]:
                    if 'hoton energy' in line:
                        label = line.split('hoton energy')[-1].strip()
                        for tok in label.split():
                            try:
                                tval = float(tok)
                            except:
                                continue
                            else:
                                envals.append(tok)
                                pardict['energy'] = tval
                    elif 'Temperature' in line:
                        label = line.split('Temperature')[-1].strip()
                        for tok in label.split():
                            try:
                                tval = float(tok)
                            except:
                                continue
                            else:
                                pardict['temperature'] = tval
                                break
                    elif 'Q' in line:
                        label = line.split('Q')[-1].strip()
                        for tok in label.split():
                            try:
                                tval = float(tok)
                            except:
                                continue
                            else:
                                pardict['Q'] = tval
                                break
                    elif '2 theta' in line:
                        label = line.split('2 theta')[-1].strip()
                        for tok in label.split():
                            try:
                                tval = float(tok)
                            except:
                                continue
                            else:
                                pardict['2theta'] = tval
                                break
                else:
                    if (sep == ' ') or (sep == ''):
                        xy = [float(z) for z in line.split()]
                    else:
                        try:
                            xy = [float(z) for z in line.split(sep)]
                        except:
                            xy = [float(z) for z in line.split()]
                    if ecol < 0:
                        curve.append((xy[xcol], xy[ycol]))
                    else:
                        curve.append((xy[xcol], xy[ycol], xy[ecol]))
        curve = np.array(curve)
        sorting = np.argsort(curve[:,0])
        curve = curve[sorting]
        print(curve.shape)
        # source.close()
        xmin,  xmax = curve[:, 0].min(),  curve[:, 0].max()
        if xmin <0:
            units = "Energy Transfer [eV]"
        elif xmax - xmin > 1000.0:
            units = "Detector channels"
        else:
            units = "Energy [eV]"
        if len(envals) == 0:
            envals = [-1.0]
    return curve, envals, units, pardict

def read_1D_xas(fname):
    curve = []
    teyvals, tpyvals = [], []
    ecol, teycol, tpycol = 0, -1, -1
    source = loadtext_wrapper(fname)
    if source is None:
        print("could not read file: ", fname)
        return curve, teyvals, tpyvals
    else:
        lastline = ""
        for line in source:
            if len(line.split()) > 0:
                if '#' in line.split()[0][0]:
                    lastline = line
                else:
                    try:
                        xy = [float(z) for z in line.split(',')]
                    except:
                        xy = [float(z) for z in line.split()]
                    curve.append(xy)
        head = lastline.strip('#\n').split(',')
        for coln, tok in enumerate(head):
            if tok == "E":
                ecol = coln
            elif tok == "CURR1":
                tpycol = coln
            elif tok == "CURR2":
                teycol = coln
        curve = np.array(curve)
        envals = curve[:, ecol]
        sorting = np.argsort(envals)
        envals = envals[sorting]
        curve = curve[sorting]
        print(curve.shape)
        if tpycol >= 0:
            tpyvals = curve[:, tpycol]
        else:
            tpyvals = np.zeros(envals.shape)
        if teycol >= 0:
            teyvals = curve[:, teycol]
        else:
            teyvals = np.zeros(envals.shape)
        # source.close()
    return curve, teyvals, tpyvals

def ReadFits(fname):
    """Reads the .sif file produced by the Andor iKon-L CCD camera.
    The dimensions of the pixel array can be changed if necessary.
    """
    fitsfile = fits_module.open(fname)
    img = fitsfile[0]
    d_array = img.data
    if d_array.shape[0] < 2050 and d_array.shape[1] < 2050: # we have a file from the ANDOR camera
        return d_array.astype(np.float64), ""
    else: # we have a file from the Sydor detector
        return d_array.T.astype(np.float64), ""

def ReadAsc(fname):
    """Reads the old 2D .asc file format which for some reason
    was used on PEAXIS in the very beginning.
    Or, if things don't work out, it doesn't read it.
    """
    source = open(fname, 'r')
    data = []
    for n, line in enumerate(source):
        if n >= 2048:
            continue
        else:
            templine = str(line).replace('\t', ' ')
            toks = [float(x) for x in templine.split()[1:]]
            data.append(toks)
    source.close()
    d_array = np.array(data)# .astype(np.float64)
    print('ASC dimensions:',  d_array.shape)
    return d_array.T, ""

def ReadAndor(fname, dimensions = (2048,2048), byte_size=4, data_type='c'):
    """Reads the .sif file produced by the Andor iKon-L CCD camera.
    The dimensions of the pixel array can be changed if necessary.
    """
    size = os.path.getsize(fname)
    source = open(fname, 'rb')
    nrows = dimensions[-1]
    header, data = [],[]
    bindata = None    
    total_length = 0
    header = np.fromfile(source, np.byte, size - 4*dimensions[0]*dimensions[1] -2*4, '') # 2772 is close
    print("Headersize",  size - 4*dimensions[0]*dimensions[1] -2*4)
    data = np.fromfile(source, np.float32, dimensions[0]*dimensions[1], '')
    header = bytes(header)
    lines = header.split(b'\n')
    header = []
    for n, line in enumerate(lines):
        try:
            header.append(line.decode('ascii'))
        except UnicodeDecodeError:
            print('header lines skipped:', n+1, 'with length:', len(line))
    source.close()
    return np.array(data).reshape(dimensions), header

def SphericalCorrection(inarray, params = [1.0,1.0,1.0], locut = 0, hicut = 2048, direct_offsets = None):
    """This is a dummy function for now.
    Later on, it will apply a function to transform the image,
    so that all the lines are straight instead of curved.
    """
    xaxis = np.arange(locut, hicut, 1) + 1 
    if params[0] > 5.0 and params[1] > 5.0 and params[2] >5.0:
        if direct_offsets is not None:
            offsets = direct_offsets.copy()
        else:
            return inarray
    else:
        offsets = polynomial(params, xaxis)
    offsets -= offsets.mean()
    result = inarray.copy()
    for n, i in enumerate(xaxis):
        # result[:,n] = discrete_rebin(array[:,n], offset = offsets[n], padding = 6, nsubdiv=100)
        result[:,n] = discrete_rebin(inarray[:,n], offset = offsets[n], padding = 6, nsubdiv=100)
    return result

def RemoveCosmics(array, NStd = 3):
    """This function will filter out the cosmic rays from the signal.
    At the moment it assumes that each column should have uniform counts.
    Then it keeps only the part that is within N standard deviations away from the mean.
    """
    for n, line in enumerate(array):
        lmean = line.mean()
        lstddev = line.std()
        badpix = np.where(np.abs(line-lmean) > (NStd*lstddev))
        goodpix = np.where(np.abs(line-lmean) <= (NStd*lstddev))
        newmean = line[goodpix].mean()
        array[n][badpix] = newmean
    return array

def WriteProfile(fname, profile,
                        header = None, params = [None,  None], 
                        varlog = [None,  None]):
    target = open(fname, 'w')
    if header is not None:
        for ln in header:
            newln = "".join([str(x) for x in ln if str(x).isprintable()])
            target.write(' '.join(['#',newln.strip().strip('\n'),'\n']))
    if params[0] is not None and params[1] is not None:
        keys, vals = params[0], params[1]
        for kk in keys:
            val = str(vals[kk])
            target.write(' '.join([str(x) for x in ['# ADLER',kk, ':', val.strip().strip('()[]\n'), '\n']]))
    if varlog[0] is not None and varlog[1] is not None:
        keys, vals = list(varlog[0]), varlog[1]
        if len(keys)>0:
            target.write("# VARIABLES TIMELOG\n")
            target.write(" ".join(['#'] + keys + ['\n']))
            for ln in range(len(vals[keys[0]])):
                target.write(" ".join(['#'] + [str(round(vals[k][ln], 5)) for k in keys] + ['\n']))
    for n, val in enumerate(profile):
        target.write(",".join([str(x) for x in [val[0], val[1]]]) + '\n')
    target.close()

def WriteEnergyProfile(fname, array, header, params = None):
    target = open(fname, 'w')
    for ln in header:
        newln = "".join([str(x) for x in ln if x.isprintable()])
        target.write(' '.join(['#',newln.strip().strip('\n'),'\n']))
    if params is not None:
        keys, vals = params[0], params[1]
        for kk in keys:
            try:
                val = str(vals[kk[0]][kk[1]])
            except:
                val = "Missing"
            target.write(' '.join([str(x) for x in ['#',kk[0], kk[1], ':', val.strip().strip('()\n'), '\n']]))
    for n, val in enumerate(array):
        target.write(",".join([str(x) for x in [val[0], val[1]]]) + '\n')
    target.close()

def make_profile(inarray, reduction = 1.0,  xaxis = None,  tpoolin = None, pbarin = None,  maxthreads = 1, limits = None):
    temparray = inarray[limits[2]:limits[3], limits[0] : limits[1]]
    profile = temparray.sum(1)
    initial_x = np.arange(len(profile))+1+limits[2]
    if xaxis is None:
        final_x = np.linspace(limits[2]+1, limits[3], int(len(profile)/reduction))
    else:
        final_x = xaxis.copy()
    final_y = np.zeros(len(final_x))
    finalprof = np.column_stack([final_x, final_y])
    profile = np.column_stack([initial_x, profile])
    # pprofile = merge2curves(profile, finalprof)
    # the_object = MergeCurves(profile,  finalprof, tpool = tpoolin, pbar = pbarin)
    the_object = MergeCurves(profile,  finalprof, tpool = None, pbar = pbarin, mthreads = maxthreads)
    the_object.runit()
    pprofile = the_object.postprocess()
    return pprofile

def make_stripe(data,  xlims,  ylims):
    stripe = data[ylims[0]:ylims[1],xlims[0]:xlims[1]].sum(0)
    stripe = np.column_stack([np.arange(xlims[0], xlims[1])+1, stripe])
    return stripe
    
def make_histogram(data,  xlims,  ylims):
    temp = data[ylims[0]:ylims[1],xlims[0]:xlims[1]]
    flat = np.ravel(temp)
    ordered = np.sort(flat)
    cons_limits = (ordered[2:10].mean() -0.5,  ordered[-10:-2].mean()+0.5)
    # nbins = max(int(round(len(flat)**0.5)),  10)
    nbins = max(int(round(cons_limits[1]) - round(cons_limits[0])) + 1,  3)
    hist,  binlims = np.histogram(temp,  bins = nbins,  range = cons_limits)
    return hist,  binlims

def load_lise(fname):
    header,  values,  finalvalues = [],  [],  []
    result,  names,  units = {}, {}, {}
    numbs,  sequence = [], []
    vallength = 0
    payattention = False
    dataspotted = False
    source = loadtext_wrapper(fname)   
    if not (source is None):
        for line in source:
            toks = line.split()
            if len(toks) > 0:
                if '#' in toks[0]:
                    header.append(line.strip('# '))
                else:
                    values.append(toks)
                    vallength = max(vallength,  len(toks))
    for line in values:
        if len(line) == vallength:
            finalvalues.append([float(x) for x in line])
    for line in header:
        if 'LEGEND' in line:
            payattention = True
        if not payattention:
            continue
        else:
            if ':' in line:
                parts = line.split(':')
                number = int(parts[0])
                name = parts[1].split('>')[0].strip(' ')
                unit = parts[1].split('in')[-1].strip(' ')
                numbs.append(number)
                names[number] = name
                units[number] = unit
            elif 'DATA' in line:
                dataspotted = True
            elif dataspotted:
                toks = line.split()
                if len(toks) == len(numbs):
                    sequence = [int(x) for x in toks]
    finalvalues = np.array(finalvalues)
    for num in numbs:
        result[num] = finalvalues[:, num]
    return result, names, units, numbs

def load_datheader(fname):
    header = []
    resdict, unitdict = {}, {}
    dollarcount = 0
    source = loadtext_wrapper(fname)  
    if not (source is None):
        for line in source:
            if len(line.split()) > 0:
                if '#' in line.split()[0][0]:
                    header.append(line)
                    if '$' in line:
                        dollarcount += 1
    # toplevel = DataGroup(default_class= DataGroup, label='Header')
    #     toplevel['other'] = DataGroup(label= 'other')
    #     for entry in lines:
    #         if '#' in entry:
    #             toks = entry.split('#')
    #             if len(toks) > 1:
    #                 category, subentry = toks[0], '#'.join(toks[1:])
    #                 if not category in toplevel.keys():
    #                     toplevel[category] = DataGroup(label = category)
    #                 relevantGroup = toplevel[category]
    #         else:
    #             relevantGroup = toplevel['other']
    #             subentry = entry
    #         key,  value = subentry.split('$')[0].strip(' '),  subentry.split('$')[1].strip(' ')
    #         relevantGroup[key] = DataEntry(value, key)
    #     self.the_header = toplevel
    if dollarcount > 10:
        for ll in header:
            toks = ll.split('$')
            if not len(toks) == 3:
                continue
            try:
                resdict[toks[0]] = float(toks[1])
            except:
                resdict[toks[0]] = toks[1]
            unitdict[toks[0]] = toks[2]
        return resdict,  unitdict
    for ll in header:
        toks = ll.split()
        foundit = False
        for tn,  tok in enumerate(toks):
            if tn > 1:
                try:
                    val = float(tok)
                except:
                    continue
                else:
                    foundit = True
                    keyval = " ".join(toks[:tn])
                    resdict[keyval] = val
                    unitdict[keyval] = toks[-1]
                    break
        if not foundit:
            ntoks = ll.split(':')
            if len(ntoks) > 1:
                resdict[ntoks[0]] = ntoks[1]
    return resdict,  unitdict

def load_datlog(fname):
    header,  values,  finalvalues = [],  [],  []
    vallength = 0
    finalheader = None
    source = loadtext_wrapper(fname)   
    if not (source is None):
        for line in source:
            toks = line.split()
            if len(toks) > 0:
                if '#' in toks[0]:
                    header.append(toks)
                else:
                    values.append(toks)
                    vallength = max(vallength,  len(toks))
    for line in values:
        if len(line) == vallength:
            try:
                finalvalues.append([float(x) for x in line])
            except:
                continue
    for line in header:
        if len(line) == vallength+1:
            finalheader = line[1:]
    finalvalues = np.array(finalvalues)
    result = {}
    if finalheader is not None:
        for n,  k in enumerate(finalheader):
            result[k] = finalvalues[:, n]
    return result

#class WorkerSignals(QObject):
#    '''
#    Defines the signals available from a running worker thread.
#
#    Supported signals are:
#
#    finished
#        No data
#    
#    error
#        `tuple` (exctype, value, traceback.format_exc() )
#    
#    result
#        `object` data returned from processing, anything
#
#    progress
#        `int` indicating % progress 
#
#    '''
#    finished = pyqtSignal()
#    error = pyqtSignal(tuple)
#    result = pyqtSignal(object)
#    progress = pyqtSignal(int)

def load_lise_logs(flist):
    logs, ns,  us, nbs = [], [], [], []
    vallog, unitlog = {}, {}
    for fname in flist:
        result, names, units, numbs = load_lise(fname)
        logs.append(result)
        ns.append(names)
        us.append(units)
        nbs.append(numbs)
    for nn, lg in enumerate(logs):
        names = ns[nn]
        units = us[nn]
        numbs = nbs[nn]
        lk = [names[int(x)] + " (" + units[int(x)].strip(' \r\n\t')  + ")" for x in lg.keys()]
        for nk, k in enumerate(lk):
            if k in vallog.keys():
                vallog[k] = np.concatenate([vallog[k],  lg[nk]])
            else:
                vallog[k] = lg[nk].copy()
    if 'time' in vallog.keys():
        sequence = np.argsort(vallog['time'])
        for k in vallog.keys():
            vallog[k] = vallog[k][sequence]
    return vallog
    
def load_only_logs(flist):
    logs = []
    vallog = {}
    for fname in flist:
        logs.append(load_datlog(fname))
    for lg in logs:
        lk = lg.keys()
        for k in lk:
            if k in vallog.keys():
                vallog[k] = np.concatenate([vallog[k],  lg[k]])
            else:
                vallog[k] = lg[k].copy()
    if 'Time' in vallog.keys():
        sequence = np.argsort(vallog['Time'])
        for k in vallog.keys():
            vallog[k] = vallog[k][sequence]
    return vallog

def guess_XAS_xaxis(vallog):
    relval = 1e15
    guessed = 'Step'
    try:
        temp = vallog['TARGET']
        pnum = len(temp)
    except KeyError:
        return guessed
    if np.all(vallog['TARGET']==0):
        kkvals, probfuncs = [], []
        for kk in vallog.keys():
            keystring = str(kk)
            if keystring in ['TARGET', 'RING', 'OPEN', 'LOCKED', 'XBPML_HOR', 'XBPML_VER',
                                'TEMP1', 'TEMP2', 'CURR1', 'CURR2', 'Step', 'Time', 'RelTime']:
                continue
            else:
                uniqvals = np.unique(vallog[keystring])
                uniqstep = uniqvals[1:] - uniqvals[:-1]
                stepsize = uniqstep.mean()
                stepvar = uniqstep.std()
                if abs(stepvar) < 1e-6 and abs(stepsize) < 1e-6:
                    continue
                costfunc = np.nan_to_num( np.array([stepvar]) / np.array([abs(stepsize)]) )[0] # correct it!
                kkvals.append(keystring)
                probfuncs.append(costfunc)
        probfuncs = np.array(probfuncs)
        target = probfuncs.min()
        for n in np.arange(len(probfuncs)):
            if probfuncs[n] == target:
                guessed = kkvals[n]
    else:
        for kk in vallog.keys():
            keystring = str(kk)
            if keystring in ['TARGET', 'RING', 'OPEN', 'LOCKED', 'XBPML_HOR', 'XBPML_VER',
                                'TEMP1', 'TEMP2', 'CURR1', 'CURR2']:
                continue
            else:
                comp = vallog[keystring]
                costfun = np.abs(comp-temp).sum()
                if costfun < relval:
                    relval = costfun
                    guessed = keystring
        # if (relval/pnum) > 0.1:
        #     print('XAS x-axis ',  guessed, 'was probably wrong. Replaced with Step.')
        #     guessed = 'Step'
    print('Guessed the XAS x-axis to be ', guessed)
    return guessed

def load_and_average_logs(flist):
    logs = []
    vallog = {}
    errlog = {}
    for fname in flist:
        logs.append(load_datlog(fname))
    for lg in logs:
        lk = lg.keys()
        for k in lk:
            if k in vallog.keys():
                vallog[k] = np.concatenate([vallog[k],  lg[k]])
            else:
                vallog[k] = lg[k].copy()
    if 'Time' in vallog.keys():
        sequence = np.argsort(vallog['Time'])
        for k in vallog.keys():
            vallog[k] = vallog[k][sequence]
    if 'TARGET' in vallog.keys():
        xkey = guess_XAS_xaxis(vallog)
        if np.all(vallog['TARGET']==0):
            xkey = 'Step'
            points = np.sort(np.unique(vallog[xkey]))
        else:
            points = np.sort(np.unique(vallog[xkey]))
        extended_points = np.concatenate([
            [points[0] - abs(points[1] - points[0])], 
            points,
            [points[-1] + abs(points[-1]-points[-2])]
        ])
        limits = (extended_points[1:] + extended_points[:-1])/2.0
        full = vallog[xkey]
        count = np.zeros(len(points)).astype(np.int)
        for nn in range(len(points)):
            count[nn] += len(full[np.where(np.logical_and(full >= limits[nn], full< limits[nn+1]))])
        limits = np.concatenate([limits[:1], limits[1:][np.where(count > 0)]])
        newpoints = (limits[1:] + limits[:-1])/2.0
        for k in vallog.keys():
            temp = vallog[k]
            newone,  newerr = [],  []
            for counter in range(len(newpoints)):
                subset = temp[np.where(np.logical_and(full < limits[counter+1], full >= limits[counter]))]
                newone.append(subset.mean())
                newerr.append(subset.std())
            vallog[k] = np.array(newone)
            errlog[k] = np.array(newerr)
    for kk in vallog.keys():
        try:
            temp = errlog[kk]
        except KeyError:
            errlog[kk] = np.zeros(vallog[kk].shape)
    return vallog, errlog, xkey

def load_filter_and_average_logs(flist, cutoff = None):
    logs = []
    vallog = {}
    errlog = {}
    for fname in flist:
        logs.append(load_datlog(fname))
    for lg in logs:
        lk = lg.keys()
        for k in lk:
            if k in vallog.keys():
                vallog[k] = np.concatenate([vallog[k],  lg[k]])
            else:
                vallog[k] = lg[k].copy()
    if 'Time' in vallog.keys():
        sequence = np.argsort(vallog['Time'])
        for k in vallog.keys():
            vallog[k] = vallog[k][sequence]
        # new_part
        if cutoff is None:
            cutoff = int(round(len(sequence)/2))
        for k in vallog.keys():
            if 'CURR' in str(k):
                new_y = rfft(vallog[k])
                new_y[-cutoff:] = 0.0
                vallog[k] = irfft(new_y)
    if 'TARGET' in vallog.keys():
        xkey = guess_XAS_xaxis(vallog)
        if np.all(vallog['TARGET']==0):
            xkey = 'Step'
            points = np.sort(np.unique(vallog[xkey]))
        else:
            points = np.sort(np.unique(vallog[xkey]))
        extended_points = np.concatenate([
            [points[0] - abs(points[1] - points[0])], 
            points,
            [points[-1] + abs(points[-1]-points[-2])]
        ])
        limits = (extended_points[1:] + extended_points[:-1])/2.0
        full = vallog[xkey]
        count = np.zeros(len(points)).astype(np.int)
        for nn in range(len(points)):
            count[nn] += len(full[np.where(np.logical_and(full >= limits[nn], full< limits[nn+1]))])
        limits = np.concatenate([limits[:1], limits[1:][np.where(count > 0)]])
        newpoints = (limits[1:] + limits[:-1])/2.0
        for k in vallog.keys():
            temp = vallog[k]
            newone,  newerr = [],  []
            for counter in range(len(newpoints)):
                subset = temp[np.where(np.logical_and(full < limits[counter+1], full >= limits[counter]))]
                newone.append(subset.mean())
                newerr.append(subset.std())
            vallog[k] = np.array(newone)
            errlog[k] = np.array(newerr)
    for kk in vallog.keys():
        try:
            temp = errlog[kk]
        except KeyError:
            errlog[kk] = np.zeros(vallog[kk].shape)
    return vallog, errlog, xkey

def place_data_in_bins(bigarray, new_limits):
    points = (new_limits[1:] + new_limits[:-1])/2.0
    # full = vallog['E']
    spread = bigarray[:, 1].std()
    count = np.zeros(len(points)).astype(np.int)
    for nn in range(len(points)):
        count[nn] += len(bigarray[np.where(np.logical_and(bigarray[:, 0] >= new_limits[nn], bigarray[:, 0]< new_limits[nn+1]))])
    # new_limits = np.concatenate([new_limits[:1], new_limits[1:][np.where(count > 0)]])
    final = np.zeros((len(points), 3))
    final[:, 0] = points
    for counter in range(len(points)):
        subset = bigarray[:, 1][np.where(np.logical_and(bigarray[:, 0] < new_limits[counter+1], bigarray[:, 0] >= new_limits[counter]))]
        if count[counter] > 0:
            final[counter, 1] = subset.mean()
            final[counter, 2] = subset.std()
        else:
            final[counter, 1] = 0.0
            final[counter, 2] = spread
    return final

def place_points_in_bins(vallog,  redfac = 1.0):
    errlog = {}
    if 'Time' in vallog.keys():
        sequence = np.argsort(vallog['Time'])
        for k in vallog.keys():
            vallog[k] = vallog[k][sequence]
    if 'TARGET' in vallog.keys():
        xkey = guess_XAS_xaxis(vallog)
        if np.all(vallog['TARGET']==0):
            xkey = 'Step'
            points = np.sort(np.unique(vallog[xkey]))
        else:
            points = np.sort(np.unique(vallog[xkey]))
        extended_points = np.concatenate([
            [points[0] - abs(points[1] - points[0])], 
            points,
            [points[-1] + abs(points[-1]-points[-2])]
        ])
        limits = (extended_points[1:] + extended_points[:-1])/2.0
        lmin,  lmax,  lnum = limits.min(),  limits.max(), len(limits)
        new_limits = np.linspace(lmin, lmax,  int(lnum/redfac))
        points = (new_limits[1:] + new_limits[:-1])/2.0
        # xkey = guess_XAS_xaxis(vallog)
        full = vallog[xkey]
        count = np.zeros(len(points)).astype(np.int)
        for nn in range(len(points)):
            count[nn] += len(full[np.where(np.logical_and(full >= new_limits[nn], full< new_limits[nn+1]))])
        new_limits = np.concatenate([new_limits[:1], new_limits[1:][np.where(count > 0)]])
        for k in vallog.keys():
            temp = vallog[k]
            newone,  newerr = [],  []
            for counter in range(len(points)):
                subset = temp[np.where(np.logical_and(full < new_limits[counter+1], full >= new_limits[counter]))]
                newone.append(subset.mean())
                newerr.append(subset.std())
            vallog[k] = np.array(newone)
            errlog[k] = np.array(newerr)
    return vallog, errlog

class ShiftProfilesParallel(QObject):
    def __init__(self, profiles, tpool = None, pbar = None,  mthreads =1):
        super().__init__()
        self.reference = profiles[0]
        self.profiles = profiles[1:]
        self.shiftlist = []
        self.profilelist = []
        self.counter = 0
        self.mutex = QMutex()
        if tpool is not None:
            self.threadpool = tpool
        else:
            # self.threadpool = QThreadPool(self)
            self.threadpool = CustomThreadpool(MAX_THREADS = mthreads)
        self.progbar = pbar
        self.finished = False
        self.noproblems = False
        self.runnables = []
        # timer = QTimer(self)
        for n in range(len(self.profiles)):
            rr = ShiftOneProfile(self.profiles[n], self.reference,  report_to = self)
            self.runnables.append(rr)
            # rr.setAutoDelete(False)
            # rr.progress.connect(self.debug_output)
            # rr.result.connect(self.accept_values)
            # rr.finished.connect(self.increment)    @pyqtSlot(object)
    @pyqtSlot()
    def increment(self):
        self.mutex.lock()
        self.counter += 1
        # self.progbar.setValue(self.counter)
        self.mutex.unlock()
    def accept_values(self, results):
        self.mutex.lock()
        self.shiftlist.append(results[0])
        self.profilelist.append(results[1])
        self.mutex.unlock()
    def runit(self):
        for rr in self.runnables:
            self.threadpool.add_task(rr.run)
        self.threadpool.run()
        for rr in self.runnables:
            self.accept_values(rr.results)

class ShiftOneProfile(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)
    def __init__(self,  prof, match, spread = 25.0, report_to = None):
        super().__init__()
        # self.signals = WorkerSignals()
        self.reply_here = report_to
        self.p = prof.copy()
        self.match = match
        self.spread = spread
            #pfit, pcov, infodict, errmsg, success = leastsq(profile_offsets, [0.0, 1.0, 0.0],
            #                                                                       args = (p, profiles[0], self.threadpool, self.progbar, self.maxthreads), 
            #                                                                       full_output = 1)
            #offsets.append(pfit[0])
        self.finished.connect(self.reply_here.increment)
    @pyqtSlot()
    def run(self):
        results = shgo(shgo_profile_offsets, [(-self.spread, self.spread)], args = (self.p, self.match, None, None, 1), 
                            sampling_method = 'sobol'
                            )
        shift = results['x'][0]
        self.p[:, 0] -= shift
        self.results = [shift, self.p]
        self.finished.emit()  # Done

class LoadAndorFile(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)
    def __init__(self,  fname,  bpp,  cray, report_to = None):
        super().__init__()
        # self.signals = WorkerSignals()
        fpath, shortname = os.path.split(fname)
        if "." in shortname:
            nameroot = ".".join(shortname.split(".")[:-1])
        else:
            nameroot = shortname
        self.fpath = fpath
        self.fname = fname
        self.bpp = bpp
        self.cray = cray
        self.nameroot = nameroot
        self.reply_here = report_to
        self.finished.connect(self.reply_here.increment)
    @pyqtSlot()
    def run(self):
        # print("Starting a file-loading thread.")
        # self.reply_here.debug_output(15)
        data, header = ReadAndor(self.fname)
        tadded_header, units_for_header = load_datheader(os.path.join(self.fpath, self.nameroot+".dat"))
        tadded_log = load_datlog(os.path.join(self.fpath, self.nameroot+".xas"))
        if self.cray > 0.0:
            data = RemoveCosmics(data, NStd = self.cray)
        if self.bpp is None:
            self.bpp = np.percentile(data, 10).mean()*0.99
        data -= self.bpp
        print("Thread has data with shape",  data.shape)
        # self.result.emit((data, tadded_header, tadded_log,  self.fname))  # Return the result of the processing
        # self.reply_here.accept_values([data, tadded_header, tadded_log,  self.fname])
        # self.reply_here.increment()
        self.results = [data, tadded_header, tadded_log,  self.fname]
        self.finished.emit()  # Done

class MergeCurves(QObject):
    def __init__(self, source, target, tpool = None, pbar = None,  mthreads = 1):
        super().__init__()
        xs1, ys1 = source[:,0], source[:,1]
        xs2, ys2 = target[:,0], target[:,1]
        step1 = xs1[1:] - xs1[:-1]
        step1 = np.concatenate([step1[:1], step1, step1[-1:]])
        lowlim1 = xs1 - 0.5*step1[:-1]
        highlim1 = xs1 + 0.5*step1[1:]
        # print('step1: ', step1.min(), step1.max(), step1.mean())
        # now we have low limits, high limits and middle for each bin, plus the bin size
        step2 = xs2[1:] - xs2[:-1]
        step2 = np.concatenate([step2[:1], step2, step2[-1:]])
        lowlim2 = xs2 - 0.5*step2[:-1]
        highlim2 = xs2 + 0.5*step2[1:]
        # print('step2: ', step2.min(), step2.max(), step2.mean())
        # now we can begin
        newy = np.zeros(len(ys2))
        lowest_m=0
        highest_m=len(ys1)-1
        self.arrays = [lowlim1, lowlim2, highlim1, highlim2, step1, step2, xs2, newy, ys1]
        self.consts = [lowest_m, highest_m]
        self.mutex = QMutex()
        self.datalist = []
        if tpool is not None:
            self.threadpool = tpool
        else:
            # self.threadpool = QThreadPool(self)
            self.threadpool = CustomThreadpool(MAX_THREADS = mthreads)
        self.progbar = pbar
        self.counter = 0
        self.total = len(ys1)
        # self.progbar.setRange(self.counter,  self.total)
        self.target = target.copy()
        nthreads = mthreads
        limits = np.linspace(0, len(newy), nthreads+1).astype(int)
        self.runnables = []
        for nt in range(nthreads):
            self.runnables.append(CurveMergeWorker(limits[nt],  limits[nt+1], self.arrays,  self.consts, self))
    @pyqtSlot(int)
    def debug_output(self, somenum):
        print("Received a progress signal: ",  somenum)
    @pyqtSlot()
    def increment(self,  snum = 1):
        self.mutex.lock()
        self.counter += snum
        # self.progbar.setValue(self.counter)
        self.mutex.unlock()
    @pyqtSlot(object)
    def accept_values(self, results):
        self.mutex.lock()
        # print("Received values from thread, ",  results[3])
        self.datalist.append(results)
        self.mutex.unlock()
    def isitover(self):
        self.mutex.lock()
        self.finished = self.counter == self.total
        self.mutex.unlock()
    # def runit(self):       
    #     for rr in self.runnables:
    #         rr.run()
    #     for rr in self.runnables:
    #         self.accept_values(rr.results)
    def runit(self):
        for rr in self.runnables:
            self.threadpool.add_task(rr.fastrun)
        self.threadpool.run()
        for rr in self.runnables:
            self.accept_values(rr.results)
    def postprocess(self):
        for r in self.datalist:
            beg,  end = r[0]
            data = r[1]
            self.target[beg:end, 1] += data[beg:end, 1]
        return self.target

class MergeManyArrays(QObject):
    def __init__(self, arrays, offsets, tpool = None, pbar = None,  mthreads = 1):
        super().__init__()
        self.packages = []
        for nn, source in enumerate(arrays):
            xs1 = np.arange(0, len(source)) - offsets[nn]
            step1 = xs1[1:] - xs1[:-1]
            step1 = np.concatenate([step1[:1], step1, step1[-1:]])
            lowlim1 = xs1 - 0.5*step1[:-1]
            highlim1 = xs1 + 0.5*step1[1:]
            self.packages.append([xs1, step1,  lowlim1, highlim1, source])
        # print('step1: ', step1.min(), step1.max(), step1.mean())
        # now we have low limits, high limits and middle for each bin, plus the bin size
        xs2 = np.arange(0, len(source))
        step2 = xs2[1:] - xs2[:-1]
        step2 = np.concatenate([step2[:1], step2, step2[-1:]])
        lowlim2 = xs2 - 0.5*step2[:-1]
        highlim2 = xs2 + 0.5*step2[1:]
        # print('step2: ', step2.min(), step2.max(), step2.mean())
        # now we can begin
        # newarr = np.zeros(source.shape)
        # lowest_m=0
        # highest_m=len(ys1)-1
        # self.arrays = [lowlim1, lowlim2, highlim1, highlim2, step1, step2, xs2, newy, ys1]
        self.arrays = [lowlim2, highlim2, step2, xs2]
        # self.consts = [lowest_m, highest_m]
        self.mutex = QMutex()
        self.datalist = []
        if tpool is not None:
            self.threadpool = tpool
        else:
            # self.threadpool = QThreadPool(self)
            self.threadpool = CustomThreadpool(MAX_THREADS = mthreads)
        self.progbar = pbar
        self.counter = 0
        self.total = len(arrays)
        # self.progbar.setRange(self.counter,  self.total)
        # self.target = target.copy()
        # limits = np.linspace(0, len(newy), nthreads+1).astype(int)
        self.runnables = []
        for nt in range(len(self.packages)):
            self.runnables.append(ArrayMergeWorker(self.packages[nt], self.arrays, self))
    @pyqtSlot(int)
    def debug_output(self, somenum):
        print("Received a progress signal: ",  somenum)
    @pyqtSlot()
    def increment(self,  snum = 1):
        self.mutex.lock()
        self.counter += snum
        # self.progbar.setValue(self.counter)
        self.mutex.unlock()
    @pyqtSlot(object)
    def accept_values(self, results):
        self.mutex.lock()
        # print("Received values from thread, ",  results[3])
        self.datalist.append(results)
        self.mutex.unlock()
    def isitover(self):
        self.mutex.lock()
        self.finished = self.counter == self.total
        self.mutex.unlock()
    # def runit(self):       
    #     for rr in self.runnables:
    #         rr.run()
    #     for rr in self.runnables:
    #         self.accept_values(rr.results)
    def runit(self):
        for rr in self.runnables:
            self.threadpool.add_task(rr.fastrun)
        self.threadpool.run()
        for rr in self.runnables:
            self.accept_values(rr.results)
    def postprocess(self):
        temp = np.array(self.datalist)
        return temp.sum(0)

class MergeManyCurves(QObject):
    def __init__(self, sources, target, tpool = None, pbar = None,  mthreads = 1):
        super().__init__()
        self.packages = []
        for source in sources:
            xs1, ys1 = source[:,0], source[:,1]
            step1 = xs1[1:] - xs1[:-1]
            step1 = np.concatenate([step1[:1], step1, step1[-1:]])
            lowlim1 = xs1 - 0.5*step1[:-1]
            highlim1 = xs1 + 0.5*step1[1:]
            self.packages.append([xs1, ys1,  step1,  lowlim1, highlim1])
        # print('step1: ', step1.min(), step1.max(), step1.mean())
        # now we have low limits, high limits and middle for each bin, plus the bin size
        xs2, ys2 = target[:,0], target[:,1]
        step2 = xs2[1:] - xs2[:-1]
        step2 = np.concatenate([step2[:1], step2, step2[-1:]])
        lowlim2 = xs2 - 0.5*step2[:-1]
        highlim2 = xs2 + 0.5*step2[1:]
        # print('step2: ', step2.min(), step2.max(), step2.mean())
        # now we can begin
        newy = np.zeros(len(ys2))
        lowest_m=0
        highest_m=len(ys1)-1
        # self.arrays = [lowlim1, lowlim2, highlim1, highlim2, step1, step2, xs2, newy, ys1]
        self.arrays = [lowlim2, highlim2, step2, xs2, newy]
        self.consts = [lowest_m, highest_m]
        self.mutex = QMutex()
        self.datalist = []
        if tpool is not None:
            self.threadpool = tpool
        else:
            # self.threadpool = QThreadPool(self)
            self.threadpool = CustomThreadpool(MAX_THREADS = mthreads)
        self.progbar = pbar
        self.counter = 0
        self.total = len(ys1)
        # self.progbar.setRange(self.counter,  self.total)
        self.target = target.copy()
        # limits = np.linspace(0, len(newy), nthreads+1).astype(int)
        self.runnables = []
        for nt in range(len(sources)):
            self.runnables.append(CurveAltMergeWorker(self.packages[nt], self.arrays,  self.consts, self))
    @pyqtSlot(int)
    def debug_output(self, somenum):
        print("Received a progress signal: ",  somenum)
    @pyqtSlot()
    def increment(self,  snum = 1):
        self.mutex.lock()
        self.counter += snum
        # self.progbar.setValue(self.counter)
        self.mutex.unlock()
    @pyqtSlot(object)
    def accept_values(self, results):
        self.mutex.lock()
        # print("Received values from thread, ",  results[3])
        self.datalist.append(results)
        self.mutex.unlock()
    def isitover(self):
        self.mutex.lock()
        self.finished = self.counter == self.total
        self.mutex.unlock()
    # def runit(self):       
    #     for rr in self.runnables:
    #         rr.run()
    #     for rr in self.runnables:
    #         self.accept_values(rr.results)
    def runit(self):
        for rr in self.runnables:
            self.threadpool.add_task(rr.fastrun)
        self.threadpool.run()
        for rr in self.runnables:
            self.accept_values(rr.results)
    def postprocess(self):
        for r in self.datalist:
            self.target[:, 1] += r[:, 1]
        return self.target

@jit(nopython = True, parallel = True)
def helper_multiplier(overlap, data):
    newy = np.zeros(data.shape)
    oth = len(data)
    for nl in prange(data.shape[1]):
        line = overlap[:, :-1] * (data[:, nl].reshape((1,  oth)))
        line = line.sum(1)
        newy[:, nl] = line[:-1]
    return newy

@jit(parallel = True)
def helper_multiplier_alt(overlap, data):
    newy = np.zeros(data.shape)
    for nl in prange(data.shape[0]):
        # line = np.dot(overlap, data[nl, :])
        line = overlap.dot(data[nl, :])
        # line = line.sum(1)
        newy[:, nl] = line[:-1]
    return newy

class ArrayMergeWorker(QObject):
    finished = pyqtSignal()
    def __init__(self, otherarrays, arrays, report_to = None):
        super().__init__()
        self.arrays = arrays
        self.otherarrays = otherarrays
        self.reply_here = report_to
    @pyqtSlot()
    def fastrun(self):
        # templock = QMutex()
        xs1, step1,  lowlim1, highlim1, data = self.otherarrays
        lowlim2, highlim2, step2, xs2 = self.arrays
        # newy = np.zeros(data.shape)
        limits1 = np.concatenate([lowlim1,  highlim1[-1:]])
        limits2 = np.concatenate([lowlim2,  highlim2[-1:]])
        overlap = 1.0 - np.abs( ( limits2.reshape((len(limits2), 1)) - limits1.reshape((1, len(limits1))) ) /step1.reshape( (1, len(step1) ) ) )
        temp = np.zeros(overlap.shape)
        temp[np.where(overlap > 0.0)] = overlap[np.where(overlap > 0.0)]
        # oth = len(data)
        # for nl in np.arange(data.shape[1]):
        #     line = temp[:, :-1] * (data[:, nl].reshape((1,  oth)))
        #     line = line.sum(1)
        #     newy[:, nl] = line[:-1]
        # self.results = newy
        # temp = temp[:, :-1].astype(np.float32)
        temp = csc_array(temp[:, :-1].astype(np.float32))
        tempres = helper_multiplier_alt(temp, data.T)
        self.results = tempres
        self.finished.emit()

class CurveAltMergeWorker(QObject):
    finished = pyqtSignal()
    def __init__(self, otherarrays, arrays, const, report_to = None):
        super().__init__()
        self.arrays = arrays
        self.otherarrays = otherarrays
        self.consts = const
        self.reply_here = report_to
    @pyqtSlot()
    def fastrun(self):
        # templock = QMutex()
        xs1, ys1,  step1,  lowlim1, highlim1 = self.otherarrays
        lowlim2, highlim2, step2, xs2, newy = self.arrays
        limits1 = np.concatenate([lowlim1,  highlim1[-1:]])
        limits2 = np.concatenate([lowlim2,  highlim2[-1:]])
        overlap = 1.0 - np.abs( ( limits2.reshape((len(limits2), 1)) - limits1.reshape((1, len(limits1))) ) /step1.reshape( (1, len(step1) ) ) )
        temp = np.zeros(overlap.shape)
        temp[np.where(overlap > 0.0)] = overlap[np.where(overlap > 0.0)]
        # temp[:, :-1] *= ys1.reshape((1,  len(ys1)))
        # newy = temp[:, :-1].sum(1)
        temp = csc_array(temp[:, :-1].astype(np.float32))
        intermediate = temp * ys1.reshape((1,  len(ys1)))
        newy = intermediate.sum(1)
        self.results = np.column_stack([xs2,newy[:-1]])
        # assert(abs(newy.sum() - ys1.sum()) < 1e-8)
        self.finished.emit()

class CurveMergeWorker(QObject):
    finished = pyqtSignal()
    def __init__(self, nbeg, nend, arrays, const, report_to = None):
        super().__init__()
        self.arrays = arrays
        self.consts = const
        self.nbeg = nbeg
        self.nend = nend
        self.reply_here = report_to
    @pyqtSlot()
    def fastrun(self):
        # templock = QMutex()
        lowlim1, lowlim2, highlim1, highlim2, step1, step2, xs2, newy, ys1 = self.arrays
        limits1 = np.concatenate([lowlim1,  highlim1[-1:]])
        limits2 = np.concatenate([lowlim2,  highlim2[-1:]])
        overlap = 1.0 - np.abs( ( limits2.reshape((len(limits2), 1)) - limits1.reshape((1, len(limits1))) ) /step1.reshape( (1, len(step1) ) ) )
        temp = np.zeros(overlap.shape)
        temp[np.where(overlap > 0.0)] = overlap[np.where(overlap > 0.0)]
        temp[:, :-1] *= ys1.reshape((1,  len(ys1)))
        newy = temp[:, :-1].sum(1)
        # temp = csc_array(temp[:, :-1].astype(np.float32))
        # intermediate = temp * ys1.reshape((1,  len(ys1)))
        # newy = intermediate.sum(1)
        self.results = [(self.nbeg, self.nend), np.column_stack([xs2,newy[:-1]])]
        # assert(abs(newy.sum() - ys1.sum()) < 1e-8)
        self.finished.emit()

class CurveMergeWorker_plain(QObject):
    finished = pyqtSignal()
    def __init__(self, nbeg, nend, arrays, const, report_to = None):
        super().__init__()
        self.arrays = arrays
        self.consts = const
        self.nbeg = nbeg
        self.nend = nend
        self.reply_here = report_to
    @pyqtSlot()
    def run(self):
        # templock = QMutex()
        lowlim1, lowlim2, highlim1, highlim2, step1, step2, xs2, newy, ys1 = self.arrays
        lowest_m, highest_m = self.consts
        for n in np.arange(self.nbeg, self.nend):
            m1 = np.argmax(highlim1 > lowlim2[n])
            m2 = np.argmax(lowlim1 > highlim2[n])
            if highlim1[0] > lowlim2[n]:
                m1 = 0
            elif highlim1[-1] < lowlim2[n]:
                m1 = len(highlim1) -1
            elif highlim1[-2] < lowlim2[n]:
                m1 = len(highlim1) -2
            if lowlim1[0] > highlim2[n]:
                m2 = 0
            elif lowlim1[-1] < highlim2[n]:
                m2 = len(lowlim1)-1
            elif lowlim1[-2] < highlim2[n]:
                m2 = len(lowlim1)-2
            # now m1 is the first bin in the source curve that overlaps with the current bin
            # and m2 is the first bin after that that doesn't
            for m in np.arange(m1-2,m2+2):
                if (m > highest_m) or (m < lowest_m):
                    continue
                if (highlim1[m] < lowlim2[n]) or (lowlim1[m] > highlim2[n]):
                    continue
                if lowlim1[m] <= lowlim2[n]:
                    if highlim1[m] > highlim2[n]:
                        # templock.lock()
                        newy[n] += ys1[m]*(highlim2[n]-lowlim2[n])/step1[m]
                        # templock.unlock()
                    else:
                        # templock.lock()
                        newy[n] += ys1[m]*(highlim1[m]-lowlim2[n])/step1[m]
                        # templock.unlock()
                else:
                    if highlim1[m] > highlim2[n]:
                        # templock.lock()
                        newy[n] += ys1[m]*(highlim2[n]-lowlim1[m])/step1[m]
                        # templock.unlock()
                    else:
                        # templock.lock()
                        newy[n] += ys1[m]
                        # templock.unlock()
                # factor = (highlim1[m] - max(lowlim2[n],lowlim1[m])/step1[m]
                # altfactor = (min(highlim2[n],highlim1[m]) - lowlim1[m])/step1[m]
            if not n%16:
                self.reply_here.increment(16)
        # acc_n = np.array(acc_n)
        # print('acc_n: ', acc_n.min(0), acc_n.max(0), acc_n.mean(0))
        # self.reply_here.accept_values([(self.nbeg, self.nend), np.column_stack([xs2,newy])])
        self.results = [(self.nbeg, self.nend), np.column_stack([xs2,newy])]
        self.finished.emit()
    @pyqtSlot()
    def fastrun(self):
        # templock = QMutex()
        lowlim1, lowlim2, highlim1, highlim2, step1, step2, xs2, newy, ys1 = self.arrays
        lowest_m, highest_m = self.consts
        overlap = np.ones([len(xs2), len(ys1)])
        # here we find the correct overlaps
        prescreen = highlim1.reshape((1, len(highlim1))) >= lowlim2.reshape((len(lowlim2), 1))
        prescreen = np.logical_and(prescreen, lowlim1.reshape((1, len(lowlim1))) <= highlim2.reshape((len(highlim2), 1))  )
        # part1 = (highlim2.reshape((len(highlim2), 1)) - lowlim2.reshape((len(lowlim2), 1)) )
        part2 = (highlim1.reshape((1, len(highlim1))) - lowlim2.reshape((len(lowlim2), 1)) ) / step1[:-1].reshape( (1, len(lowlim1) ) )
        part3 = (highlim2.reshape((len(highlim2), 1)) - lowlim1.reshape((1,  len(lowlim1))) )
        # part4 = (highlim1.reshape((1, len(highlim1))) - lowlim1.reshape((1,  len(lowlim1)) ) ) / step1[:-1].reshape( (1, len(lowlim1) ) )
        # temp1 = lowlim1.reshape((1, len(lowlim1))) <= lowlim2.reshape((len(lowlim2), 1))
        # temp2 = highlim1.reshape((1, len(highlim1))) > highlim2.reshape((len(highlim2), 1))
        overlap[np.where(np.logical_not(prescreen))] = 0.0
        crit2 = part2 <= 1.0
        crit3 = part3 <= 1.0
        overlap[np.where(np.logical_and(crit2, prescreen))] = part2[np.where(np.logical_and(crit2, prescreen))]
        overlap[np.where(np.logical_and(crit3, prescreen))] = part3[np.where(np.logical_and(crit3, prescreen))]
        # and there we conclude
        temp = overlap*ys1.reshape((1, len(ys1)))
        newy = temp.sum(1)
        self.results = [(self.nbeg, self.nend), np.column_stack([xs2,newy])]
        # assert(abs(newy.sum() - ys1.sum()) < 1e-8)
        self.finished.emit()


class LoadFileList(QObject):
    def __init__(self, flist, bpp,  cray, tpool = None, pbar = None,  mthreads =1):
        super().__init__()
        self.mutex = QMutex()
        if tpool is not None:
            self.threadpool = tpool
        else:
            # self.threadpool = QThreadPool(self)
            self.threadpool = CustomThreadpool(MAX_THREADS = mthreads)
        self.progbar = pbar
        self.bpp = bpp
        self.cray = cray
        self.counter = 0
        self.total = len(flist)
        self.flist = flist
        # self.progbar.setRange(self.counter,  self.total)
        self.names = []
        self.datalist = []
        self.headlist = []
        self.varloglist = []
        self.finished = False
        self.noproblems = False
        self.runnables = []
        # timer = QTimer(self)
        for name in self.flist:
            rr = LoadAndorFile(name, self.bpp,  self.cray,  report_to = self)
            self.runnables.append(rr)
            # rr.setAutoDelete(False)
            # rr.progress.connect(self.debug_output)
            # rr.result.connect(self.accept_values)
            # rr.finished.connect(self.increment)
    @pyqtSlot(int)
    def debug_output(self, somenum):
        print("Received a progress signal: ",  somenum)
    @pyqtSlot()
    def increment(self):
        self.mutex.lock()
        self.counter += 1
        # self.progbar.setValue(self.counter)
        self.mutex.unlock()
    @pyqtSlot(object)
    def accept_values(self, results):
        self.mutex.lock()
        print("Received values from thread, ",  results[3])
        self.datalist.append(results[0])
        self.headlist.append(results[1])
        self.varloglist.append(results[2])
        self.names.append(results[3])
        self.mutex.unlock()
    def isitover(self):
        self.mutex.lock()
        self.finished = self.counter == self.total
        self.mutex.unlock()
    def runit(self):
        for rr in self.runnables:
            self.threadpool.add_task(rr.run)
        self.threadpool.run()
        for rr in self.runnables:
            self.accept_values(rr.results)
    def postprocess(self):
        temps = []
        shortnames = []
        headers,  logs = [],  []
        nfiles,  times = 0,  []
        header,  vallog = {},  {}
        textheader = ""
        for n, tname in enumerate(self.names):
            nfiles += 1
            temps.append(self.datalist[n])
            headers.append(self.headlist[n])
            logs.append(self.varloglist[n])
            aaa, bbb = os.path.split(tname)
            shortnames.append(bbb)
        for hd in headers:
            hk = hd.keys()
            for k in hk:
                if k in header.keys():
                    header[k].append(hd[k])
                else:
                    header[k] = []
        for lg in logs:
            lk = lg.keys()
            for k in lk:
                if k in vallog.keys():
                    vallog[k] = np.concatenate([vallog[k],  lg[k]])
                else:
                    vallog[k] = lg[k].copy()
        if 'Time' in vallog.keys():
            sequence = np.argsort(vallog['Time'])
            for k in vallog.keys():
                vallog[k] = vallog[k][sequence]
        allnames = ",".join(shortnames)
        start,  end = 0, 60
        segment = allnames[start:end]
        while (len(segment) > 0):
            textheader += "# Filenames: " + segment + "\n"
            start += 60
            end += 60
            segment = allnames[start:end]
        textheader += "# Parameters: Minimum Maximum Mean Stddev\n"
        for hk in header.keys():
            tarr = np.array(header[hk])
            tmin,  tmax,  tmean,  tstd = tarr.min(),  tarr.max(),  tarr.mean(),  tarr.std()
            textheader += " ".join(['#', hk] + [str(round(x,4)) for x in [tmin, tmax, tmean, tstd]] + ['\n'])
        data = np.array(temps)# .sum(0)
        # it is better to return the files separately.
        # we can always sum them up outside of this function
        # or do something else to them first.
        return data,  textheader,  vallog,  shortnames

def simple_read(fname, bpp,  cray):
    fpath, shortname = os.path.split(fname)
    if "." in shortname:
        nameroot = ".".join(shortname.split(".")[:-1])
        extension = shortname.split(".")[-1]
    else:
        nameroot = shortname
        extension = ""
    if 'fits' in extension.lower():
        try:
            data, header = ReadFits(fname)
        except:
            print("Thought that " + fname + " was a FITS file, but ReadFits did not work...")
            data, header = ReadAndor(fname)
    elif 'asc' in extension.lower():
        try:
            data, header = ReadAsc(fname)
        except:
            print("Thought that " + fname + " was an old 2D ASC file, but ReadAsc did not work...")
            data, header = ReadAndor(fname)
    else:
        data, header = ReadAndor(fname)
    tadded_header = load_datheader(os.path.join(fpath, nameroot+".dat"))
    tadded_log = load_datlog(os.path.join(fpath, nameroot+".xas"))
    if cray > 0.0:
        data = RemoveCosmics(data, NStd = cray)
    if bpp is None:
        bpp = np.percentile(data, 10).mean()*0.99
    data -= bpp
    print("Thread has data with shape",  data.shape)
    # self.result.emit((data, tadded_header, tadded_log,  self.fname))  # Return the result of the processing
    # self.reply_here.accept_values([data, tadded_header, tadded_log,  self.fname])
    # self.reply_here.increment()
    return data, tadded_header, tadded_log,  fname

def header_read(fname):
    fpath, shortname = os.path.split(fname)
    if "." in shortname:
        nameroot = ".".join(shortname.split(".")[:-1])
    else:
        nameroot = shortname
    tadded_header, units_for_header = load_datheader(os.path.join(fpath, nameroot+".dat"))
    tadded_log = load_datlog(os.path.join(fpath, nameroot+".xas"))
    return tadded_header, tadded_log

def postprocess(inputlist):
    datalist = inputlist[0]
    headlist = inputlist[1]
    varloglist = inputlist[2]
    names = inputlist[3]
    nfiles,  times = 0,  len(datalist)*[0.0]
    temps = []
    energy = []
    shortnames = []
    headers,  logs = [],  []
    header,  vallog = {},  {}
    textheader = ""
    for n, tname in enumerate(names):
        nfiles += 1
        temps.append(datalist[n])
        headers.append(headlist[n])
        logs.append(varloglist[n])
        aaa, bbb = os.path.split(tname)
        shortnames.append(bbb)
    for hd in headers:
        hk = hd.keys()
        for k in hk:
            if k in header.keys():
                header[k].append(hd[k])
            else:
                header[k] = [hd[k]]
    for lg in logs:
        lk = lg.keys()
        for k in lk:
            if k in vallog.keys():
                vallog[k] = np.concatenate([vallog[k],  lg[k]])
            else:
                vallog[k] = lg[k].copy()
    if 'Time' in vallog.keys():
        sequence = np.argsort(vallog['Time'])
        for k in vallog.keys():
            vallog[k] = vallog[k][sequence]
    allnames = ",".join(shortnames)
    start,  end = 0, 60
    segment = allnames[start:end]
    while (len(segment) > 0):
        textheader += "# Filenames: " + segment + "\n"
        start += 60
        end += 60
        segment = allnames[start:end]
    textheader += "# Parameters: Mean Minimum Maximum Stddev\n"
    for hk in header.keys():
        tarr = np.array(header[hk])
        if "hoton energy" in hk:
            energy.append(header[hk])
        if "Measuring time" in hk:
            times = tarr
        try:
            tmin,  tmax,  tmean,  tstd = tarr.min(),  tarr.max(),  tarr.mean(),  tarr.std()
        except:
            textheader += " ".join(['#', hk] + list(tarr) + ['\n'])
        else:
            textheader += " ".join(['#', hk] + [str(round(x,4)) for x in [tmean, tmin, tmax, tstd]] + ['\n'])
    energy = np.array(energy)
    data = np.array(temps)# .sum(0)
    # it is better to return the files separately.
    # we can always sum them up outside of this function
    # or do something else to them first.
    return data,  textheader,  vallog,  shortnames, (nfiles, times), energy

def load_file(fname, bpp, cray, poly = None):
    fpath, shortname = os.path.split(fname)
    if "." in shortname:
        nameroot = ".".join(shortname.split(".")[:-1])
    else:
        nameroot = shortname
    data, header = ReadAndor(fname)
    tadded_header, units_for_header = load_datheader(os.path.join(fpath, nameroot+".dat"))
    tadded_log = load_datlog(os.path.join(fpath, nameroot+".xas"))
    # data = data[:,lcut:hcut]
    if cray > 0.0:
        data = RemoveCosmics(data, NStd = cray)
    if bpp is None:
        bpp = np.percentile(data, 10).mean()*0.99
    data -= bpp
    return data, tadded_header,  tadded_log

def load_filelist(flist,  bpp,  cray,  poly):
    if len(flist) == 0:
        return None,  None,  None
    else:
        temps = []
        shortnames = []
        headers,  logs = [],  []
        header,  vallog = {},  {}
        textheader = ""
        for tname in flist:
            tdata, theader, tlog = load_file(tname, bpp, cray, poly = poly)
            temps.append(tdata)
            headers.append(theader)
            logs.append(tlog)
            aaa, bbb = os.path.split(tname)
            shortnames.append(bbb)
        for hd in headers:
            hk = hd.keys()
            for k in hk:
                if k in header.keys():
                    header[k].append(hd[k])
                else:
                    header[k] = []
        for lg in logs:
            lk = lg.keys()
            for k in lk:
                if k in vallog.keys():
                    vallog[k] = np.concatenate([vallog[k],  lg[k]])
                else:
                    vallog[k] = lg[k].copy()
        if 'Time' in vallog.keys():
            sequence = np.argsort(vallog['Time'])
            for k in vallog.keys():
                vallog[k] = vallog[k][sequence]
        allnames = ",".join(shortnames)
        start,  end = 0, 60
        segment = allnames[start:end]
        while (len(segment) > 0):
            textheader += "# Filenames: " + segment + "\n"
            start += 60
            end += 60
            segment = allnames[start:end]
        for hk in header.keys():
            tarr = np.array(header[hk])
            tmin,  tmax,  tmean,  tstd = tarr.min(),  tarr.max(),  tarr.mean(),  tarr.std()
            textheader += " ".join(['#', hk] + [str(round(x,4)) for x in [tmin, tmax, tmean, tstd]] + ['\n'])
        data = np.array(temps)# .sum(0)
        # it is better to return the files separately.
        # we can always sum them up outside of this function
        # or do something else to them first.
        return data,  textheader,  vallog


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
            # plot1D_sliders(mcurves, [], fig = self.single_figure, text = 'Results comparison.\n', label_override = ["Energy transfer [eV]", ""], curve_labels = labels[:len(curves)])
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
            # plot2D_sliders([virt_array, map_array], [(yaxis[0],yaxis[-1]), (xaxis[0], xaxis[-1])], fig = self.single_figure, text = "")
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
        # plot1D_sliders(mcurves, [], fig = self.single_figure, text = 'Results comparison.\n', label_override = ["Energy transfer [eV]", ""], curve_labels = labels[:len(curves)])
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
        # plot2D_sliders([virt_array, map_array], [(yaxis[0],yaxis[-1]), (xaxis[0], xaxis[-1])], fig = self.single_figure, text = "")
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
        # self.threadpool = QThreadPool(self)
        # self.threadpool = QThreadPool.globalInstance()
        self.threadpool = CustomThreadpool(MAX_THREADS = max_threads)
        # self.threadpool.setMaxThreadCount(max_threads)
        self.the_object = RixsMeasurement(max_threads = max_threads)
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
        self.tdb_profile = None
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
    def load_last_params(self):
        try:
            source = open(os.path.join(expanduser("~"),'.ADLERcore.txt'), 'r')
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
            source = open(resource_path('Structured_bkg_per_second.dat'), 'r')
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
            self.the_object = RixsMeasurement(max_threads = self.maxthreads)
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
        self.the_object = RixsMeasurement()
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


class DataEntry():
    """
    The basic class for handling lines from CHaOS data headers.
    The main problem is that CHaOS file headers may change,
    and they may contain numbers, text, a mixture of the two,
    or possibly nothing.
    """

    def __init__(self, input = None, label = ""):
        self._array = None
        self._label = ""
        self._pure_numbers = True
        self.data = input
        self.label = label

    def __repr__(self):
        return f"{self.__class__.__name__}(input='{self.string}', label={self.label})"

    def __len__(self):
        if self._array is not None:
            return len(self._array)
        else:
            return 0
    
    def __add__(self, other):
        if self._label == other._label:
            result = copy.deepcopy(self)
            if self._pure_numbers == other._pure_numbers:
                result.label = self.label
                result.data = np.concatenate([self.data, other.data])
            else:
                raise TypeError("DataEntry: adding numbers to non-numbers")
            return result
        raise ValueError("DataEntry: adding two entries with different labels")
    
    def __str__(self):
        return self.string
    
    def __eq__(self, other):
        comp_labels = self.label == other.label
        comp_values = np.allclose(self.data, other.data)
        return comp_labels and comp_values

    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, input: str):
        self._label = input.strip()

    @property
    def string(self):
        return " ".join([str(x) for x in self._array])

    @property
    def data(self):
        return self._array

    @data.setter
    def data(self, input):
        self._array = None
        self._pure_numbers = True
        if input is None:
            self._array = np.array([])
        elif isinstance(input, str):
            toks = input.split()
            numbers = []
            strings = []
            for tk in toks:
                strings.append(tk)
                try:
                    num = float(tk)
                except ValueError:
                    self._pure_numbers = False
                else:
                    numbers.append(num)
            if self._pure_numbers:
                self._array = np.array(numbers)
            else:
                self._array = np.array(strings)
        else:
            try:
                len(input)
            except AttributeError:
                try:
                    num = float(input)
                except ValueError:
                    self._pure_numbers = False
                    self._array = np.array([input])
                else:
                    self._pure_numbers = True
                    self._array = np.array([num])
            else:
                try:
                    temp = [float(x) for x in input]
                except ValueError:
                    self._array = np.array([str(x) for x in input])
                    self._pure_numbers = False
                else:
                    self._array = np.array(temp)
                    self._pure_numbers = True
                    

class DataGroup(defaultdict, yaml.YAMLObject):
    
    _last_group = 1

    # yaml_tag = '!AdlerDataGroup'

    def __init__(self, default_class = DataEntry, label = None, elements = None):
        self.def_class = default_class
        super(DataGroup, self).__init__(self.def_class)

        if label is None:
            self.label = "UnnamedGroup"+str(DataGroup._last_group)
            DataGroup._last_group += 1
        else:
            self.label = label
        
        if elements is not None:
            for line in elements:
                key, value = line[0], line[1]
                self.__setitem__(key, value)
    
    def __repr__(self):
        temp = []
        for key in self.keys():
            temp.append((key, self.__getitem__(key)))
        return "%s(default_class=%r, label=%r, elements=%r)" % (self.__class__.__name__, self.def_class, self.label, temp)
    

class RixsMeasurement():
    def __init__(self, filenames = [], max_threads = 1):
        self.date_formatstring = "dd-MM-yyyy"
        self.time_formatstring = "hh:mm:ss"
        self.sourcename = ""
        self.shortsource = ""
        self.data = []
        self.headers = []
        self.logs = []
        self.history = []
        self.hcounter = 1
        self.data_files = [] # detector images loaded
        self.header_files = [] # .DAT files loaded
        self.log_files = [] # .XAS files loaded
        self.start_points = [] # this will be a (QDate, QTime) tuple
        self.tdb_profile = None
        self.calibration = []
        self.calibration_source = []
        self.calibration_dates = []
        self.calibration_params = []
        self.reverse_calibration_params = []
        self.individual_profiles_channels = []
        self.individual_profiles_absEnergy = []
        self.individual_profiles_eV = []
        self.individual_profiles_offsets = []
        self.profile_channels = np.array([])
        self.profile_absEnergy = np.array([])
        self.profile_eV = np.array([])
        self.background_channels = []
        self.background_absEnergy = []
        self.background_eV = []
        self.merged_data = []
        self.fitting_params_absEnergy = defaultdict(lambda: -1)
        self.fitting_params_channels = defaultdict(lambda: -1)
        self.fitting_params_eV = defaultdict(lambda: -1)
        self.fitted_peak_channels = []
        self.fitting_textstring_channels = ""
        self.fitted_peak_absEnergy = []
        self.fitting_textstring_absEnergy = ""
        self.fitted_peak_eV = []
        self.fitting_textstring_eV = ""
        self.textheader = ""
        self.the_log = defaultdict(lambda: np.array([]))
        self.the_header = DataGroup(default_class= DataGroup, label= "Header")
        self.nfiles = 0
        self.times = []
        self.shortnames = []
        self.energy = -1.0
        self.energies = []
        self.nominal_elastic_line_position = -999.0
        self.mev_per_channel = -1.0
        self.arbitrary_range = np.array([-8192.0, 8192.0])
        self.arbitrary_unit = -1
        self.maxthreads = max_threads
        self.active = True # only a temporary setting in the GUI
    def summariseCrucialParts(self):
        pdict = {}
        pdict['calibration'] = []
        for line in self.calibration:
            pdict['calibration'].append(" ".join([str(x) for x in line]))
        pdict['calibration_source'] = self.calibration_source
        pdict['nominal_eline'] = str(self.nominal_elastic_line_position)
        pdict['calibration_dates'] = self.calibration_dates
        pdict['calibration_params'] = [self.calibration_params]
        pdict['fitting_params_absEnergy'] = [" ".join([str(k), str(self.fitting_params_absEnergy[k])]) for k in self.fitting_params_absEnergy]
        pdict['fitting_params_channels'] = [" ".join([str(k), str(self.fitting_params_channels[k])]) for k in self.fitting_params_channels]
        pdict['fitting_params_eV'] = [" ".join([str(k), str(self.fitting_params_eV[k])]) for k in self.fitting_params_eV]
        pdict['times'] = " ".join([str(x) for x in self.times])
        pdict['energies'] = " ".join([str(x) for x in self.energies])
        pdict['energy'] = str(self.energy)
        pdict['temperature'] = str(self.the_log['TEMP1'].mean())
        pdict['arm_theta'] = str(self.the_log['THETA'].mean())
        pdict['Q'] = str(self.the_header['Q'])
        return pdict
    def assignCrucialParts(self, pdict):
        temp = []
        for line in pdict['calibration']:
            temp.append([float(x) for x in line.split()])
        self.calibration = np.array(temp)
        self.calibration_source = pdict['calibration_source']
        self.calibration_dates = pdict['calibration_dates']
        self.nominal_elastic_line_position = pdict['nominal_eline']
        self.calibration_params = [float(x) for x in pdict['calibration_params'][0].split()]
        for line in pdict['fitting_params_channels']:
            toks = line.split()
            if len(toks) == 2:
                key, val = toks[0].strip(' \n'),  toks[1].strip(' \n')
                self.fitting_params_channels[key] = float(val)
        for line in pdict['fitting_params_absEnergy']:
            toks = line.split()
            if len(toks) == 2:
                key, val = toks[0].strip(' \n'),  toks[1].strip(' \n')
                self.fitting_params_absEnergy[key] = float(val)
        for line in pdict['fitting_params_eV']:
            toks = line.split()
            if len(toks) == 2:
                key, val = toks[0].strip(' \n'),  toks[1].strip(' \n')
                self.fitting_params_eV[key] = float(val)
        self.times = np.array([float(x) for x in pdict['times']])
        self.energies = np.array([float(x.strip('[] \n')) for x in pdict['energies']])
        self.energy = float(pdict['energy'][0])
        # pdict['temperature'] = str(self.the_log['TEMP1'].mean())
        # pdict['arm_theta'] = str(self.the_log['THETA'].mean())
        # pdict['Q'] = str(self.the_header['Q'])
    def write_ADLER(self, fname):
        """
        This outputs the spectra in the old text format.
        There is one file per spectrum, with different suffixed in the file name.
        """
        parlist, pardict = [], {}
        suff = ['_channels', '_absEnergy',  '_eV']
        for nn,  fitdict in enumerate([self.fitting_params_channels, self.fitting_params_absEnergy, self.fitting_params_eV]):
            suffix = suff[nn]
            for kk in fitdict.keys():
                tkey = str(kk)+suffix
                if not tkey in parlist:
                    parlist.append(tkey)
                pardict[tkey] = fitdict[kk]
        newheader = self.textheader.split('\n')
        newheader += self.history
        for n in range(len(self.calibration)):
            newheader += ["Calibration: " + ",".join([str(x) for x in [self.calibration[n], self.calibration_source[n], self.calibration_dates[n]]])]
        logvalnames = [str(x) for x in self.the_log.keys()]
        if len(self.profile_channels) > 0:
            WriteProfile(fname+"_1D.txt", self.profile_channels,
                            header = newheader, params = [parlist,  pardict], 
                            varlog = [logvalnames,  self.the_log])
        if len(self.profile_absEnergy) > 0:
            WriteProfile(fname+"_1D_absEnergy.txt", self.profile_absEnergy,
                            header = newheader, params = [parlist,  pardict], 
                            varlog = [logvalnames,  self.the_log])
        if len(self.profile_eV) > 0:
            WriteProfile(fname+"_1D_deltaE.txt", self.profile_eV,
                            header = newheader, params = [parlist,  pardict], 
                            varlog = [logvalnames,  self.the_log])
    def write_extended_ADLER(self, fname):
        target = open(fname, 'w')
        header = self.the_header
        varlog = self.the_log
        history = self.history
        crucial = self.summariseCrucialParts()
        if header is not None:
            for hk in header.keys():
                element = header[hk]
                tkey = str(hk)
                try:
                    element.keys()
                except AttributeError:
                    newln = tkey + " $ " + element.string
                    target.write(' '.join(['#HEADER#',newln.strip().strip('\n'),'\n']))
                else:
                    for gk in element.keys():
                        item = element[gk]
                        innerkey = str(gk)
                        newln = tkey + " # " + innerkey + " $ " + item.string
                        target.write(' '.join(['#HEADER#',newln.strip().strip('\n'),'\n']))
        if crucial is not None:
            for hk in crucial.keys():
                tkey = str(hk)
                if 'calibration' in tkey:
                    for entry in crucial[hk]:
                        try:
                            entry.split()
                        except:
                            newln = tkey + " $ " + " ".join([str(x) for x in entry if str(x).isprintable()])
                        else:
                            newln = tkey + " $ " + "".join([str(x) for x in entry if str(x).isprintable()])
                        target.write(' '.join(['#CRUCIAL#',newln.strip().strip('\n'),'\n']))
                elif 'fitting_params' in tkey:
                    for entry in crucial[hk]:
                        try:
                            entry.split()
                        except:
                            newln = tkey + " $ " + " ".join([str(x) for x in entry if str(x).isprintable()])
                        else:
                            newln = tkey + " $ " + "".join([str(x) for x in entry if str(x).isprintable()])
                        target.write(' '.join(['#CRUCIAL#',newln.strip().strip('\n'),'\n']))
                else:
                    try:
                        crucial[hk].split()
                    except:
                        newln = tkey + " $ " + " ".join([str(x) for x in crucial[hk] if str(x).isprintable()])
                    else:
                        newln = tkey + " $ " + "".join([str(x) for x in crucial[hk] if str(x).isprintable()])
                    target.write(' '.join(['#CRUCIAL#',newln.strip().strip('\n'),'\n']))
        if varlog is not None:
            keys, vals = [str(x) for x in varlog.keys()], varlog
            if len(keys)>0:
                target.write("#LOG# VARIABLES TIMELOG\n")
                target.write(" ".join(['#LOG#'] + keys + ['\n']))
                for ln in range(len(vals[keys[0]])):
                    target.write(" ".join(['#LOG#'] + [str(round(vals[k][ln], 5)) for k in keys] + ['\n']))
        if history is not None:
            target.write("# ADLER PROCESSING HISTORY\n")
            for line in history:
                target.write("#HISTORY#" + str(line[0]) + "$" + line[1]  + '\n')
        target.write("#PROFILE#\n")
        for n, val in enumerate(self.profile_channels):
            target.write(",".join([str(x) for x in [val[0], val[1]]]) + '\n')
        target.close()
    def read_extended_ADLER(self, fname):
        header, log, crucial, history = [], [], [], []
        profile = []
        others = defaultdict(lambda: [])
        self.sourcename = fname
        self.shortsource = os.path.split(fname)[1]
        target = open(fname, 'r')
        for line in target:
            if len(line) <2:
                continue
            toks = line.split('#')
            if len(toks) > 1:
                group = toks[1].strip().lower()
                entry = "#".join(toks[2:])
                if group in ['header']:
                    header.append(entry)
                elif group in ['crucial']:
                    crucial.append(entry)
                elif group in ['log']:
                    log.append(entry)
                elif group in ['history']:
                    history.append(entry)
                elif group == '':
                    continue
                elif group in ['profile']:
                    continue
                else:
                    others[group].append(entry.strip(' '))
            else:
                temp = line.split(',')
                if len(temp) < 2:
                    temp = line.split()
                    if len(temp) < 2:
                        continue
                profile.append([float(x) for x in temp])
        target.close()
        # the file has been read but so for we only sorted the lines
        self.readLog(log)
        self.readHeader(header)
        self.readCrucial(crucial)
        self.readHistory(history)
        self.profile_channels = np.array(profile)
        self.reconstruct_profiles()
    def readHeader(self, lines):
        toplevel = DataGroup(default_class= DataGroup, label='Header')
        toplevel['other'] = DataGroup(label= 'other')
        for entry in lines:
            if '#' in entry:
                toks = entry.split('#')
                if len(toks) > 1:
                    category, subentry = toks[0], '#'.join(toks[1:])
                    if not category in toplevel.keys():
                        toplevel[category] = DataGroup(label = category)
                    relevantGroup = toplevel[category]
            else:
                relevantGroup = toplevel['other']
                subentry = entry
            key,  value = subentry.split('$')[0].strip(' '),  subentry.split('$')[1].strip(' ')
            relevantGroup[key] = DataEntry(value, key)
        self.the_header = toplevel
    def readHistory(self, history):
        temp_list = []
        for entry in history:
            temp_list.append(entry.strip())
        self.history = temp_list
    def readCrucial(self, crucial):
        temp_dict = defaultdict(lambda: ['-1'])
        for entry in crucial:
            key,  value = entry.split('$')[0].strip(' '),  entry.split('$')[1].strip(' ')
            if not key in temp_dict.keys():
                temp_dict[key] = [value]
            else:
                temp_dict[key].append(value)
        self.assignCrucialParts(temp_dict)
    def readLog(self, log):
        lognames, logvals = [], []
        for entry in log:
            values = entry.split()
            if len(values) < 5:
                continue
            else:
                try:
                    float(values[0])
                except:
                    lognames = values
                else:
                    logvals.append([float(x) for x in values])
        logvals = np.array(logvals)
        self.the_log = defaultdict(lambda: np.array([]))
        for n,  key in enumerate(lognames):
            self.the_log[key] = logvals[:, n]
    def write_yaml(self, fname, full_output = False,  compressed = False):
        """
        For those who like text files, and still need some structure in the file,
        there is YAML format.
        """
        data1, data2, data3 = {}, {}, {}
        if full_output:
            if compressed:
                data3['data'] = []
                for n in self.data:
                    data3['data'].append(self.compress_array(n))
                data3['merged_data'] = self.compress_array(self.merged_data)
            else:
                data3['data'] = self.data
                data3['merged_data'] = self.merged_data
        else:
            data3['data'] = []
            data3['merged_data'] = []
        data1['the_header'] = self.the_header
        data2['logs'] = self.logs
        data2['headers'] = self.headers
        data1['history'] = self.history
        data1['hcounter'] = self.hcounter
        data1['data_files'] = self.data_files
        data1['header_files'] = self.header_files
        data1['log_files'] = self.log_files
        temp = self.start_points
        try:
            data1['start_points'] = (temp[0].tostring(self.date_formatstring), temp[1].tostring(self.time_formatstring))
        except:
            data1['start_points'] = (None, None)
        data3['tdb_profile'] = self.tdb_profile
        data1['calibration'] = self.calibration
        data1['calibration_source'] = self.calibration_source
        data1['calibration_dates'] = self.calibration_dates
        data1['calibration_params'] = self.calibration_params
        data1['reverse_calibration_params'] = self.reverse_calibration_params
        data3['individual_profiles_channels'] = self.individual_profiles_channels
        data3['individual_profiles_absEnergy'] = self.individual_profiles_absEnergy
        data3['individual_profiles_eV'] = self.individual_profiles_eV
        data3['individual_profiles_offsets'] = self.individual_profiles_offsets
        data3['profile_channels'] = self.profile_channels
        data3['profile_absEnergy'] = self.profile_absEnergy
        data3['profile_eV'] = self.profile_eV
        data3['background_channels'] = self.background_channels
        data3['background_absEnergy'] = self.background_absEnergy
        data3['background_eV'] = self.background_eV
        data1['fitting_params_absEnergy'] = [(str(k), float(self.fitting_params_absEnergy[k])) for k in self.fitting_params_absEnergy]
        data1['fitting_params_channels'] = [(str(k), float(self.fitting_params_channels[k])) for k in self.fitting_params_channels]
        data1['fitting_params_eV'] = [(str(k), float(self.fitting_params_eV[k])) for k in self.fitting_params_eV]
        data3['fitted_peak_channels'] = self.fitted_peak_channels
        data1['fitting_textstring_channels'] = self.fitting_textstring_channels
        data3['fitted_peak_absEnergy'] = self.fitted_peak_absEnergy
        data1['fitting_textstring_absEnergy'] = self.fitting_textstring_absEnergy
        data3['fitted_peak_eV'] = self.fitted_peak_eV
        data1['fitting_textstring_eV'] = self.fitting_textstring_eV
        data2['textheader'] = self.textheader
        data2['the_log'] = self.the_log
        data1['nfiles'] = self.nfiles
        data1['times'] = [float(x) for x in self.times]
        data1['shortnames'] = self.shortnames
        try:
            data1['energy'] = float(self.energy)
        except:
            data1['energy'] = self.energy
        data1['nominal_elastic_line_position'] = str(self.nominal_elastic_line_position)
        topdata = {'1. Readable' : data1,  '2. Logs' : data2,  '3. Profiles' : data3}
        # and now we write out
        stream = open(fname + '.yaml', 'w')
        yaml.dump(topdata, stream)
        stream.close()
    def read_yaml(self, fname):
        stream = open(fname, 'r')
        topdata = yaml.load(stream, Loader)
        stream.close()
        self.sourcename = fname
        self.shortsource = os.path.split(fname)[1]
        # and now the reverse
        data1 = topdata['1. Readable']
        data2 = topdata['2. Logs']
        data3 = topdata['3. Profiles']
        for n in data3['data']:
            if len(n) == 3:
                a, b, c = n
                self.data.append(self.decompress_array(a, b, c))
            else:
                self.data.append(n)
        if len(data3['merged_data'] ) == 3:
            a, b, c = data3['merged_data']
            self.merged_data = self.decompress_array(a, b, c)
        else:
            self.merged_data = data3['merged_data'] 
        self.the_header = data1['the_header']
        self.logs = data2['logs']
        self.headers = data2['headers']
        self.history =data1['history']
        self.hcounter =data1['hcounter']
        self.data_files =data1['data_files']
        self.header_files =data1['header_files']
        self.log_files =data1['log_files']
        temp =data1['start_points']
        try:
            self.start_points = (QDate.fromString(temp[0], self.date_formatstring), QTime.fromString(temp[1], self.time.formatstring))
        except:
            self.start_points = (None, None)
        self.tdb_profile =data3['tdb_profile'] 
        self.calibration =data1['calibration']
        self.calibration_source =data1['calibration_source']
        self.calibration_dates =data1['calibration_dates']
        self.calibration_params =data1['calibration_params']
        self.reverse_calibration_params =data1['reverse_calibration_params']
        self.individual_profiles_channels =data3['individual_profiles_channels'] 
        self.individual_profiles_absEnergy =data3['individual_profiles_absEnergy'] 
        self.individual_profiles_eV =data3['individual_profiles_eV'] 
        self.individual_profiles_offsets =data3['individual_profiles_offsets'] 
        self.profile_channels =data3['profile_channels'] 
        self.profile_absEnergy =data3['profile_absEnergy'] 
        self.profile_eV =data3['profile_eV'] 
        self.background_channels =data3['background_channels'] 
        self.background_absEnergy =data3['background_absEnergy'] 
        self.background_eV =data3['background_eV']
        temp1 = data1['fitting_params_absEnergy']
        temp2 = data1['fitting_params_channels'] 
        temp3 = data1['fitting_params_eV'] 
        self.fitting_params_absEnergy = defaultdict(lambda: -1)
        self.fitting_params_channels =defaultdict(lambda: -1)
        self.fitting_params_eV =defaultdict(lambda: -1)
        for tup in temp1:
            key,  val = tup[0], tup[1]
            self.fitting_params_absEnergy[key] = val
        for tup in temp2:
            key,  val = tup[0], tup[1]
            self.fitting_params_channels[key] = val
        for tup in temp3:
            key,  val = tup[0], tup[1]
            self.fitting_params_eV[key] = val
        self.fitted_peak_channels =data3['fitted_peak_channels'] 
        self.fitting_textstring_channels =data1['fitting_textstring_channels'] 
        self.fitted_peak_absEnergy =data3['fitted_peak_absEnergy']
        self.fitting_textstring_absEnergy =data1['fitting_textstring_absEnergy'] 
        self.fitted_peak_eV = data3['fitted_peak_eV'] 
        self.fitting_textstring_eV =data1['fitting_textstring_eV'] 
        self.textheader =data2['textheader'] 
        self.the_log =data2['the_log'] 
        self.nfiles =data1['nfiles'] 
        self.times = np.array(data1['times'] )
        self.shortnames =data1['shortnames'] 
        self.energy =data1['energy'] 
        self.nominal_elastic_line_position =data1['nominal_elastic_line_position'] 
        self.reconstruct_profiles()
    def write_hdf5(self, fname, full_output = False):
        """
        This will have to be improved.
        The idea is to store all the relevant information in HDF5 format.
        As for matching the NeXus specifications, this may take some time...
        """
        f = h5py.File(fname, "w")  # create the HDF5 NeXus file
        mainentry = f.create_group(u"RIXSscan")
        header = mainentry.create_group(u"header")
        history = mainentry.create_group(u"history")
        for hentry in self.history:
            history.attrs[str(hentry[0])] = str(hentry[1])
        # nxdata.attrs["NX_class"] = u"NXdata"
        # nxdata.attrs[u"signal"] = u"counts"
        # nxdata.attrs[u"axes"] = u"two_theta"
        # nxdata.attrs[u"two_theta_indices"] = [0,]
        logdata = mainentry.create_group(u"log")
        for kk in self.the_log.keys():
            tkey = str(kk)
            temp = logdata.create_dataset(tkey, data=self.the_log[kk])
        # tth.attrs[u"units"] = u"degrees"
        rixsprofile = mainentry.create_group('profile')
        units = ['counts', 'channels', 'eV', 'eV']
        labels = ['signal', 'detector position', 'photon energy', 'energy transfer']
        profs = [self.profile_channels[:, 1], self.profile_channels[:, 0], self.profile_absEnergy[:, 0], self.profile_eV[:, 0]]
        for n in range(len(units)):
            temp = rixsprofile.create_dataset(labels[n], data = profs[n])
            temp.attrs['units'] = units[n]
        f.close()   # be CERTAIN to close the file
    def write_NeXus(self, fname):
        """
        This will have to be improved.
        The idea is to store all the relevant information in HDF5 format.
        As for matching the NeXus specifications, this may take some time...
        """
        f = h5py.File("writer_1_3.hdf5", "w")  # create the HDF5 NeXus file
        nxentry = f.create_group(u"Scan")
        nxentry.attrs[u"NX_class"] = u"NXentry"
        nxdata = nxentry.create_group(u"data")
        nxdata.attrs["NX_class"] = u"NXdata"
        nxdata.attrs[u"signal"] = u"counts"
        nxdata.attrs[u"axes"] = u"two_theta"
        nxdata.attrs[u"two_theta_indices"] = [0,]
        tth = nxdata.create_dataset(u"two_theta", data=tthData)
        tth.attrs[u"units"] = u"degrees"
        counts = nxdata.create_dataset(u"counts", data=countsData)
        counts.attrs[u"units"] = u"counts"
        f.close()   # be CERTAIN to close the file
    def compress_array(self, inp_array):
        data = inp_array.copy()
        shape = data.shape
        dtype = data.dtype.str
        bdata = data.tobytes()
        compdata = gzip.compress(bdata)
        return compdata, shape, dtype
    def decompress_array(self, compdata, shape, dtype):
        bdata = gzip.decompress(compdata)
        linedata = np.frombuffer(bdata, dtype=dtype)
        truedata = linedata.reshape(shape)
        return truedata
    def assignCalibrationFromFiles(self, calmeas):
        positions, energies = [], []
        names = []
        for cm in calmeas:
            energies.append(cm.energy)
            positions.append(cm.fitting_params_channels['centre'])
            names.append(cm.data_files)
        calibration = np.column_stack([energies,  positions])
        sorting = np.argsort(energies)
        calibration = calibration[sorting]
        self.calibration = calibration
        self.calibration_source = names
    def assignCalibration(self, caldata):
        positions, energies = [], []
        names, dates = [], []
        count = 0
        for cm in caldata:
            try:
                pos = float(cm[1])
            except:
                pos = -111
            try:
                ene = float(cm[2])
            except:
                ene = -111.0
            name = cm[0]
            date = cm[3]
            if ene > 0.0 and pos > 0.0:
                energies.append(ene)
                positions.append(pos)
                names.append(name)
                dates.append(date)
                count += 1
        if count > 0:
            calibration = np.column_stack([energies,  positions])
            sorting = np.argsort(energies)
            calibration = calibration[sorting]
            self.calibration = calibration
            self.calibration_source = names
            self.calibration_dates = dates
        else:
            self.calibration = []
            self.calibration_source = []
            self.calibration_dates = []
    def reconstruct_profiles(self):
        self.reconstructPeak()
        if not len(self.calibration_params) >1:
            if len(self.calibration) > 1:
                self.makeEnergyCalibration()
                self.makeEnergyProfile()
            else:
                print("not enough information to reconstruct absolute energy in ",  self.sourcename)
                return None
        else:
            self.makeEnergyProfile()
        try:
            float(self.fitting_params_channels['centre'])
        except:
            print("not enough information to reconstruct the energy transfer axis in ", self.sourcename)
        else:
            self.makePeakInEnergy()
    def addHistory(self, line):
        self.history.append([self.hcounter, line])
        self.hcounter += 1
    def loadFiles(self, filenames = []):
        if len(filenames) > 0:
            for fname in filenames:
                data, header, log, names = self.plainRead(fname)
                self.data.append(data)
                self.headers.append(header)
                self.logs.append(log)
                self.data_files.append(names[0])
                self.header_files.append(names[1])
                self.log_files.append(names[2])
                self.individual_profiles_channels.append([])
                self.individual_profiles_absEnergy.append([])
                self.individual_profiles_eV.append([])
    def plainRead(self, fname):
        fpath, shortname = os.path.split(fname)
        if "." in shortname:
            nameroot = ".".join(shortname.split(".")[:-1])
            extension = shortname.split(".")[-1]
        else:
            nameroot = shortname
            extension = ""
        if 'fits' in extension.lower():
            try:
                data, header = ReadFits(fname)
            except:
                print("Thought that " + fname + " was a FITS file, but ReadFits did not work...")
                data, header = ReadAndor(fname)
        elif 'asc' in extension.lower():
            try:
                data, header = ReadAsc(fname)
            except:
                print("Thought that " + fname + " was an old 2D ASC file, but ReadAsc did not work...")
                data, header = ReadAndor(fname)
        else:
            data, header = ReadAndor(fname)
        datname = fname
        headname = os.path.join(fpath, nameroot+".dat")
        logname = os.path.join(fpath, nameroot+".xas")
        tadded_header, units_for_header = load_datheader(headname)
        tadded_log = load_datlog(logname)
        return data, tadded_header, tadded_log, [datname, headname, logname]
    def set_integration_range(self, numrange, unitnum):
        self.arbitrary_unit = unitnum
        self.arbitrary_range = numrange
    def integrate_range(self):
        if self.arbitrary_unit == 0:
            prof = self.profile_absEnergy
        elif self.arbitrary_unit == 1:
            prof = self.profile_eV
        else:
            prof = self.profile_channels
        if len(prof) < 1:
            return -1.0
        lowlim, highlim = self.arbitrary_range
        crit = np.where(np.logical_and(prof[:, 0] >= lowlim, prof[:, 0] <= highlim))
        if len(prof[:, 1][crit]) > 0:
            value = prof[:, 1][crit].sum()
        else:
            value = 1.0
        return value
    def norm_number(self,  flags = [], unit = 2):
        the_norm = 1.0
        norm_values = []
        # Currently the valid options are: ["Integrated ring current",  "Total counts", "Time", "Number of scans",
        #                             "Peak area", "Peak position", "Peak width", "Arbitrary range"]
        # step 1:  integrated current
        timestamps = self.the_log['RelTime']
        current = self.the_log['RING']*1e-3
        tsteps = timestamps[1:] - timestamps[:-1]
        tsteps[np.where(np.abs(tsteps) > 5*np.percentile(tsteps, 90))] = 0.0
        IRC = np.abs(tsteps * current[:len(tsteps)])
        norm_values.append(float(IRC.sum()))
        # step 2: total counts
        norm_values.append(float(np.abs(self.profile_channels[:, 1]).sum()))
        # step 3: time
        norm_values.append(self.times.sum())
        # step 4: Number of scans
        norm_values.append(len(self.times))
        # step 5: peak area
        if unit == 0:
            fit = self.fitting_params_absEnergy
        elif unit == 1:
            fit = self.fitting_params_eV
        else:
            fit = self.fitting_params_channels
        norm_values.append(abs(fit['area']))
        # step 6: peak position
        norm_values.append(abs(fit['centre']))
        # step 7: peak position
        norm_values.append(abs(fit['fwhm']))
        # step 8: arbitrary range
        norm_values.append(self.integrate_range())
        for nn, flag in enumerate(flags):
            if flag:
                the_norm *= norm_values[nn]
        return the_norm
    def postprocess(self):
        datalist = self.data
        headlist = self.headers
        varloglist = self.logs
        names = self.data_files
        nfiles,  times = 0,  len(datalist)*[0.0]
        temps = []
        energy = []
        shortnames = []
        headers,  logs = [],  []
        header,  vallog = DataGroup(default_class= DataGroup, label= "Header"), defaultdict(lambda: np.array([]))
        textheader = ""
        for n, tname in enumerate(names):
            nfiles += 1
            temps.append(datalist[n])
            headers.append(headlist[n])
            logs.append(varloglist[n])
            aaa, bbb = os.path.split(tname)
            shortnames.append(bbb)
        for hd in headers:
            hk = hd.keys()
            sdate, stime = None, None
            for tk in hk:
                k = str(tk).strip(" # # ")
                if k in header.keys():
                    header[k].append(hd[tk])
                else:
                    header[k] = [hd[tk]]
                if "Date" in str(k):
                    sdate = QDate.fromString(hd[tk], 'yyyy-MM-dd')
                if "Time at start" in str(k):
                    stime = QTime.fromString(hd[tk], 'hh:mm:ss')
            self.start_points.append((sdate, stime))
        for lg in logs:
            lk = lg.keys()
            for k in lk:
                if k in vallog.keys():
                    vallog[k] = np.concatenate([vallog[k],  lg[k]])
                else:
                    vallog[k] = lg[k].copy()
        if 'Time' in vallog.keys():
            sequence = np.argsort(vallog['Time'])
            for k in vallog.keys():
                vallog[k] = vallog[k][sequence]
        allnames = ",".join(shortnames)
        start,  end = 0, 60
        segment = allnames[start:end]
        while (len(segment) > 0):
            textheader += "# Filenames: " + segment + "\n"
            start += 60
            end += 60
            segment = allnames[start:end]
        textheader += "# Parameters: Mean Minimum Maximum Stddev\n"
        for hk in header.keys():
            tarr = np.array(header[hk])
            if "hoton energy" in hk:
                energy.append(header[hk])
            if "Measuring time" in hk:
                times = tarr
            try:
                tmin,  tmax,  tmean,  tstd = tarr.min(),  tarr.max(),  tarr.mean(),  tarr.std()
            except:
                textheader += " ".join(['#', hk] + list(tarr) + ['\n'])
            else:
                textheader += " ".join(['#', hk] + [str(round(x,4)) for x in [tmean, tmin, tmax, tstd]] + ['\n'])
        energy = np.array(energy)
        data = np.array(temps)# .sum(0)
        self.data = data
        self.textheader = textheader
        self.the_log = vallog
        self.the_header = header
        self.nfiles = nfiles
        self.times = times
        self.energies = energy
        self.energy = energy.mean()
        self.shortnames = shortnames
    def removeCrays(self, cray):
        for n, data in enumerate(self.data):
            self.data[n] = RemoveCosmics(data, NStd=cray)
        self.addHistory("RemoveCosmics, NStd="+str(cray))
    def subtractBackground(self, bpp):
        for n, data in enumerate(self.data):
            if bpp is None:
                bpp = np.percentile(data, 10).mean()*0.99
            self.data[n] -= bpp
        self.addHistory("SubtractBackground, bpp="+str(bpp))
    def subtractDarkCurrent(self, scaling = 1.0):
        for n, data in enumerate(self.data):
            det_width = data.shape[1]
            det_length = data.shape[0]
            scalefac = 1.0/det_width * scaling
            if (self.tdb_profile is not None) and scaling > 0.0 :
                temp = self.tdb_profile[:, 1].copy()
                temp *= scalefac
                if not len(temp) == data.shape[1]:
                    continue
                else:
                    temp2 = temp*self.times[n]
                    temp2.reshape((1, det_width))
                    self.data[n] -= temp2
            else:
                continue
        self.addHistory("SubtractDarkCurrent, scaling="+str(scaling))
    def makeSeparateProfiles(self, redfactor = 1, cuts = None):
        for n, data in enumerate(self.data):
            if cuts is None:
                cuts = [0, data.shape[1], 0, data.shape[0]]
            self.individual_profiles_channels[n] = make_profile(data, reduction = redfactor, limits = cuts)
        self.addHistory("makeSeparateProfiles, redfactor="+str(redfactor) + ", cuts=" + str(cuts))
    def mergeArrays(self, offsets = None):
        if offsets is None:
            offsets = self.offsets.copy()
        if len(self.data) == 1:
            self.merged_data = self.data[0].copy()
            return None
        the_object = MergeManyArrays(self.data, offsets, mthreads = self.maxthreads)
        the_object.runit()
        self.merged_data = the_object.postprocess()
    def mergeProfiles(self, offsets = None):
        if offsets is None:
            offsets = self.offsets.copy()
        temp_profs = []
        if len(self.individual_profiles_channels) == 1:
            self.profile_channels = self.individual_profiles_channels[0].copy()
            return None
        for n, prof in enumerate(self.individual_profiles_channels):
            temp = prof.copy()
            temp[:, 0] -= offsets[n]
            temp_profs.append(temp)
        target = temp_profs[0].copy()
        # target[:, 0] = 0.0
        the_object = MergeManyCurves(temp_profs[1:], target, mthreads = self.maxthreads)
        the_object.runit()
        self.profile_channels = the_object.postprocess()
        self.addHistory("mergeProfiles, offsets="+str(offsets))
    def makeEnergyCalibration(self, fitorder=2):
        if len(self.calibration) < 2:
            return None
        order = min(len(self.calibration), fitorder)
        pfit, pcov, infodict, errmsg, success = leastsq(fit_polynomial, np.zeros(order), args = (self.calibration[:, 0:2], ),
                                                                                    full_output=1)
        pfit2, pcov2, infodict2, errmsg2, success2 = leastsq(fit_polynomial, np.zeros(order), args = (self.calibration[:, 2::-1], ),
                                                                                    full_output=1)
        self.calibration_params = pfit2
        self.reverse_calibration_params = pfit
        temp = polynomial(self.calibration_params, [500, 1500])
        mev_per_channel = temp.max() - temp.min()
        self.mev_per_channel = mev_per_channel
    def eline(self):
        if len(self.reverse_calibration_params) >0:
            self.nominal_elastic_line_position = polynomial(self.reverse_calibration_params, [self.energy])
            return self.nominal_elastic_line_position
        else:
            return None
    def reconstructPeak(self):
        fit = self.fitting_params_channels
        prof = self.profile_channels.copy()
        centre, centre_err = fit['centre'],  fit['centre_error']
        fwhm, fwhm_err = abs(fit['fwhm']), fit['fwhm_error']
        maxval, maxval_err = fit['maxval'], fit['maxval_error']
        area, area_err = fit['area'], fit['area_error']
        fitstring = ' '.join(['FWHM =', str(abs(round(fwhm,3))), ' +/- ', str(abs(round(fwhm_err,3))),'\n',
                        'centre =', str(round(centre,3)), ' +/- ', str(abs(round(centre_err,3))),'\n',
                        'area =', str(abs(round(area,3))), ' +/- ', str(abs(round(area_err,3)))])# , '\n', 
                        # '$\chi^{2}$=', str(round(chi2, 3))])
        newx = np.linspace(centre - 2*fwhm, centre + 2*fwhm,  100)
        newy = gaussian(maxval, fwhm, centre, newx)
        # to complete the peak curve, I need the vertical offset
        crit = np.where(np.logical_and(prof[:, 0] >= centre - 2*fwhm, prof[:, 0] <= centre +2*fwhm))
        oldx = prof[:, 0][crit]
        oldy = prof[:, 1][crit]
        refpeak = gaussian(maxval, fwhm, centre, oldx)
        diff = oldy - refpeak
        shift = diff.mean()
        newy += shift
        # and there, the problem is solved.
        peak = np.column_stack([newx, newy])
        self.fitted_peak_channels = peak
        self.fitting_textstring_channels = fitstring
    def fitPeak(self, manual_limits = None, bkg_perc = 50.0, redfac = 1.0):
        guess = self.eline()
        bkg = self.profile_channels[np.where(self.profile_channels[:, 1] < np.percentile(self.profile_channels[:, 1], bkg_perc))]
        self.background_channels = bkg
        if manual_limits is not None:
            lim1, lim2 = manual_limits[0], manual_limits[1]
            fit, peak, chi2 = elastic_line(self.profile_channels, bkg, olimits = (lim1,  lim2))
        elif guess is None:
            # self.logger("Guessing the elastic line position based on the intensity.")
            fit, peak, chi2 = elastic_line(self.profile_channels, bkg)
        elif len(self.energies) > 0:
            # self.logger("Guessing the elastic line position based on the energy calibration.")
            enval = self.energy
            xpos = guess[0]
            pwidth = 10 * redfac
            det_length = len(self.profile_channels)
            uplim_lim1 = det_length -1 - pwidth
            uplim_lim2 = det_length -1
            # self.logger("Based on the photon energy " + str(enval) + " the elastic line is at " + str(round(xpos, 3)) + " channels." )
            if xpos > 0 and xpos < det_length:
                fitlist, vallist, reslist = [], [], []
                for shift in np.arange(-20, 21, 10):
                    for ssize in [7,  15,  30]:
                        lim1 = int(xpos - (2* ssize + 1))
                        lim2 = int(xpos + ssize)
                        if lim1 < 0:
                            lim1 = 0
                        if lim1 > uplim_lim1:
                            lim1 = uplim_lim1
                        if lim2 > uplim_lim2:
                            lim2 = uplim_lim2
                        if lim2 < pwidth:
                            lim2 = pwidth
                        if lim2 - lim1 < pwidth:
                            lim2 = lim1 +pwidth
                        if lim1 > lim2:
                            lim1 = lim2 - pwidth
                        fit, peak, chi2 = elastic_line(self.profile_channels, bkg, olimits = (lim1,  lim2))
                        if fit is not None:
                            centre = fit[0][2]
                            fwhm = fit[0][1]
                            maxval = fit[0][0]
                            if not ((centre > lim1) and (centre < lim2)):
                                continue
                            reslist.append([centre, fwhm, maxval])
                            fwhm_quality = abs(round(fit[0][1] / fit[1][1],3))
                            centre_quality = abs(round(fit[0][2] / fit[1][2],3))
                            # vallist.append(maxval*(fwhm_quality + centre_quality)/abs(centre))
                            vallist.append((fwhm_quality + centre_quality)/abs(centre))
                            fitlist.append([fit,  peak, chi2])
                vallist = np.array(vallist)
                if len(vallist) == 0:
                    self.logger("All the automatic fits failed.")
                else:
                    maxval = vallist.max()
                    for nnn in range(len(vallist)):
                        if abs(maxval - vallist[nnn]) < 1e-10:
                            fit, peak, chi2 = fitlist[nnn]
        if fit is not None:
            peak_area = fit[0][0]*abs(fit[0][1])/gauss_denum*(2*np.pi)**0.5
            peak_area_error = peak_area*((fit[1][0]/fit[0][0])**2 + (fit[1][1]/fit[0][1])**2)**0.5
            fitstring = ' '.join(['FWHM =', str(abs(round(fit[0][1],3))), ' +/- ', str(abs(round(fit[1][1],3))),'\n',
                        'centre =', str(round(fit[0][2],3)), ' +/- ', str(abs(round(fit[1][2],3))),'\n',
                        'area =', str(abs(round(peak_area,3))), ' +/- ', str(abs(round(peak_area_error,3)))])# , '\n', 
                        # '$\chi^{2}$=', str(round(chi2, 3))])
        else:
            fitstring = "The fitting failed. Not enough data points?"
            return None
        self.fitted_peak_channels = peak
        self.fitting_params_channels['centre']=fit[0][2]
        self.fitting_params_channels[    'area']=peak_area 
        self.fitting_params_channels[    'maxval']=fit[0][0]
        self.fitting_params_channels[    'fwhm']=abs(fit[0][1])
        self.fitting_params_channels[   'centre_error']=fit[1][2]
        self.fitting_params_channels[  'area_error']=peak_area_error
        self.fitting_params_channels[   'maxval_error']=fit[1][0]
        self.fitting_params_channels[   'fwhm_error']=abs(fit[1][1])
        self.fitting_textstring_channels = fitstring
        # temp[:, 0] = polynomial(self.calibration_params, temp[:, 0]) # finish this next time.
    def makePeakInEnergy(self):
        guess = self.eline()
        if guess is None:
            return None
        if len(self.fitted_peak_channels) == 0:
            return None
        else:
            peak = self.fitted_peak_channels.copy()
        temp = self.fitting_params_channels
        centre = temp['centre']
        area = temp['area']
        maxval = temp['maxval']
        fwhm = temp['fwhm']
        centre_error = temp['centre_error']
        area_error = temp['area_error']
        maxval_error = temp['maxval_error']
        fwhm_error = temp['fwhm_error']
        absE = polynomial(self.calibration_params, self.profile_channels[:, 0])
        recalc = interp1d(self.profile_channels[:, 0], absE, fill_value = 0.0, bounds_error = False)
        peak[:, 0] = polynomial(self.calibration_params, peak[:, 0])
        if len(self.background_channels) > 0:
            self.background_absEnergy = np.column_stack([recalc(self.background_channels[:, 0]), self.background_channels[:, 1]])
        newcentre = recalc([centre])[0]
        tfwhm = recalc([centre-fwhm, centre+fwhm])
        newfwhm = 0.5*(tfwhm[1]-tfwhm[0])
        newarea = maxval*newfwhm/gauss_denum*(2*np.pi)**0.5
        newcentre_error = centre_error/centre*newcentre
        newfwhm_error = fwhm_error/fwhm*newfwhm
        newarea_error = area_error/area*newarea
        self.fitted_peak_absEnergy = peak
        self.fitting_params_absEnergy['centre']=newcentre 
        self.fitting_params_absEnergy['area']= newarea 
        self.fitting_params_absEnergy['maxval']=maxval 
        self.fitting_params_absEnergy['fwhm']=newfwhm
        self.fitting_params_absEnergy['centre_error']=newcentre_error
        self.fitting_params_absEnergy['area_error']=newarea_error
        self.fitting_params_absEnergy['maxval_error']=maxval_error 
        self.fitting_params_absEnergy['fwhm_error']=newfwhm_error
        self.fitting_textstring_absEnergy = ' '.join(['FWHM =', str(abs(round(newfwhm,3))), ' +/- ', str(abs(round(newfwhm_error,3))),'\n',
                        'centre =', str(round(newcentre,3)), ' +/- ', str(abs(round(newcentre_error,3))),'\n',
                        'area =', str(abs(round(newarea,3))), ' +/- ', str(abs(round(newarea_error,3)))])# , '\n', 
        altpeak = peak.copy()
        altpeak[:, 0] = newcentre - altpeak[:, 0]
        if len(self.background_channels) > 0:
            temp = newcentre - recalc(self.background_channels[:, 0])
            self.background_eV = np.column_stack([temp, self.background_channels[:, 1]])
        temp2 = newcentre - recalc(self.profile_channels[:, 0])
        self.profile_eV = np.column_stack([temp2, self.profile_channels[:, 1]])
        self.fitted_peak_eV = altpeak
        self.fitting_params_eV['centre']=0.0
        self.fitting_params_eV['area']= newarea 
        self.fitting_params_eV[ 'maxval']=maxval 
        self.fitting_params_eV[ 'fwhm']=newfwhm
        self.fitting_params_eV[ 'centre_error']=newcentre_error
        self.fitting_params_eV[  'area_error']=newarea_error 
        self.fitting_params_eV[  'maxval_error']=maxval_error 
        self.fitting_params_eV[ 'fwhm_error']=newfwhm_error
        self.fitting_textstring_eV = ' '.join(['FWHM =', str(abs(round(1000.0*newfwhm,3))), ' +/- ', str(abs(round(1000.0*newfwhm_error,3))),' meV\n',
                        'centre =', str(round(0.0,3)), ' +/- ', str(abs(round(newcentre_error,3))),'\n',
                        'area =', str(abs(round(newarea,3))), ' +/- ', str(abs(round(newarea_error,3)))])# , '\n', 
    def makeEnergyProfile(self, fitorder=2):
        self.makeEnergyCalibration(fitorder)
        self.nominal_elastic_line_position = polynomial(self.reverse_calibration_params, [self.energy])
        self.individual_profiles_absEnergy = []
        for prof in self.individual_profiles_channels:
            temp = prof.copy()
            temp[:, 0] = polynomial(self.calibration_params, temp[:, 0])
            self.individual_profiles_absEnergy.append(temp)
        self.profile_eV = []
        for prof in [self.profile_channels]:
            temp = prof.copy()
            temp[:, 0] = polynomial(self.calibration_params, temp[:, 0])
            self.profile_absEnergy = temp
        self.addHistory("Converted to energy, with calibration parameters: " + str(self.calibration_params))
    def fitCurvature(self, limits = None, segsize = 16, bkg_perc = 65):
        if limits is None:
            limits = self.eline_limits
        if self.merged_data is not None:
            curve = curvature_profile(self.merged_data, blocksize = segsize, percentile = bkg_perc,
                                                olimits = limits)
            pfit, pcov, infodict, errmsg, success = leastsq(fit_polynomial, [curve[:,1].mean(), 0.0, 0.0],
                                                                                args = (curve,), full_output = 1)
            curvefit = polynomial(pfit, curve[:,0])
            curvefit = np.column_stack([curve[:,0], curvefit])
            self.curvature = curve
            self.curvature_fit = curvefit
            self.curvature_params = pfit
            # self.curvatureresult.emit([curve,  curvefit,  pfit])
        # else:
            # self.did_nothing.emit()
    def fakeData(self, cuts, bpp):
        names = ['Mock_File_001', 'Mock_Header_001', 'Mock_Logfile_001']
        # self.data = np.random.random((2048, 2048))*self.bpp
        width = int(abs(cuts[3] - cuts[2]))
        height = int(abs(cuts[1] - cuts[0]))
        try:
            data = rand_mt.normal(loc = 0.2*bpp,  scale = 0.1*bpp,  size = (width, height))
        except:
            data = rand_mt.standard_normal(size = (width * height))*0.1*bpp + 0.2*bpp
            data = self.data.reshape((width, height))
        header = {'Photon energy': -111.1, "Measuring time": 111.1}
        logvals = {'Time' : np.arange(100), 
                               'XPOS' : np.arange(100) + np.random.random(100), 
                               'Current1' : 1e-10*(10.0 +np.random.random(100))}
        self.shortnames = 'MockAdler001'
        self.times = np.array([999.0])
        self.original_data2D = data
        self.data2D = self.original_data2D
        self.merged_data = data
        self.plotax = [(cuts[2:4], cuts[0:2])]
        self.addHistory('FakeDataGenerated')
        self.data.append(data)
        self.headers.append(header)
        self.logs.append(logvals)
        self.data_files.append(names[0])
        self.header_files.append(names[1])
        self.log_files.append(names[2])
        self.individual_profiles_channels.append([])
        self.individual_profiles_absEnergy.append([])
        self.individual_profiles_eV.append([])
    def __add__(self, other):
        temp = RixsMeasurement()
        # temp.flist = self.flist + other.flist
        pdict1 = self.summariseCrucialParts()
        pdict2 = other.summariseCrucialParts()
        temp.data = self.data + other.data
        temp.headers = self.headers + other.headers
        temp.logs = self.logs + other.logs
        temp.data_files = self.data_files + other.data_files
        temp.header_files = self.header_files + other.header_files
        temp.log_files = self.log_files + other.log_files
        temp.individual_profiles_channels = self.individual_profiles_channels + other.individual_profiles_channels
        temp.individual_profiles_absEnergy = self.individual_profiles_absEnergy + other.individual_profiles_absEnergy
        temp.individual_profiles_eV = self.individual_profiles_eV + other.individual_profiles_eV
        profs1 = [self.profile_channels.copy(), self.profile_eV.copy(), self.profile_absEnergy.copy()]
        profs2 = [other.profile_channels.copy(), other.profile_eV.copy(), other.profile_absEnergy.copy()]
        newp = [np.array([]), np.array([]), np.array([])]
        temp.energies = self.energies + other.energies
        temp.postprocess()
        for nn in range(3):
            tp1 = profs1[nn]
            tp2 = profs2[nn]
            if len(tp1) == 0:
                if len(tp2) == 0:
                    continue
                else:
                    newp[nn] = tp2
            else:
                if len(tp2) == 0:
                    newp[nn] = tp1
                else:
                    step1 = abs((tp1[1:, 0] - tp1[:-1, 0]).mean())
                    step2 = abs((tp2[1:, 0] - tp2[:-1, 0]).mean())
                    newstep = min(step1, step2)
                    newmin = min(tp1[:, 0].min(), tp2[:, 0].min())
                    newmax = max(tp1[:, 0].max(), tp2[:, 0].max())
                    newx = np.arange(newmin, newmax+0.1*newstep, newstep)
                    target = np.column_stack([newx, np.zeros(len(newx))])
                    the_object = MergeManyCurves([tp1, tp2], target, mthreads = self.maxthreads)
                    the_object.runit()
                    newp[nn] = the_object.postprocess()
        temp.profile_channels = newp[0]
        temp.profile_eV = newp[1]
        temp.profile_absEnergy = newp[2]
        # and to be on the safe side:
        temp.shortsource = "SummedMeasurements"
        temp.times = self.times + other.times
        # temp.energy = 0.5*()
        for kk in other.the_log.keys():
            try:
                temp.the_log[kk] = np.concatenate([self.the_log[kk], other.the_log[kk]])
            except:
                continue
        # pdict['temperature']
        # pdict['arm_theta']
        # pdict['Q']
        return temp


class AdlerCore(QObject):
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
        # self.threadpool = QThreadPool(self)
        # self.threadpool = QThreadPool.globalInstance()
        self.threadpool = CustomThreadpool(MAX_THREADS = max_threads)
        # self.threadpool.setMaxThreadCount(max_threads)
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
        self.tdb_profile = None
        self.energy_profile = None
        self.fitted_peak_energy = None
        self.fitting_params_energy = None
        self.fitting_textstring_energy = None
        self.fitted_peak_channels = None
        self.fitting_params_channels = None
        self.fitting_textstring_channels = None
        self.plotax,  self.plotax1D = None,  None
        self.curvature,  self.curvature_fit,  self.curvature_params = None,  None,  None
        self.segment_plot = None
        self.histogram_plot = None
        self.curvature_corrected = False
        self.fft_applied = False
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
    def load_last_params(self):
        try:
            source = open(os.path.join(expanduser("~"),'.ADLERcore.txt'), 'r')
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
            source = open(resource_path('Structured_bkg_per_second.dat'), 'r')
        else:
            self.logger("Loading background per channel per second from BKG_OVERRIDE.txt")
        arr = []
        for line in source:
            arr.append([float(x) for x in line.split()])
        source.close()
        self.tdb_profile = np.array(arr)
    def load_files(self,  flist):
        if len(flist) > 0:
            self.current_flist = flist
            self.temp_path = os.path.split(flist[0])[0]
            self.name_as_segments = simplify_number_range(flist)
            self.temp_name = "_".join(['Merged']+self.name_as_segments)
            # the_object = LoadFileList(flist,  self.bpp, self.cray,  self.threadpool,  self.progbar)
            datalist = []
            headlist = []
            varloglist = []
            names = []
            for fname in flist:
                data, tadded_header, tadded_log,  fname = simple_read(fname, self.bpp,  self.cray)
                datalist.append(data)
                headlist.append(tadded_header)
                varloglist.append(tadded_log)
                names.append(fname)
            datashape = data.shape
            self.det_width = datashape[1]
            self.det_length = datashape[0]
            # the_object = LoadFileList(flist,  self.bpp, self.cray,  None,  self.progbar,  mthreads = self.maxthreads)
            # the_object.runit()
            # the_object.isitover()
            self.data,  self.header,  self.logvals,  self.logvalnames = None,  None,  None,  None
            self.shortnames = None
            self.energies = []
            self.timedata = (1, np.zeros(1))
            # self.logger("LoadFileList -> have all files been processed? " + str(the_object.finished))
            # self.logger("LoadFileList -> did all threads terminate? " + str(the_object.noproblems))
            self.data,  self.header,  self.logvals,  self.shortnames,  self.timedata,  self.energies = postprocess([datalist, headlist, varloglist, names])
            self.logvalnames = self.logvals.keys()
            self.data = self.data[:, :, self.cuts[0]:self.cuts[1]]
            self.det_width = self.data.shape[2]
            # self.data,  self.header,  self.logvals = load_filelist(flist,  self.bpp,  self.cray,  self.poly)
            # self.plotax = [(1,self.det_length), self.cuts]
            self.plotax = [(1,self.det_length), self.cuts]
            scalefac = 1.0/self.det_width * self.tdb_fact
            if (self.tdb_profile is not None) and self.tdb_fact > 0.0 :
                temp = self.tdb_profile[:, 1].copy()
                temp *= scalefac
                if not len(temp) == self.data.shape[1]:
                    self.bkg_profile = None
                else:
                    for nd in np.arange(len(self.data)):
                        temp2 = temp*self.timedata[1][nd]
                        for n in np.arange(self.det_width):
                            self.data[nd, :, n] -= temp2
            else:
                self.bkg_profile = None
            # print("STATS: ", self.timedata[0], self.timedata[1].sum(),  self.data.sum(),  self.data.std())
    def adjust_offsets_automatically(self,  tolerance = 10.0):
        profiles,  offsets = [],  [0.0]
        shifted = []
        for ds in self.data:
            profiles.append(make_profile(ds, xaxis = np.linspace(1, self.det_length, int(self.det_length/self.redfac)), 
                                 tpoolin = self.threadpool, pbarin = self.progbar,  maxthreads = self.maxthreads))
        temp = np.zeros(profiles[0].shape)
        temp[:, 0] = profiles[0][:, 0]
        temp2 = temp.copy()
        shifted.append(profiles[0].copy())
        for prof in profiles[1:]:
            p = prof.copy()
            pfit, pcov, infodict, errmsg, success = leastsq(profile_offsets, [0.0, 1.0, 0.0],
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
        for ds in self.data:
            profiles.append(make_profile(ds, xaxis = np.linspace(1, self.det_length, int(self.det_length/self.redfac)), 
                                 tpoolin = self.threadpool, pbarin = self.progbar,  maxthreads = self.maxthreads))
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
        newdata = [self.data[0].copy()]
        for n, of in enumerate(offsets):
            if n>0:
                newdata.append(apply_offset_to_2D_data(self.data[n], self.cuts[0], self.cuts[1], of))
        self.data = np.array(newdata)
        self.processing_history.append(['OffsetsOnLoad']+offsets)
        self.finished_offsets.emit()
    @pyqtSlot(object)
    def justload_manyfiles(self, flist):
        self.load_files(flist)
        self.original_data2D = self.data.sum(0)
        self.data2D = self.original_data2D
        self.finished_2D.emit()
        self.save_last_params()
    @pyqtSlot(object)
    def preprocess_manyfiles(self, flist):
        self.load_files(flist)
        off,  shift = self.adjust_offsets_automatically()
        self.offsets = off
        self.finished_preprocess.emit()
        self.save_last_params()
    @pyqtSlot(object)
    def expensive_merge_manyfiles(self, flist):
        self.load_files(flist)
        off,  shift = self.adjust_offsets_expensive()
        self.apply_offsets(off)
        self.original_data2D = self.data.sum(0)
        self.data2D = self.original_data2D
        self.finished_process.emit()
        self.finished_2D.emit()
        self.save_last_params()
    @pyqtSlot(object)
    def process_manyfiles(self, flist):
        self.load_files(flist)
        off,  shift = self.adjust_offsets_automatically()
        self.apply_offsets(off)
        self.original_data2D = self.data.sum(0)
        self.data2D = self.original_data2D
        self.finished_process.emit()
        self.finished_2D.emit()
        self.save_last_params()
    @pyqtSlot()
    def generate_mock_dataset(self):
        self.current_flist = ['Mock_File_001']
        # self.temp_path = os.path.split(flist[0])[0]
        self.name_as_segments = ['Mock_File_001']
        self.temp_name = "_".join(['Merged']+self.name_as_segments)
        # self.data = np.random.random((2048, 2048))*self.bpp
        try:
            self.data = rand_mt.normal(loc = 0.2*self.bpp,  scale = 0.1*self.bpp,  size = (2048, 2048))
        except:
            self.data = rand_mt.standard_normal(size = (2048 * 2048))*0.1*self.bpp + 0.2*self.bpp
            self.data = self.data.reshape((2048, 2048))
        self.header = '# Fake dataset for ADLER'
        self.logvals = {'Time' : np.arange(100), 
                               'XPOS' : np.arange(100) + np.random.random(100), 
                               'Current1' : 1e-10*(10.0 +np.random.random(100))}
        self.logvalnames = self.logvals.keys()
        self.shortnames = 'MockAdler001'
        self.timedata = (1,  np.array([999.0]))
        self.original_data2D = self.data
        self.data2D = self.original_data2D
        self.plotax = [(1,2048), self.cuts]
        self.processing_history.append(['FakeDataGenerated'])
        self.finished_process.emit()
        self.finished_2D.emit()
    @pyqtSlot()
    def finalise_manyfiles(self):
        self.apply_offsets(self.offsets)
        self.original_data2D = self.data.sum(0)
        self.data2D = self.original_data2D
        self.finished_2D.emit()
    def curve_profile(self):  
        lcut,  hcut = self.cuts
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
        lcut,  hcut = self.cuts
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
        self.raw_profile = make_profile(self.data2D,  self.redfac,  xaxis = np.linspace(1, self.det_length, int(self.det_length/self.redfac)), 
                                                  tpoolin = self.threadpool, pbarin = self.progbar, maxthreads = self.maxthreads)
        self.plotax1D = [(1,self.det_length), (self.cuts[0]+1, self.cuts[1])]
        WriteProfile(os.path.join(self.temp_path, self.temp_name+"_1D.txt"), self.raw_profile,
                        header = self.header.split('\n'), params = [self.parlist,  self.pardict], 
                        varlog = [self.logvalnames,  self.logvals])
        x1, eV1 = self.eVpair1
        x2, eV2 = self.eVpair2
        if self.eline[0] < 0 and self.eline[1] < 0:
            return self.raw_profile.copy(), None, None, None, None
        else:
            bkg = self.raw_profile[np.where(self.raw_profile[:, 1] < np.percentile(self.raw_profile[:, 1], self.bkg_perc))]
            if guess:
                if (eV1 < 0) or (eV2 < 0):
                    self.logger("Guessing the elastic line position based on the intensity.")
                    fit, peak, chi2 = elastic_line(self.raw_profile, bkg)
                elif len(self.energies) > 0:
                    self.logger("Guessing the elastic line position based on the energy calibration.")
                    enval = self.energies.mean()
                    slope = (x2-x1)/(eV2 - eV1)
                    xpos = x1 + (enval - eV1)*slope
                    pwidth = 10 * self.redfac
                    uplim_lim1 = self.det_length -1 - pwidth
                    uplim_lim2 = self.det_length -1
                    self.logger("Based on the photon energy " + str(enval) + " the elastic line is at " + str(round(xpos, 3)) + " channels." )
                    if xpos > 0 and xpos < self.det_length:
                        fitlist, vallist = [], []
                        for ssize in [7,  15,  30]:
                            lim1 = int(xpos - (2* ssize + 1))
                            lim2 = int(xpos + ssize)
                            if lim1 < 0:
                                lim1 = 0
                            if lim1 > uplim_lim1:
                                lim1 = uplim_lim1
                            if lim2 > uplim_lim2:
                                lim2 = uplim_lim2
                            if lim2 < pwidth:
                                lim2 = pwidth
                            if lim2 - lim1 < pwidth:
                                lim2 = lim1 +pwidth
                            if lim1 > lim2:
                                lim1 = lim2 - pwidth
                            fit, peak, chi2 = elastic_line(self.raw_profile, bkg, olimits = (lim1,  lim2))
                            if fit is not None:
                                fwhm_quality = abs(round(fit[0][1] / fit[1][1],3))
                                centre_quality = abs(round(fit[0][2] / fit[1][2],3))
                                vallist.append(fwhm_quality + centre_quality)
                                fitlist.append([fit,  peak, chi2])
                        vallist = np.array(vallist)
                        if len(vallist) == 0:
                            self.logger("All the automatic fits failed.")
                        else:
                            maxval = vallist.max()
                            for nnn in range(len(vallist)):
                                if abs(maxval - vallist[nnn]) < 1e-10:
                                    fit, peak, chi2 = fitlist[nnn]
                    else:
                        self.logger("Reverting to the guess of the elastic line position based on the intensity.")
                        fit, peak, chi2 = elastic_line(self.raw_profile, bkg)
                else:
                    fit, peak, chi2 = elastic_line(self.raw_profile, bkg)
            else:
                fit, peak, chi2 = elastic_line(self.raw_profile, bkg, olimits = (self.eline[0],  self.eline[1]))
            if fit is not None:
                # chi2 = chisquared(peak, self.raw_profile)
                self.logger('heigth, FWHM, centre, baseline')
                self.logger(str(fit[0]))
                self.logger(str(fit[1]))
                peak_area = fit[0][0]*abs(fit[0][1])/gauss_denum*(2*np.pi)**0.5
                peak_area_error = peak_area*((fit[1][0]/fit[0][0])**2 + (fit[1][1]/fit[0][1])**2)**0.5
                fitstring = ' '.join(['FWHM =', str(abs(round(fit[0][1],3))), ' +/- ', str(abs(round(fit[1][1],3))),'\n',
                            'centre =', str(round(fit[0][2],3)), ' +/- ', str(abs(round(fit[1][2],3))),'\n',
                            'area =', str(round(peak_area,3)), ' +/- ', str(abs(round(peak_area_error,3)))])# , '\n', 
                            # '$\chi^{2}$=', str(round(chi2, 3))])
            else:
                fitstring = "The fitting failed. Not enough data points?"
            self.fitted_peak_channels = peak
            self.fitting_params_channels = fit
            self.fitting_textstring_channels = fitstring
            tempthing = self.raw_profile.copy()
            if final:
                self.fittingresult.emit([tempthing,  bkg, peak,  fit,  fitstring])
                if fit is not None:
                    self.params_fitting.emit({ 'FIT_centre':np.array([round(fit[0][2],3), abs(round(fit[1][2],3))]), 
                                                       'FIT_fwhm':np.array([abs(round(fit[0][1],3)), abs(round(fit[1][2],3))]), 
                                                       'FIT_area':np.array([round(peak_area,3), abs(round(peak_area_error,3))])})
            return tempthing,  bkg, peak,  fit,  fitstring
    @pyqtSlot()
    def minimal_process_file(self, inp_prof, guess = False,  final = True):  
        self.raw_profile = inp_prof
        self.plotax1D = [(1,self.det_length), (self.cuts[0]+1, self.cuts[1])]
        WriteProfile(os.path.join(self.temp_path, self.temp_name+"_1D.txt"), self.raw_profile,
                        header = self.header.split('\n'), params = [self.parlist,  self.pardict], 
                        varlog = [self.logvalnames,  self.logvals])
        x1, eV1 = self.eVpair1
        x2, eV2 = self.eVpair2
        if self.eline[0] < 0 and self.eline[1] < 0:
            return self.raw_profile.copy(), None, None, None, None
        else:
            bkg = self.raw_profile[np.where(self.raw_profile[:, 1] < np.percentile(self.raw_profile[:, 1], self.bkg_perc))]
            if guess:
                if (eV1 < 0) or (eV2 < 0):
                    self.logger("Guessing the elastic line position based on the intensity.")
                    fit, peak, chi2 = elastic_line(self.raw_profile, bkg)
                elif len(self.energies) > 0:
                    self.logger("Guessing the elastic line position based on the energy calibration.")
                    enval = self.energies.mean()
                    slope = (x2-x1)/(eV2 - eV1)
                    xpos = x1 + (enval - eV1)*slope
                    pwidth = 10 * self.redfac
                    uplim_lim1 = self.det_length -1 - pwidth
                    uplim_lim2 = self.det_length -1
                    self.logger("Based on the photon energy " + str(enval) + " the elastic line is at " + str(round(xpos, 3)) + " channels." )
                    if xpos > 0 and xpos < self.det_length:
                        fitlist, vallist = [], []
                        for ssize in [7,  15,  30]:
                            lim1 = int(xpos - (2* ssize + 1))
                            lim2 = int(xpos + ssize)
                            if lim1 < 0:
                                lim1 = 0
                            if lim1 > uplim_lim1:
                                lim1 = uplim_lim1
                            if lim2 > uplim_lim2:
                                lim2 = uplim_lim2
                            if lim2 < pwidth:
                                lim2 = pwidth
                            if lim2 - lim1 < pwidth:
                                lim2 = lim1 +pwidth
                            if lim1 > lim2:
                                lim1 = lim2 - pwidth
                            fit, peak, chi2 = elastic_line(self.raw_profile, bkg, olimits = (lim1,  lim2))
                            if fit is not None:
                                fwhm_quality = abs(round(fit[0][1] / fit[1][1],3))
                                centre_quality = abs(round(fit[0][2] / fit[1][2],3))
                                vallist.append(fwhm_quality + centre_quality)
                                fitlist.append([fit,  peak, chi2])
                        vallist = np.array(vallist)
                        if len(vallist) == 0:
                            self.logger("All the automatic fits failed.")
                        else:
                            maxval = vallist.max()
                            for nnn in range(len(vallist)):
                                if abs(maxval - vallist[nnn]) < 1e-10:
                                    fit, peak, chi2 = fitlist[nnn]
                    else:
                        self.logger("Reverting to the guess of the elastic line position based on the intensity.")
                        fit, peak, chi2 = elastic_line(self.raw_profile, bkg)
                else:
                    fit, peak, chi2 = elastic_line(self.raw_profile, bkg)
            else:
                fit, peak, chi2 = elastic_line(self.raw_profile, bkg, olimits = (self.eline[0],  self.eline[1]))
            if fit is not None:
                # chi2 = chisquared(peak, self.raw_profile)
                self.logger('heigth, FWHM, centre, baseline')
                self.logger(str(fit[0]))
                self.logger(str(fit[1]))
                peak_area = fit[0][0]*abs(fit[0][1])/gauss_denum*(2*np.pi)**0.5
                peak_area_error = peak_area*((fit[1][0]/fit[0][0])**2 + (fit[1][1]/fit[0][1])**2)**0.5
                fitstring = ' '.join(['FWHM =', str(abs(round(fit[0][1],3))), ' +/- ', str(abs(round(fit[1][1],3))),'\n',
                            'centre =', str(round(fit[0][2],3)), ' +/- ', str(abs(round(fit[1][2],3))),'\n',
                            'area =', str(round(peak_area,3)), ' +/- ', str(abs(round(peak_area_error,3)))])# , '\n', 
                            # '$\chi^{2}$=', str(round(chi2, 3))])
            else:
                fitstring = "The fitting failed. Not enough data points?"
            self.fitted_peak_channels = peak
            self.fitting_params_channels = fit
            self.fitting_textstring_channels = fitstring
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
        x1, eV1 = self.eVpair1
        x2, eV2 = self.eVpair2
        lcut, hcut = self.cuts
        emin,  emax = self.eline
        if (eV1 < 0) or (eV2 < 0):
            self.logger("Invalid energy range, skipping the calculation")
            self.did_nothing.emit()
            return None, None,  None, None,  None
        else:
            profi, back, peak, fit,  fitstring = self.process_file(guess,  final = False)
            xaxis = profi[:,0].copy()
            xstep = (xaxis[1:] - xaxis[:-1]).mean()
            slope = (eV2-eV1)/(x2-x1)
            xaxis = np.concatenate([xaxis[:1]-xstep, xaxis, xaxis[-1:]+xstep])
            new_xaxis = (xaxis - x1)*slope + eV1
            recalc = interp1d(xaxis, new_xaxis)
            profi[:,0] = recalc(profi[:,0])
            self.energy_profile = profi.copy()
            if emin < 0 and emax < 0:
                WriteProfile(os.path.join(self.temp_path, self.temp_name+"_1D_absEnergy.txt"), self.energy_profile,
                        header = self.header.split('\n'), params = [self.parlist,  self.pardict], 
                        varlog = [self.logvalnames,  self.logvals])
                return profi, back, peak, fit
            back[:,0] = recalc(back[:,0])
            peak[:,0] = recalc(peak[:,0])
            new_centre = recalc(fit[0][2])
            self.logger('heigth, FWHM, centre, baseline')
            self.logger(str(fit[0]))
            self.logger(str(fit[1]))
            peak_area = fit[0][0]*abs(slope*fit[0][1])/gauss_denum*(2*np.pi)**0.5
            peak_area_error = peak_area*((fit[1][0]/fit[0][0])**2 + (fit[1][1]/fit[0][1])**2)**0.5
            profi[:,0] -= new_centre
            back[:,0] -= new_centre
            peak[:,0] -= new_centre
            profi[:,0] *= -1
            back[:,0] *= -1
            peak[:,0] *= -1
            fitstring = ' '.join(['FWHM (meV) =', str(abs(round(1000.0*slope*fit[0][1],3))), ' +/- ', str(abs(round(1000.0*slope*fit[1][1],3))),'\n',
                        'centre (meV) =', str(round(0.0,3)), ' +/- ', str(abs(round(1000.0*fit[1][2],3))),'\n',
                        'area =', str(round(peak_area,3)), ' +/- ', str(abs(round(peak_area_error,3))) ])
                        # , '\n', 
                        # '$\chi^{2}$=', str(round(chi2, 3))])
            self.energy_profile = profi                
            WriteProfile(os.path.join(self.temp_path, self.temp_name+"_1D_deltaE.txt"), self.energy_profile,
                        header = self.header.split('\n'), params = [self.parlist,  self.pardict], 
                        varlog = [self.logvalnames,  self.logvals])
            self.fitted_peak_energy = peak
            self.fitting_params_energy = fit
            self.fitting_textstring_energy = fitstring
            self.energyresult.emit([profi, back, peak, fit,  fitstring])
            self.params_energy.emit({  'ENERGY_centre':np.array([0.0, abs(round(1000.0*fit[1][2],3))]), 
                                                    'ENERGY_fwhm': np.array([abs(round(1000.0*slope*fit[0][1],3)), abs(round(1000.0*slope*fit[1][1],3))]), 
                                                    'ENERGY_area':np.array([round(peak_area,3), abs(round(peak_area_error,3))])})
            return profi, back, peak, fit,  fitstring

def precise_merge(curves, outname = 'best_merge.txt'):
    datalist, energylist, unitlist, dictlist = [],[],[], []
    for fname in curves:
        a,b,c,d = read_1D_curve_extended(fname)
        datalist.append(a)
        energylist.append(b)
        unitlist.append(c)
        dictlist.append(d)
    # energies = np.array(energylist).mean()
    # I ignore the units and energies. Let's assume I put the correct files in the list.
    minx, maxx, stepx = 1e5,-1e5,-1.0
    for dat in datalist:
        minx = min(minx, dat[:,0].min())
        maxx = max(maxx, dat[:,0].max())
        step = (dat[1:,0] - dat[:-1,0]).mean()
        stepx = max(stepx, step)
    newx = np.arange(minx, maxx + 0.1*stepx, stepx)
    target = np.zeros((len(newx),2))
    target[:,0] = newx
    for dat in datalist:
        target = merge2curves(dat,target)
    WriteEnergyProfile(outname, target, [])
    return target

def single_n2_profile(params, xarr, fixed_lorentz = 0.12, yarr = None, extraweight = False):
    num_peaks = int((len(params) - 3)/2)
    profile = np.zeros(xarr.shape)
    dists = []
    scalefac = 1.0
    for n in range(num_peaks):
        profile += my_voigt(xarr, abs(params[2*n]), abs(params[2*n+1]), abs(params[-3]), fixed_lorentz)
        dists.append(params[2*n+1])
    dists = np.sort(dists)
    dists = dists[1:] - dists[:-1]
    profile += params[-2]*xarr + params[-1]
    if yarr is not None:
        profile -= yarr
        if extraweight:
            if len(dists) > 0:
                scalefac = 1.0 + max(0.0, 0.15 - dists.min())
    return profile * scalefac
    
def single_neon_profile(params, xarr, fixed_lorentz = 0.254, yarr = None, extraweight = False):
    num_peaks = int((len(params) - 5)/2)
    profile = np.zeros(xarr.shape)
    scalefac = 1.0
    for n in range(num_peaks):
        profile += my_voigt(xarr, abs(params[2*n]), abs(params[2*n+1]), abs(params[-5]), fixed_lorentz)
    profile += abs(params[-4])*np.arctan((xarr-params[-3])/params[-2]) + params[-1]
    if yarr is not None:
        profile -= yarr
        if extraweight:
            scalefac = 1.0 + max(0.0, 870 - params[-3])
    return profile * scalefac
    
def single_edge_profile(params, xarr, fixed_lorentz = 0.254, yarr = None):
    profile = np.zeros(xarr.shape)
    profile += params[-4]*np.arctan((xarr-params[-3])/params[-2]) + params[-1]
    if yarr is not None:
        profile -= yarr
    return profile

def fit_edge_profile(num_arr,  nlines = 0):
    mid_y = (num_arr[:, 1].max()+num_arr[:, 1].min())/2.0
    crit = np.abs(num_arr[:, 1] - mid_y)
    ref = crit.min()
    mid_x = num_arr[:, 0][np.where(crit==ref)].mean()
    step = np.abs(num_arr[1:, 0] - num_arr[:-1, 0]).mean()
    params = [num_arr[-1, 1], mid_x, 10*step, num_arr[0, 1]]
    pfit, pcov, infodict, errmsg, success = leastsq(single_edge_profile, params, args = (num_arr[:, 0], 0.254, num_arr[:, 1]), full_output =1)
    final_profile = single_edge_profile(pfit, num_arr[:, 0], 0.254)
    GOF = ((final_profile - num_arr[:, 1])**2).sum()**0.5/ np.abs(num_arr[:, 1]).sum()
    print('GOF: ',  GOF, ((final_profile - num_arr[:, 1])**2).sum()**0.5, np.abs(num_arr[:, 1]).sum())
    return final_profile, pfit[1],  pfit[2],  GOF

def fit_n2_gas(inp_num_arr, fixed_gamma = 0.06):
    peak_list = np.array([400.76, 400.996, 401.225, 401.45, 401.66, 401.88, 402.1, 402.29, 402.48])
    params = []
    # fixed_gamma = 0.012
    seq = np.argsort(inp_num_arr[:, 0])
    num_arr = inp_num_arr[seq]
    temp_prof = interp1d(num_arr[:, 0],  num_arr[:, 1])
    for peak in peak_list:
        if peak >= num_arr[:, 0].min() and peak <= num_arr[:, 0].max():
            params += [temp_prof(peak), peak]
    params += [0.01, 0.0, 0.0]
    pfit, pcov, infodict, errmsg, success = leastsq(single_n2_profile, params, args = (num_arr[:, 0], fixed_gamma, num_arr[:, 1],  True), full_output =1)
    final_profile = single_n2_profile(pfit, num_arr[:, 0], fixed_gamma)
    # determine the fit quality
    GOF = ((final_profile - num_arr[:, 1])**2 ).sum()**0.5/ np.abs(num_arr[:, 1]).sum()
    print('GOF: ',  GOF, ((final_profile - num_arr[:, 1])**2 ).sum()**0.5, np.abs(num_arr[:, 1]).sum() )
    resline = np.concatenate([[GOF], pfit])
    results = [resline]
    if 1.0/GOF < 50:
        for xx in range(3):
            newparams = np.array(params)
            plen = len(newparams)
            rands = np.random.rand(plen)
            newparams[0:-3:2] *= rands[0:-3:2]
            newparams[1:-3:2] += 0.1*(rands[1:-3:2] - 0.5)
            pfit, pcov, infodict, errmsg, success = leastsq(single_n2_profile, params, args = (num_arr[:, 0], fixed_gamma, num_arr[:, 1],  True), full_output =1)
            final_profile = single_n2_profile(pfit, num_arr[:, 0], fixed_gamma)
            # determine the fit quality
            GOF = ((final_profile - num_arr[:, 1])**2 ).sum()**0.5/ np.abs(num_arr[:, 1]).sum()
            resline = np.concatenate([[GOF], pfit])
            results.append(resline)
        results = np.array(results)
        results = results[np.argsort(results[:, 0])]
        pfit = results[0, 1:]
        final_profile = single_n2_profile(pfit, num_arr[:, 0], fixed_gamma)
        # determine the fit quality
        GOF = ((final_profile - num_arr[:, 1])**2 ).sum()**0.5/ np.abs(num_arr[:, 1]).sum()
        print('best GOF: ',  GOF, ((final_profile - num_arr[:, 1])**2 ).sum()**0.5, np.abs(num_arr[:, 1]).sum() )
    # final_profile = single_n2_profile(params, num_arr[:, 0], fixed_gamma)
    # gauss_sigma = params[-3]
    # peak_pos = params[1:-3:2]
    gauss_sigma = pfit[-3]
    lorentz_gamma = fixed_gamma
    peak_pos = pfit[1:-3:2]
    return final_profile, gauss_sigma, lorentz_gamma, peak_pos,  GOF

def fit_neon_gas(inp_num_arr,  fixed_gamma = 0.126):
    peak_list = np.array([867.2, 868.8, 869.36, 869.6,  869.95, 870.23, 870.42, 870.60])
    atan2_params = [870.15, 0.125]
    # y = y0 + A*atan((x-xc)/w)
    params = []
    seq = np.argsort(inp_num_arr[:, 0])
    num_arr = inp_num_arr[seq]
    temp_prof = interp1d(num_arr[:, 0],  num_arr[:, 1])
    for peak in peak_list[:1]:
        if peak >= num_arr[:, 0].min() and peak <= num_arr[:, 0].max():
            params += [temp_prof(peak), peak]
        else:
            params += [num_arr[:, 1].max(), num_arr[:, 0].mean()]
    for peak in peak_list[1:]:
        if peak >= num_arr[:, 0].min() and peak <= num_arr[:, 0].max():
            params += [temp_prof(peak), peak]
    if len(params) == 2:
        params[0] = num_arr[:, 1].max()
        params[1] = num_arr[:, 0][np.where(num_arr[:, 1] == params[0])][0]
        params += [0.01, 0.0,  atan2_params[0], atan2_params[1], 0.0]
    else:
        params += [0.01, num_arr[-1, 1],  atan2_params[0], atan2_params[1], num_arr[0, 1]]
    pfit, pcov, infodict, errmsg, success = leastsq(single_neon_profile, params, args = (num_arr[:, 0], fixed_gamma, num_arr[:, 1],  True), full_output =1)
    final_profile = single_neon_profile(pfit, num_arr[:, 0], fixed_gamma)
    # determine the fit quality
    GOF = ((final_profile - num_arr[:, 1])**2).sum()**0.5/ np.abs(num_arr[:, 1]).sum()
    print('GOF: ',  GOF, ((final_profile - num_arr[:, 1])**2 ).sum()**0.5, np.abs(num_arr[:, 1]).sum() )
    #
    resline = np.concatenate([[GOF], pfit])
    results = [resline]
    if 1.0/GOF < 50:
        for xx in range(3):
            newparams = np.array(params)
            plen = len(newparams)
            rands = np.random.rand(plen)
            newparams[0:-5:2] *= rands[0:-5:2]
            newparams[1:-5:2] += 0.1*(rands[1:-5:2] - 0.5)
            pfit, pcov, infodict, errmsg, success = leastsq(single_neon_profile, params, args = (num_arr[:, 0], fixed_gamma, num_arr[:, 1],  True), full_output =1)
            final_profile = single_neon_profile(pfit, num_arr[:, 0], fixed_gamma)
            # determine the fit quality
            GOF = ((final_profile - num_arr[:, 1])**2 ).sum()**0.5/ np.abs(num_arr[:, 1]).sum()
            resline = np.concatenate([[GOF], pfit])
            results.append(resline)
        results = np.array(results)
        results = results[np.argsort(results[:, 0])]
        pfit = results[0, 1:]
        final_profile = single_neon_profile(pfit, num_arr[:, 0], fixed_gamma)
        # determine the fit quality
        GOF = ((final_profile - num_arr[:, 1])**2 ).sum()**0.5/ np.abs(num_arr[:, 1]).sum()
        print('best GOF: ',  GOF, ((final_profile - num_arr[:, 1])**2 ).sum()**0.5, np.abs(num_arr[:, 1]).sum() )
    gauss_sigma = pfit[-5]
    lorentz_gamma = fixed_gamma
    peak_pos = pfit[1:-5:2]
    return final_profile, gauss_sigma, lorentz_gamma, peak_pos,  GOF

class XasCore(QObject):
    cleared = pyqtSignal()
    loaded = pyqtSignal()
    fileparams = pyqtSignal(object)
    logmessage = pyqtSignal(str)
    finished_fitting = pyqtSignal()
    finished_overplot = pyqtSignal()
    finished_rixsmap = pyqtSignal()
    finished_merge = pyqtSignal()
    finished_filter = pyqtSignal()
    finished_flux = pyqtSignal()
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
        self.current_interp = 'linear'
        self.interp_kinds = ['linear','slinear', 'quadratic', 'cubic', 'zero' ]
        self.xguess = []
        self.tey_profiles = []
        self.tpy_profiles = []
        self.raw_tey_profiles = []
        self.raw_tpy_profiles = []
        self.reduced_tey_profiles = []
        self.reduced_tpy_profiles = []
        self.reduced_raw_tey_profiles = []
        self.reduced_raw_tpy_profiles = []
        self.lastreduction = 1.0
        self.cutoff = 30
        self.binsize = 0.1
        self.fullnames = []
        self.orig_logs = []
        self.shortnames = []
        self.timedata = None
        self.prof_numbers = []
        self.mplot_curves = []
        self.mplot_raw_curves = []
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
        self.flux_curves = []
        self.flux_labels = []
        self.filter_override = ["Inverse units", "Fourier Transform"]
        self.map2Dplotax = [(1, 2048,),  (1, 2048)]
        self.prof_count = 0
        self.legpos = 0
        self.retvals = []
        xas = XasCorrector()
        xas.readCurves(resource_path('FluxCorrectionHarm1.txt'), harmnum=1)
        xas.readCurves(resource_path('FluxCorrectionHarm3.txt'), harmnum=3)
        self.xas_corrector = xas
    def assign_boxes(self, boxes):
        self.boxes = boxes
        for db in boxes:
            names, values = db.returnValues()
            self.parlist += names
            for nam in names:
                self.pardict[nam] = values[nam]
            db.values_changed.connect(self.make_params_visible)
        self.make_params_visible()
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
            elif 'binsize' in k:
                self.binsize = val
            elif 'cutoff' in k:
                self.cutoff = val
    def reduce_profiles(self):
        if (len(self.tey_profiles) == len(self.reduced_tey_profiles) and
            len(self.tpy_profiles) == len(self.reduced_tpy_profiles) and
            len(self.raw_tey_profiles) == len(self.reduced_raw_tey_profiles) and
            len(self.raw_tpy_profiles) == len(self.reduced_raw_tpy_profiles)):
            if self.redfac == self.lastreduction:
                return False
        self.reduced_tey_profiles = []
        self.reduced_tpy_profiles = []
        self.reduced_raw_tey_profiles = []
        self.reduced_raw_tpy_profiles = []
        if self.redfac == 1.0:
            for nump, p in enumerate(self.tey_profiles):
                if len(p) == 0:
                    self.reduced_tey_profiles.append(self.raw_tey_profiles[nump])
                else:
                    self.reduced_tey_profiles.append(p)
            for nump, p in enumerate(self.tpy_profiles):
                if len(p) == 0:
                    self.reduced_tpy_profiles.append(self.raw_tpy_profiles[nump])
                else:
                    self.reduced_tpy_profiles.append(p)
            for p in self.raw_tey_profiles:
                self.reduced_raw_tey_profiles.append(p)
            for p in self.raw_tpy_profiles:
                self.reduced_raw_tpy_profiles.append(p)
        else:
            for num, p in enumerate(self.tey_profiles):
                xax = self.xguess[num]
                if self.orig_logs[num] is None:
                    steps = len(p)
                    newsteps = int(round(steps/self.redfac))
                    target = np.zeros([newsteps, 3])
                    target[:, 0] = np.linspace(p[:, 0].min(), p[:, 0].max(), newsteps)
                    newone = merge2curves_errors(p,  target)
                    newone[:, 1:] /= self.redfac
                    self.reduced_tey_profiles.append(newone)
                else:
                    avglog, errlog = place_points_in_bins(copy.deepcopy(self.orig_logs[num]),
                                                                           redfac = self.redfac)
                    if len(avglog[xax]) > 0:
                        newone = np.column_stack([avglog[xax], avglog['CURR2'], errlog['CURR2']])
                    else:
                        newone = np.column_stack([avglog['TARGET'], avglog['CURR2'], errlog['CURR2']])
                    self.reduced_tey_profiles.append(newone)
            for num, p in enumerate(self.tpy_profiles):
                xax = self.xguess[num]
                if self.orig_logs[num] is None:
                    steps = len(p)
                    newsteps = int(round(steps/self.redfac))
                    target = np.zeros([newsteps, 3])
                    target[:, 0] = np.linspace(p[:, 0].min(), p[:, 0].max(), newsteps)
                    newone = merge2curves_errors(p,  target)
                    newone[:, 1:] /= self.redfac
                    self.reduced_tpy_profiles.append(newone)
                else:
                    avglog, errlog = place_points_in_bins(copy.deepcopy(self.orig_logs[num]),
                                                                            redfac = self.redfac)
                    if len(avglog[xax]) > 0:
                        newone = np.column_stack([avglog[xax], avglog['CURR1'], errlog['CURR1']])
                    else:
                        newone = np.column_stack([avglog['TARGET'], avglog['CURR2'], errlog['CURR2']])
                    self.reduced_tpy_profiles.append(newone)
            for p in self.raw_tey_profiles:
                self.reduced_raw_tey_profiles.append(p)
            for p in self.raw_tpy_profiles:
                self.reduced_raw_tpy_profiles.append(p)
        self.lastreduction = self.redfac
        return True
    @pyqtSlot(int)
    def setInterpolation(self, newnum):
        self.current_interp = self.interp_kinds[newnum]
        # self.rixs_axis_label = self.rixsaxes[newnum]
    @pyqtSlot()
    def take_table_values(self):
        self.retvals = self.table_obj.return_values()
    def logger(self,  message):
        now = time.gmtime()
        timestamp = ("ProfileCore "
                     +"-".join([str(tx) for tx in [now.tm_mday, now.tm_mon, now.tm_year]])
                     + ',' + ":".join([str(ty) for ty in [now.tm_hour, now.tm_min, now.tm_sec]]) + '| ')
        self.logmessage.emit(timestamp + message)
    @pyqtSlot(object)
    def load_profiles(self,  flist):
        fnames, snames, tey_profiles, tpy_profiles, raw_tey_profiles, raw_tpy_profiles = [], [], [], [], [], []
        original_logs = []
        xguesses = []
        for fnum, fname in enumerate(flist):
            self.temp_path, short_name = os.path.split(fname)
            if 'txt' in short_name[-4:]:
                try:
                    envals, tey, tpy = read_1D_xas(fname)
                except:
                    self.logger("Could not parse file:" + str(fname))
                    continue
                else:
                    self.prof_numbers.append(self.prof_count+fnum)
                    fnames.append(fname)
                    xguesses.append('E')
                    snames.append(short_name)
                    tey_profiles.append(np.column_stack([envals, tey, np.zeros(envals.shape)]))
                    tpy_profiles.append(np.column_stack([envals, tpy, np.zeros(envals.shape)]))
                    raw_tey_profiles.append(np.column_stack([envals, tey, np.zeros(envals.shape)]))
                    raw_tpy_profiles.append(np.column_stack([envals, tpy, np.zeros(envals.shape)]))
                    original_logs.append(None)
            else:
                try:
                    varlog = load_only_logs([fname])
                    # xguess = guess_XAS_xaxis(varlog)
                    avglog, errlog, xguess = load_and_average_logs([fname])
                except:
                    self.logger("Could not parse file:" + str(fname))
                else:
                    envals = varlog[xguess]
                    xguesses.append(xguess)
                    self.prof_numbers.append(self.prof_count+fnum)
                    fnames.append(fname)
                    snames.append(short_name)
                    if len(avglog[xguess]) > 0:
                        tey_profiles.append(np.column_stack([avglog[xguess], avglog['CURR2'], errlog['CURR2']]))
                        tpy_profiles.append(np.column_stack([avglog[xguess], avglog['CURR1'], errlog['CURR1']]))
                    else:
                        tey_profiles.append(np.column_stack([avglog['TARGET'], avglog['CURR2'], errlog['CURR2']]))
                        tpy_profiles.append(np.column_stack([avglog['TARGET'], avglog['CURR1'], errlog['CURR1']]))
                    raw_tey_profiles.append(np.column_stack([varlog[xguess],varlog['CURR2'], np.zeros(envals.shape)]))
                    raw_tpy_profiles.append(np.column_stack([varlog[xguess], varlog['CURR1'], np.zeros(envals.shape)]))
                    original_logs.append(varlog)
        self.fullnames += fnames
        self.xguess += xguesses
        self.shortnames += snames
        self.tey_profiles += tey_profiles
        self.tpy_profiles += tpy_profiles
        self.raw_tey_profiles += raw_tey_profiles
        self.raw_tpy_profiles += raw_tpy_profiles
        self.prof_count += len(fnames)
        self.orig_logs += original_logs
        self.loaded.emit()
        self.fileparams.emit([snames, tey_profiles, tpy_profiles, raw_tey_profiles, raw_tpy_profiles])
        return snames, tey_profiles, tpy_profiles, raw_tey_profiles, raw_tpy_profiles
    @pyqtSlot()
    def clear_profiles(self):
        self.fullnames = []
        self.shortnames = []
        self.tey_profiles = []
        self.tpy_profiles = []
        self.raw_tey_profiles = []
        self.raw_tpy_profiles = []
        self.orig_logs = []
        self.xguess = []
        self.reduced_tey_profiles = []
        self.reduced_tpy_profiles = []
        self.reduced_raw_tey_profiles = []
        self.reduced_raw_tpy_profiles = []
        self.timedata = None
        self.prof_numbers = []
        self.orig_logs = []
        self.prof_count = 0
        self.cleared.emit()  
    def manual_merge(self):
        self.mergeworked = False
        if len(self.retvals) < 1:
            self.finished_merge.emit()
            return None
        else:
            self.reduce_profiles()
            nums, xmin, xmax,  usetey, usetpy, names = [], [], [], [], [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                xmin.append(self.retvals[nr][1])
                xmax.append(self.retvals[nr][2])
                usetey.append(self.retvals[nr][3])
                usetpy.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
            nums =np.array(nums)
            xmin =np.array(xmin)
            xmax =np.array(xmax)
            othermin,  othermax = np.min(self.cuts),  np.max(self.cuts)
            fwhm_guess = 0.05
            minx, maxx, stepx = 1e5,-1e5,-1.0    
            curves,  curves2 = [],  []
            rcurves,  rcurves2 = [], []
            labels = []
            for rn, num in enumerate(nums):
                if not usetey[rn]:
                    continue
                dat = self.reduced_tey_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                dat = dat[np.where(dat[:, 0] > tempmin)]
                dat = dat[np.where(dat[:, 0] < tempmax)]
                minx = min(minx, dat[:,0].min())
                maxx = max(maxx, dat[:,0].max())
                step = (dat[1:,0] - dat[:-1,0]).mean()
                stepx = max(stepx, step)
                curves.append(dat)
                rcurves.append(self.reduced_raw_tey_profiles[num])
            for rn, num in enumerate(nums):
                if not usetpy[rn]:
                    continue
                dat = self.reduced_tpy_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                dat = dat[np.where(dat[:, 0] > tempmin)]
                dat = dat[np.where(dat[:, 0] < tempmax)]
                minx = min(minx, dat[:,0].min())
                maxx = max(maxx, dat[:,0].max())
                step = (dat[1:,0] - dat[:-1,0]).mean()
                stepx = max(stepx, step)
                curves2.append(dat)
                rcurves2.append(self.reduced_raw_tpy_profiles[num])
            newx = np.arange(minx, maxx + 0.1*stepx, stepx)
            target = np.zeros((len(newx),3))
            target[:,0] = newx
            for n in range(len(curves)):
                target = merge2curves_errors(curves[n],target)
            target2 = np.zeros((len(newx),3))
            target2[:,0] = newx
            for n in range(len(curves2)):
                target2 = merge2curves_errors(curves2[n],target2)
            self.merged_curve = target
            raw1 = np.row_stack(rcurves)
            raw2 = np.row_stack(rcurves2)
            raw1 = raw1[np.argsort(raw1[:, 0])]
            raw2 = raw2[np.argsort(raw2[:, 0])]
            self.mergeworked = True
            # now we try to add the fitting            
            self.fullnames += ["No file"]
            self.shortnames += ["Merged data"]
            self.tey_profiles += [target]
            self.tpy_profiles += [target2]
            self.raw_tey_profiles += [raw1]
            self.raw_tpy_profiles += [raw2]
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
        nums, xmin, xmax, usetey, usetpy, names = [], [], [], [], [], []
        for nr in range(len(self.retvals)):
            nums.append(self.retvals[nr][0])
            xmin.append(self.retvals[nr][1])
            xmax.append(self.retvals[nr][2])
            usetey.append(self.retvals[nr][3])
            usetpy.append(self.retvals[nr][4])
            names.append(self.retvals[nr][5])
        nums =np.array(nums)
        xmin =np.array(xmin)
        xmax =np.array(xmax)
        othermin,  othermax = np.min(self.cuts),  np.max(self.cuts)
        curves = []
        labels = []   
        for rn, num in enumerate(nums):
            if not usetey[rn]:
                continue
            temp = self.reduced_tey_profiles[num].copy()
            tempmin = max(othermin, xmin[rn])
            tempmax = min(othermax, xmax[rn])
            temp = temp[np.where(temp[:, 0] > tempmin)]
            temp = temp[np.where(temp[:, 0] < tempmax)]
            curves.append(temp)
            labels.append('Saved_TEY_from_' + names[rn])  
        for rn, num in enumerate(nums):
            if not usetpy[rn]:
                continue
            temp = self.reduced_tpy_profiles[num].copy()
            tempmin = max(othermin, xmin[rn])
            tempmax = min(othermax, xmax[rn])
            temp = temp[np.where(temp[:, 0] > tempmin)]
            temp = temp[np.where(temp[:, 0] < tempmax)]
            curves.append(temp)
            labels.append('Saved_TPY_from_' + names[rn])    
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
            nums, xmin, xmax, usetey, usetpy, names = [], [], [], [], [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                xmin.append(self.retvals[nr][1])
                xmax.append(self.retvals[nr][2])
                usetey.append(self.retvals[nr][3])
                usetpy.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
            nums =np.array(nums)
            xmin =np.array(xmin)
            xmax =np.array(xmax)
            othermin,  othermax = np.min(self.cuts),  np.max(self.cuts)
            curves = []
            labels = []   
            for rn, num in enumerate(nums):
                if not usetey[rn]:
                    continue
                temp = self.reduced_tey_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TEY from ' + names[rn])         
            for rn, num in enumerate(nums):
                if not usetpy[rn]:
                    continue
                temp = self.reduced_tpy_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TPY from ' + names[rn])
            for n in range(len(curves)):
                xaxis = curves[n][:,0]
                xstep = (xaxis[1:] - xaxis[:-1]).mean()
                # new_x = fftfreq(len(xaxis), xstep)
                new_y = rfft(curves[n][:, 1])
                new_x = np.arange(len(new_y)) + 1
                # norm = curves[n][:,1].sum()
                curves[n] = np.column_stack([new_x, new_y, np.zeros(len(new_x))])
                # curves[n] = curves[n][np.where(curves[n][:,1] > 0.0)]
                # curves[n][:,1] *= 100.0
            self.mplot_curves = curves
            self.mplot_labels = labels
            self.mplot_override = ["Inverse units", "Fourier Transform"]
            self.mplot_raw_curves = []
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
            nums, xmin, xmax, usetey, usetpy, names = [], [], [], [], [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                xmin.append(self.retvals[nr][1])
                xmax.append(self.retvals[nr][2])
                usetey.append(self.retvals[nr][3])
                usetpy.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
            nums =np.array(nums)
            xmin =np.array(xmin)
            xmax =np.array(xmax)
            othermin,  othermax = np.min(self.cuts),  np.max(self.cuts)
            curves = []
            labels = []
            tags = []
            teylist = []
            tpylist = []
            taggeddict = {}
            firsttpy = 0
            for rn, num in enumerate(nums):
                if not usetey[rn]:
                    continue
                temp = self.reduced_tey_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TEY from ' + names[rn])      
                teylist.append(names[rn])
                tags.append(names[rn])
                taggeddict[names[rn]] = [None, None]
                firsttpy = rn+1
            for rn, num in enumerate(nums):
                if not usetpy[rn]:
                    continue
                temp = self.reduced_tpy_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TPY from ' + names[rn])
                tpylist.append(names[rn])
                tags.append(names[rn])
                taggeddict[names[rn]] = [None, None]
            for n in range(len(curves)):
                xaxis = curves[n][:,0]
                xstep = (xaxis[1:] - xaxis[:-1]).mean()
                new_x = fftfreq(len(xaxis), xstep)
                new_y = rfft(curves[n][:, 1])
                new_y[-self.cutoff:] = 0.0
                result = irfft(new_y)
                # norm = curves[n][:,1].sum()
                curves[n] = np.column_stack([xaxis, result,  np.zeros(len(xaxis))])
                labels[n] = 'Filtered_' + labels[n]
                if n < firsttpy:
                    taggeddict[tags[n]][0] = np.column_stack([xaxis, result,  np.zeros(len(xaxis))])
                else:
                    taggeddict[tags[n]][1] = np.column_stack([xaxis, result,  np.zeros(len(xaxis))])
                # curves[n] = curves[n][np.where(curves[n][:,1] > 0.0)]
                # curves[n][:,1] *= 100.0
            self.filter_curves = []
            self.filter_labels = []
            self.filter_units = []
            self.filter_units.append("Energy [eV]")
            for tag in np.unique(tags):
                # tag = tags[n]
                print("Filtering: tag ", tag)
                self.fullnames += ["No file"]
                self.shortnames += ['Filtered '+tag]
                teyprof = taggeddict[tag][0]
                tpyprof = taggeddict[tag][1]
                if teyprof is not None:
                    self.tey_profiles += [teyprof]
                    self.raw_tey_profiles += [teyprof]
                else:
                    xax = tpyprof[:, 0]
                    self.tey_profiles += [np.column_stack([xax, np.zeros((len(xax), 2))])]
                    self.raw_tey_profiles += [np.column_stack([xax, np.zeros((len(xax), 2))])]
                if tpyprof is not None:
                    self.tpy_profiles += [tpyprof]
                    self.raw_tpy_profiles += [tpyprof]
                else:
                    xax = teyprof[:, 0]
                    self.tpy_profiles += [np.column_stack([xax, np.zeros((len(xax), 2))])]
                    self.raw_tpy_profiles += [np.column_stack([xax, np.zeros((len(xax), 2))])]
                self.filter_curves.append([self.tey_profiles[-1], self.tpy_profiles[-1]])
                self.prof_count += 1
                self.filter_labels.append('Filtered '+tag)
            self.mplot_curves = curves
            self.mplot_raw_curves = []
            self.mplot_labels = labels
            self.mplot_override = ["Inverse units", "Fourier Transform"]
            self.overplotworked = True
            self.finished_filter.emit()
            return "Done"
    def flux_correction(self):
        self.flux_curves = []
        self.flux_labels = []
        self.flux_units = []
        self.flux_energies = []
        self.overplotworked = False
        if len(self.retvals) < 1:
            self.finished_flux.emit()
            return None
        else:
            self.reduce_profiles()
            nums, xmin, xmax, usetey, usetpy, names = [], [], [], [], [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                xmin.append(self.retvals[nr][1])
                xmax.append(self.retvals[nr][2])
                usetey.append(self.retvals[nr][3])
                usetpy.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
            nums =np.array(nums)
            xmin =np.array(xmin)
            xmax =np.array(xmax)
            othermin,  othermax = np.min(self.cuts),  np.max(self.cuts)
            curves = []
            labels = []
            tags = []
            teylist = []
            tpylist = []
            taggeddict = {}
            firsttpy = 0
            for rn, num in enumerate(nums):
                if not usetey[rn]:
                    continue
                temp = self.reduced_tey_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TEY from ' + names[rn])      
                teylist.append(names[rn])
                tags.append(names[rn])
                taggeddict[names[rn]] = [None, None]
                firsttpy = rn+1
            for rn, num in enumerate(nums):
                if not usetpy[rn]:
                    continue
                temp = self.reduced_tpy_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TPY from ' + names[rn])
                tpylist.append(names[rn])
                tags.append(names[rn])
                taggeddict[names[rn]] = [None, None]
            for n in range(len(curves)):
                xaxis = curves[n][:,0]
                # xstep = (xaxis[1:] - xaxis[:-1]).mean()
                fcorr = self.xas_corrector.returnInterpolatedCurve(xaxis, kind_notcovered = self.current_interp)
                fcrit = fcorr[:, 1] == 0.0
                new_y = curves[n][:, 1]/fcorr[:, 1] * 1.602176487 * 10**(-19)
                new_y[fcrit] = 0.0
                # norm = curves[n][:,1].sum()
                curves[n] = np.column_stack([xaxis, new_y,  np.zeros(len(xaxis))])
                labels[n] = 'FluxCorrected_' + labels[n]
                if n < firsttpy:
                    taggeddict[tags[n]][0] = np.column_stack([xaxis, new_y,  np.zeros(len(xaxis))])
                else:
                    taggeddict[tags[n]][1] = np.column_stack([xaxis, new_y,  np.zeros(len(xaxis))])
                # curves[n] = curves[n][np.where(curves[n][:,1] > 0.0)]
                # curves[n][:,1] *= 100.0
            self.flux_curves = []
            self.flux_labels = []
            self.flux_units = []
            self.flux_units.append("Energy [eV]")
            for tag in np.unique(tags):
                # tag = tags[n]
                print("Correcting: tag ", tag)
                self.fullnames += ["No file"]
                self.shortnames += ['FluxCorrected '+tag]
                teyprof = taggeddict[tag][0]
                tpyprof = taggeddict[tag][1]
                if teyprof is not None:
                    self.tey_profiles += [teyprof]
                    self.raw_tey_profiles += [teyprof]
                else:
                    xax = tpyprof[:, 0]
                    self.tey_profiles += [np.column_stack([xax, np.zeros((len(xax), 2))])]
                    self.raw_tey_profiles += [np.column_stack([xax, np.zeros((len(xax), 2))])]
                if tpyprof is not None:
                    self.tpy_profiles += [tpyprof]
                    self.raw_tpy_profiles += [tpyprof]
                else:
                    xax = teyprof[:, 0]
                    self.tpy_profiles += [np.column_stack([xax, np.zeros((len(xax), 2))])]
                    self.raw_tpy_profiles += [np.column_stack([xax, np.zeros((len(xax), 2))])]
                self.flux_curves.append([self.tey_profiles[-1], self.tpy_profiles[-1]])
                self.prof_count += 1
                self.flux_labels.append('FluxCorrected '+tag)
            self.mplot_curves = curves
            self.mplot_raw_curves = []
            self.mplot_labels = labels
            self.mplot_override = ["Energy [eV]", "arb. units"]
            self.overplotworked = True
            self.finished_flux.emit()
            return "Done"
    def multiplot(self):
        self.overplotworked = False
        if len(self.retvals) < 1:
            self.finished_overplot.emit()
            return None
        else:
            self.reduce_profiles()
            nums, xmin, xmax, names = [], [], [], []
            usetey,  usetpy = [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                xmin.append(self.retvals[nr][1])
                xmax.append(self.retvals[nr][2])
                usetey.append(self.retvals[nr][3])
                usetpy.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
            nums =np.array(nums)
            xmin =np.array(xmin)
            xmax =np.array(xmax)
            othermin,  othermax = np.min(self.cuts),  np.max(self.cuts)
            curves = []
            raw_curves = []
            labels = []
            for rn, num in enumerate(nums):
                if not usetey[rn]:
                    continue
                temp = self.reduced_tey_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TEY from ' + names[rn])
            for rn, num in enumerate(nums):
                if not usetey[rn]:
                    continue
                temp = self.reduced_raw_tey_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                raw_curves.append(temp)
                # step = cur[np.where( cur[:,1] < np.percentile(cur[:,1], 60.0) )][:,1].std()
            for rn, num in enumerate(nums):
                if not usetpy[rn]:
                    continue
                temp = self.reduced_tpy_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TPY from ' + names[rn])
            for rn, num in enumerate(nums):
                if not usetpy[rn]:
                    continue
                temp = self.reduced_raw_tpy_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                raw_curves.append(temp)
            for n in range(len(curves)):
                xaxis = curves[n][:,0]
                xstep = xaxis[1:] - xaxis[:-1]
                xstep = np.concatenate([xstep[:1], xstep])
                norm = (np.abs(curves[n][:,1]) * xstep).sum()
                # norm = curves[n][:,1].sum()
                curves[n][:,1:] /= norm
                shift = np.percentile(curves[n][:,1], self.bkg_perc) - 0.01
                curves[n][:,1] -= shift
                raw_curves[n][:, 1:] /= norm
                raw_curves[n][:, 1] -= shift
                # curves[n] = curves[n][np.where(curves[n][:,1] > 0.0)]
                # curves[n][:,1] *= 100.0
            self.mplot_curves = curves
            self.mplot_raw_curves = raw_curves
            self.mplot_labels = labels
            self.mplot_override = ["Energy [eV]", ""]
            self.overplotworked = True
            self.finished_overplot.emit()
            return "Done"
    def absoluteplot(self):
        self.overplotworked = False
        if len(self.retvals) < 1:
            self.finished_overplot.emit()
            return None
        else:
            self.reduce_profiles()
            nums, xmin, xmax, names = [], [], [], []
            usetey,  usetpy = [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                xmin.append(self.retvals[nr][1])
                xmax.append(self.retvals[nr][2])
                usetey.append(self.retvals[nr][3])
                usetpy.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
            nums =np.array(nums)
            xmin =np.array(xmin)
            xmax =np.array(xmax)
            othermin,  othermax = np.min(self.cuts),  np.max(self.cuts)
            curves = []
            raw_curves = []
            labels = []
            for rn, num in enumerate(nums):
                if not usetey[rn]:
                    continue
                temp = self.reduced_tey_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TEY from ' + names[rn])
            for rn, num in enumerate(nums):
                if not usetey[rn]:
                    continue
                temp = self.reduced_raw_tey_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                raw_curves.append(temp)
                # step = cur[np.where( cur[:,1] < np.percentile(cur[:,1], 60.0) )][:,1].std()
            for rn, num in enumerate(nums):
                if not usetpy[rn]:
                    continue
                temp = self.reduced_tpy_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                curves.append(temp)
                labels.append('TPY from ' + names[rn])
            for rn, num in enumerate(nums):
                if not usetpy[rn]:
                    continue
                temp = self.reduced_raw_tpy_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                raw_curves.append(temp)
            self.mplot_curves = curves
            self.mplot_raw_curves = raw_curves
            self.mplot_labels = labels
            self.mplot_override = ["Energy [eV]", ""]
            self.overplotworked = True
            self.finished_overplot.emit()
            return "Done"
    def many_as_one(self):
        self.overplotworked = False
        if len(self.retvals) < 1:
            self.finished_overplot.emit()
            return None
        else:
            # self.reduce_profiles()
            nums, xmin, xmax, names = [], [], [], []
            usetey,  usetpy = [], []
            for nr in range(len(self.retvals)):
                nums.append(self.retvals[nr][0])
                xmin.append(self.retvals[nr][1])
                xmax.append(self.retvals[nr][2])
                usetey.append(self.retvals[nr][3])
                usetpy.append(self.retvals[nr][4])
                names.append(self.retvals[nr][5])
            nums =np.array(nums)
            xmin =np.array(xmin)
            xmax =np.array(xmax)
            othermin,  othermax = np.min(self.cuts),  np.max(self.cuts)
            curves = []
            raw_curves = []
            labels = []
            for rn, num in enumerate(nums):
                if not usetey[rn]:
                    continue
                temp = self.raw_tey_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                raw_curves.append(temp)
                labels.append('TEY from ' + names[rn])
                # step = cur[np.where( cur[:,1] < np.percentile(cur[:,1], 60.0) )][:,1].std()
            for rn, num in enumerate(nums):
                if not usetpy[rn]:
                    continue
                temp = self.raw_tpy_profiles[num].copy()
                tempmin = max(othermin, xmin[rn])
                tempmax = min(othermax, xmax[rn])
                temp = temp[np.where(temp[:, 0] > tempmin)]
                temp = temp[np.where(temp[:, 0] < tempmax)]
                raw_curves.append(temp)
                labels.append('TPY from ' + names[rn])
            total_points = np.row_stack(raw_curves)
            total_points = total_points[np.argsort(total_points[:, 0])]
            if self.binsize > 0.0:
                binsize = self.binsize
            else:
                binsize = 0.1
            new_x = np.arange(total_points[0, 0] - 0.5*binsize, total_points[-1, 0] + 0.51*binsize, binsize)
            result = place_data_in_bins(total_points, new_x)
            self.mplot_curves = [result]
            self.mplot_raw_curves = raw_curves
            self.mplot_labels = ["All scans, averaged"] + labels
            self.mplot_override = ["Energy [eV]", ""]
            self.merged_curve = result
            self.overplotworked = True   
            self.fullnames += ["No file"]
            self.shortnames += ["Merged data"]
            self.tey_profiles += [result]
            self.tpy_profiles += [result]
            self.raw_tey_profiles += [total_points]
            self.raw_tpy_profiles += [total_points]
            self.prof_count += 1
            self.mergeworked = True   
            self.finished_merge.emit()
            return "Done"

    
def single_flex_profile_cf(params, xarr, yarr, errarr = None, npeaks = 0, 
                                            bkg_order = 0, include_edge=False, 
                                            penalty = 10.0):
    # profile = np.zeros(xarr.shape)
    # for n in range(npeaks):
    #     profile += fast_voigt(xarr, abs(params[n]), params[n+npeaks], abs(params[n+2*npeaks]), abs(params[n+3*npeaks]))
    # profile += single_background(params[npeaks*4:], xarr, bkg_order, include_edge)
    profile = single_flex_profile(params, xarr, yarr, errarr, npeaks, 
                                            bkg_order, include_edge, 
                                            penalty)
    cf = costfunction_no_overshoot(profile, yarr, errarr, penalty)
    # rsq = Rsquared(profile, yarr)
    # chisq = Chisquared(profile, yarr)
    # return cf * (1.0 + abs(rsq -1)) * (1.0 + abs(chisq-1.0))
    return cf #  * (1.0 + abs(rsq -1)) * (1.0 + abs(chisq-1.0))
    
def single_flex_profile(params, xarr, yarr, errarr = None, npeaks = 0, 
                                            bkg_order = 0, include_edge=False, 
                                            penalty = 10.0):
    profile = np.zeros(xarr.shape)
    for n in range(npeaks):
        profile += asym_voigt(xarr, abs(params[n]), params[n+npeaks], abs(params[n+2*npeaks]), abs(params[n+3*npeaks]), params[-1])
    profile += single_background(params[npeaks*4:-1], xarr, bkg_order, include_edge)
    return profile

def single_background(params,  xarr,  bkg_order = 0, include_edge = False):
    new_y = np.zeros(xarr.shape)
    for bo  in np.arange(bkg_order+1):
        if bo == 0:
            new_y += params[bo]
        else:
            new_y += params[bo]*(xarr**bo)
    if include_edge:
        new_y += params[bkg_order]*np.arctan((xarr-params[bkg_order+1])/params[bkg_order+2])
    return new_y

def costfunction_no_overshoot(new_y, yarr, errarr = None, penalty = 10.0):
    least = (new_y-yarr)**2
    least[np.where(new_y > yarr)] *= penalty
    if errarr is not None:
        abserr = np.abs(errarr)
        errspan = abserr.max() - abserr.min()
        if errspan > 0.0:
            least /= abserr + 0.01*errspan
    return least.sum()

def costfunction_no_overshoot_for_leastsq(new_y, yarr, errarr = None, penalty = 10.0):
    least = (new_y-yarr)**2
    least[np.where(new_y > yarr)] *= penalty
    if errarr is not None:
        abserr = np.abs(errarr)
        errspan = abserr.max() - abserr.min()
        if errspan > 0.0:
            least /= abserr + 0.01*errspan
    return least

def Rsquared(new_y, yarr):
    y_mean = yarr.mean()
    sstot = (yarr-y_mean)**2
    ssres = (new_y-yarr)**2
    rsquared = 1 - ssres.sum()/sstot.sum()
    return rsquared

def smooth_curve(yarr,  errarr = None):
    new_y = yarr.copy()
    length = len(new_y)
    lastind = length -1
    for ind in range(length):
        val = yarr[ind]
        for ext in range(1, 9, 2):
            temp = yarr[max(ind-ext, 0):min(ind+ext, lastind)]
            meanv = temp.mean()
            stdv = temp.std()
            if abs(val-meanv) < 0.3*stdv:
                new_y[ind] = meanv
            else:
                break
    return new_y

def Chisquared(new_y, yarr, errarr = None, penalty = 10.0):
    numerator = (new_y-yarr)**2
    denominator = yarr.std()**2
    chisq = (numerator/denominator).sum()
    return chisq

def fwhm_voigt(params):
    fL = params[3]
    fG = params[2]
    fwhm = fL/2.0 + ((fL**2)/4 + fG**2)**0.5
    return fwhm

def fit_voigt(params, xarr, yarr, errarr = None,  penalty = 10.0):
    new_y = np.zeros(xarr.shape)
    new_y += fast_voigt(xarr, params[0], params[1], params[2], params[3])
    return costfunction_no_overshoot(new_y, yarr, errarr, penalty)
    
def fit_background(params, xarr, yarr, errarr = None,  penalty = 10.0, 
                                       bkg_order = 0, include_edge = False):
    new_y = np.zeros(xarr.shape)
    new_y += single_background(params,  xarr,  bkg_order, include_edge)
    return costfunction_no_overshoot(new_y, yarr, errarr, penalty)

def fit_background_leastsq(params, xarr, yarr, errarr = None,  penalty = 10.0, 
                                       bkg_order = 0, include_edge = False):
    new_y = np.zeros(xarr.shape)
    new_y += single_background(params,  xarr,  bkg_order, include_edge)
    return costfunction_no_overshoot_for_leastsq(new_y, yarr, errarr, penalty)

def global_fitting_optimiser(data, maxpeaks = 10, bkg_order = 0, include_edge = False,
                                                    one_gauss_fwhm = False,  one_lorentz_fwhm = False, 
                                                    overshoot_penalty = 10.0):
    ncol = data.shape[1]
    if ncol <2:
        print("Not enough data for fitting. Need at least x and y columns.")
        return None
    elif ncol <3:
        print("Fitting the profile using column 1 as x and column 2 as y.")
        errors = False
    elif ncol <4:
        print("Fitting the profile using column 1 as x, column 2 as y, and column 3 as y-error.")
        errors = True
    else:
        print("Too many columns. Try limiting the input to [x,y,yerr].")
        return None
    xarr = data[:, 0]
    yarr = data[:, 1]
    if errors:
        errarr = data[:, 2]
    else:
        errarr = None
    if include_edge:
        bkg_params = np.zeros(bkg_order+4)
    else:
        bkg_params = np.zeros(bkg_order+1)
    if one_gauss_fwhm is False:
        variable_Gauss = True
    else:
        if one_gauss_fwhm < 0:
            variable_Gauss = True
        else:
            variable_Gauss = False
            fixedGauss = one_gauss_fwhm
    if one_lorentz_fwhm is False:
        variable_Lorentz = True
    else:
        if one_lorentz_fwhm < 0:
            variable_Lorentz = True
        else:
            variable_Lorentz = False
            fixedLorentz = one_lorentz_fwhm
    bkg_params[0] = yarr.min()
    output,  summary = {}, {}
    print("# N_PEAKS  costfunction  R2  Chi2")
    for peakn in np.arange(maxpeaks):
        peak_params = []
        pfit, pcov, infodict, errmsg, success = leastsq(fit_background_leastsq, bkg_params, 
                                                args = (xarr, yarr.copy(), errarr, 100.0, bkg_order, include_edge), 
                                                full_output = 1)
        new_y = single_background(pfit,  xarr,  bkg_order, include_edge)
        temp_y = yarr - new_y
        Y_span = temp_y.max() - temp_y.min()
        X_span = xarr.max() - xarr.min()
        last_gwidth = 0.1
        last_lwidth = 0.1
#        for p in np.arange(peakn):
#            pheight = temp_y.max()
#            pcentre = xarr[np.where(temp_y == pheight)].mean()
#            pfit, pcov, infodict, errmsg, success = leastsq(fit_voigt, [pheight, pcentre, last_gwidth, last_lwidth], 
#                                                           args = (xarr, yarr, errarr, 100.0), 
#                                                           full_output = 1)
#            temp_y -= my_voigt(xarr, pfit[0], pfit[1], pfit[2], pfit[3])
#            peak_params.append([pfit[0], pfit[1], pfit[2], pfit[3]])
        peakc_bounds = (xarr.min() - 0.1*X_span, xarr.max() + 0.1*X_span)
        upperlimit = Y_span*X_span/2.0
        if variable_Gauss:
            peakg_bounds = (0, X_span/2.0)
            upperlimit *= X_span/2.0
        else:
            peakg_bounds = (0, fixedGauss)
            upperlimit *= fixedGauss
        if variable_Lorentz:
            peakl_bounds = (0, X_span/2.0)
            upperlimit *= X_span/2.0
        else:
            peakl_bounds = (0, fixedLorentz)
            upperlimit *= fixedLorentz
        peakh_bounds = (0, upperlimit)
        bounds = []
        for p in np.arange(peakn):
            bounds += [peakh_bounds]
        for p in np.arange(peakn):
            bounds += [peakc_bounds]
        for p in np.arange(peakn):
            bounds += [peakg_bounds]
        for p in np.arange(peakn):
            bounds += [peakl_bounds]
        for bo in np.arange(bkg_order+1):
            if bo == 0:
                bounds += [(yarr.min() - 0.3*Y_span, yarr.min()+0.3*Y_span)]
            else:
                bounds += [(-0.3*abs(Y_span)**(1/bo), 0.3*abs(Y_span)**(1/bo))]
        if include_edge:
            bounds += [(-Y_span, Y_span), (xarr.min(), xarr.max()), (0.0, X_span/2.0)]
        print(bounds)
        results = shgo(single_flex_profile_cf, bounds, 
                                     args = ( xarr, yarr, errarr, peakn, 
                                            bkg_order, include_edge, 
                                            overshoot_penalty)  ,
                                            # iters=3,  sampling_method='sobol')
                                            )
        test = single_flex_profile(results['x'], xarr, yarr, errarr, peakn, 
                                            bkg_order, include_edge, 
                                            overshoot_penalty)
        costfunc = costfunction_no_overshoot(test, yarr, errarr = None, penalty = overshoot_penalty)
        rsq = Rsquared(test, yarr)
        chisq = Chisquared(test, yarr)
        print(peakn, costfunc, rsq,  chisq)
        summary[peakn] = [costfunc, rsq, chisq]
        # print("At N_PEAKS=",  peakn, " the result was ", costfunc, " with parameters:\n", results['x'])
        output[peakn] = (np.column_stack([xarr, test]), results['x'])
    return output,  summary

def primitive_rebin(inarray,  redrate = 5):
    tlen = len(inarray)
    nrow = int(math.floor(tlen / redrate))
    newarray = inarray[:nrow*redrate].reshape((nrow, redrate))
    lastone = inarray[nrow*redrate:]
    result = np.concatenate([newarray.sum(1), [lastone.sum()]])
    return result

def section_search(inarray, seglimit = 4,  coarseness =5):
    tot_elements = len(inarray)
    xarray = np.arange(tot_elements)
    workarray = inarray.copy()
    curr_elements = len(workarray)
    ranges = []
    multiplier = 1
    while curr_elements > seglimit:
        maxpos = np.argmax(workarray)
        ranges.append((maxpos*multiplier, (maxpos+1)*multiplier))
        workarray = primitive_rebin(workarray, redrate = coarseness)
        curr_elements = len(workarray)
        multiplier *= coarseness
    return ranges

def iterative_fitting_optimiser(data, maxpeaks = 10, bkg_order = 0, include_edge = False,
                                                    one_gauss_fwhm = False,  one_lorentz_fwhm = False, 
                                                    overshoot_penalty = 10.0):
    ncol = data.shape[1]
    if ncol <2:
        print("Not enough data for fitting. Need at least x and y columns.")
        return None
    elif ncol <3:
        print("Fitting the profile using column 1 as x and column 2 as y.")
        errors = False
    elif ncol <4:
        print("Fitting the profile using column 1 as x, column 2 as y, and column 3 as y-error.")
        errors = True
    else:
        print("Too many columns. Try limiting the input to [x,y,yerr].")
        return None
    xarr = data[:, 0]
    step = (xarr[1:]-xarr[:-1]).mean()
    yarr = data[:, 1]
    if errors:
        errarr = data[:, 2]
    else:
        errarr = None
    if include_edge:
        bkg_params = np.zeros(bkg_order+4)
    else:
        bkg_params = np.zeros(bkg_order+1)
    bkg_params[0] = yarr.min()
    output,  summary = {},  {}
    # print("# N_PEAKS  costfunction  R2  Chi2")
    print("Fitting with maxpeaks=", maxpeaks)
    for peakn in np.arange(maxpeaks):
        peak_params = []
        result = minimize(fit_background, bkg_params, 
                                                args = (xarr, yarr.copy(), errarr,
                                                100*overshoot_penalty,
                                                bkg_order, include_edge))
        new_bkg_params = result['x']
        new_y = single_background(result['x'],  xarr,  bkg_order, include_edge)
        # temp_y = smooth_curve(yarr) - new_y
        temp_y = yarr - new_y
        temp_xarr = xarr.copy()
        if errarr is not None:
            temp_err = errarr.copy()
        else:
            temp_err = None
        last_gwidth = 2*step
        last_lwidth = 2*step
        for p in np.arange(peakn):
            testranges = section_search(temp_y)
            fits, intensities = [], []
            for tr in testranges:
                reduced_y = temp_y[tr[0]:tr[1]]
                reduced_x = temp_xarr[tr[0]:tr[1]]
                if len(reduced_y) < 3:
                    continue
                ind_centre = np.argmax(reduced_y)
                pheight = reduced_y[ind_centre]
                pcentre = reduced_x[ind_centre]
                low_fwhm,  high_fwhm = ind_centre,  ind_centre
                y1_fwhm,  y2_fwhm = pheight, pheight
                while y1_fwhm >= 0.49*pheight and low_fwhm > 0:
                    low_fwhm -= 1
                    y1_fwhm = reduced_y[low_fwhm]
                while y2_fwhm >= 0.49*pheight and high_fwhm < len(reduced_y) -1:
                    high_fwhm += 1
                    y2_fwhm = reduced_y[high_fwhm]
                width = abs(reduced_x[high_fwhm] - reduced_x[low_fwhm])
                if width > 0:
                    last_gwidth = width / (0.5 + 1.25**0.5)
                    last_lwidth = last_gwidth
                else:
                    last_gwidth = step/2.0
                    last_lwidth = step/2.0
                result = minimize(fit_voigt, [pheight, pcentre, last_gwidth, last_lwidth], 
                                                           args = (temp_xarr, temp_y, temp_err,
                                                           overshoot_penalty
                                                           ))
                pfit = result['x']
                # pfit = [pheight, pcentre, last_gwidth/2.0, last_lwidth/2.0]
                last_gwidth = pfit[2]
                last_lwidth = pfit[3]
                fits.append(pfit)
                intensities.append(fast_voigt(reduced_x, pfit[0], pfit[1], pfit[2], pfit[3]).sum()/(tr[1]-tr[0]))
            if len(intensities) == 0:
                continue
            pfit = fits[np.argmax(intensities)]
            last_gwidth = pfit[2]
            last_lwidth = pfit[3]
            temp_y -= fast_voigt(temp_xarr, pfit[0], pfit[1], pfit[2], pfit[3])
            peak_params += [pfit[0], pfit[1], pfit[2], pfit[3]]
#            ind_centre = np.argmax(temp_y)
#            pheight = temp_y[ind_centre]
#            pcentre = temp_xarr[ind_centre]
#            low_fwhm,  high_fwhm = ind_centre,  ind_centre
#            y1_fwhm,  y2_fwhm = pheight, pheight
#            while y1_fwhm >= 0.49*pheight and low_fwhm > 0:
#                low_fwhm -= 1
#                y1_fwhm = temp_y[low_fwhm]
#            while y2_fwhm >= 0.49*pheight and high_fwhm < len(temp_y) -1:
#                high_fwhm += 1
#                y2_fwhm = temp_y[high_fwhm]
#            width = abs(temp_xarr[high_fwhm] - temp_xarr[low_fwhm])
#            if width > 0:
#                last_gwidth = width / (0.5 + 1.25**0.5)
#                last_lwidth = last_gwidth
#            else:
#                last_gwidth = step/2.0
#                last_lwidth = step/2.0
#            print("Peak #", p, " initial guess: ", [pheight, pcentre, last_gwidth, last_lwidth])
#            # temp_fwhm = fwhm_voigt([pheight, pcentre, last_gwidth/2.0, last_lwidth/2.0])
#            result = minimize(fit_voigt, [pheight, pcentre, last_gwidth, last_lwidth], 
#                                                           args = (temp_xarr, temp_y, temp_err,
#                                                           overshoot_penalty
#                                                           ))
#            pfit = result['x']
#            # pfit = [pheight, pcentre, last_gwidth/2.0, last_lwidth/2.0]
#            last_gwidth = pfit[2]
#            last_lwidth = pfit[3]
#            temp_y -= fast_voigt(temp_xarr, pfit[0], pfit[1], pfit[2], pfit[3])
#            peak_params += [pfit[0], pfit[1], pfit[2], pfit[3]]
            # # optional trimming
            # temp_y = np.concatenate([temp_y[:low_fwhm],  temp_y[high_fwhm:]])
            # temp_xarr = np.concatenate([temp_xarr[:low_fwhm],  temp_xarr[high_fwhm:]])
            # if temp_err is not None:
            #     temp_err = np.concatenate([temp_err[:low_fwhm],  temp_err[high_fwhm:]])
        init_params = np.array([])
        init_params = np.concatenate([
        peak_params[0:4*peakn:4], peak_params[1:4*peakn:4], peak_params[2:4*peakn:4], peak_params[3:4*peakn:4]
        ])
#        for p in np.arange(peakn):
#            # init_params += [peak_params[p]]
#            init_params = np.concatenate([init_params, peak_params[p:p+1]])
#        for p in np.arange(peakn):
#            # init_params += [peak_params[peakn+p]]
#            init_params = np.concatenate([init_params, peak_params[peakn+p:peakn+p+1]])
#        for p in np.arange(peakn):
#            # init_params += [peak_params[2*peakn+p]]
#            init_params = np.concatenate([init_params, peak_params[2*peakn+p:2*peakn+p+1]])
#        for p in np.arange(peakn):
#            # init_params += [peak_params[3*peakn+p]]
#            init_params = np.concatenate([init_params, peak_params[3*peakn+p:3*peakn+p+1]])
        init_params = np.concatenate([init_params, new_bkg_params, [0]])
        print(peakn, init_params)
        result = minimize(single_flex_profile_cf, init_params, 
                                    args = ( xarr, yarr, errarr, peakn, 
                                           bkg_order, include_edge, 
                                           overshoot_penalty))
        test = single_flex_profile(result['x'], xarr, yarr, errarr, peakn, 
                                           bkg_order, include_edge, 
                                           overshoot_penalty)
        print("Asymmetry parameter: ", result['x'][-1])
        # test = single_flex_profile(init_params, xarr, yarr, errarr, peakn, 
        #                                     bkg_order, include_edge, 
        #                                     overshoot_penalty)
        costfunc = costfunction_no_overshoot(test, yarr, errarr = None, penalty = overshoot_penalty)
        rsq = Rsquared(test, yarr)
        chisq = Chisquared(test, yarr)
        # print(peakn, costfunc, rsq,  chisq)
        summary[peakn] = [costfunc, rsq, chisq]
        # print("At N_PEAKS=",  peakn, " the result was ", costfunc, " with parameters:\n", result['x'])
        output[peakn] = (np.column_stack([xarr, test]), result['x'])
        # output[peakn] = (np.column_stack([xarr, test]), init_params)
    return output,  summary


class FitCore(QObject):
    cleared = pyqtSignal()
    loaded = pyqtSignal()
    fileparams = pyqtSignal(object)
    logmessage = pyqtSignal(str)
    finished_fitting = pyqtSignal(object)
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
        self.lastreduction = 1.0
        self.cutoff = 30
        self.binsize = 0.1
        self.separator = ' '
        self.comment = '#'
        self.xcolumn = 0
        self.ycolumn = 1
        self.ecolumn = -1
        self.fullnames = []
        self.orig_logs = []
        self.shortnames = []
        self.timedata = None
        self.prof_numbers = []
        self.mplot_curves = []
        self.mplot_raw_curves = []
        self.mplot_labels =[]
        self.mplot_override = ['X axis',  'Y axis']
        self.mplot_fits = []
        self.mplot_fitparams = []
        self.map2D = None
        self.overplotworked = False
        self.rixs_worked = False
        self.fitsworked = False
        self.mergeworked = False
        self.merged_curve = None
        self.prof_count = 0
        self.legpos = 0
        self.retvals = []
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
    def new_loadparams(self, pdict):
        for kk in pdict.keys():
            kname = str(kk)
            if kk == 'separator':
                self.separator = pdict[kk]
            elif kk == 'comment':
                self.comment = pdict[kk]
            elif kk == 'xcolumn':
                self.xcolumn = pdict[kk]
            elif kk == 'ycolumn':
                self.ycolumn = pdict[kk]
            elif kk == 'ecolumn':
                self.ecolumn = pdict[kk]
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
            elif 'legpos' in k:
                self.legpos = val
            elif 'offmax' in k:
                self.offmax = val
            elif 'redfac' in k:
                self.redfac = val
            elif 'maxpeaks' in k:
                self.maxpeaks = val+1
            elif 'polyorder' in k:
                self.polyorder = val
            elif 'fixedL' in k:
                self.fixedLwidth = val
            elif 'fixedG' in k:
                self.fixedGwidth = val
            elif 'useedge' in k:
                self.useedge = val
            elif 'penalty' in k:
                self.penalty = val
    def reduce_profiles(self):
        if (len(self.tey_profiles) == len(self.reduced_tey_profiles) and
            len(self.tpy_profiles) == len(self.reduced_tpy_profiles) and
            len(self.raw_tey_profiles) == len(self.reduced_raw_tey_profiles) and
            len(self.raw_tpy_profiles) == len(self.reduced_raw_tpy_profiles)):
            if self.redfac == self.lastreduction:
                return False
        self.reduced_tey_profiles = []
        self.reduced_tpy_profiles = []
        self.reduced_raw_tey_profiles = []
        self.reduced_raw_tpy_profiles = []
        if self.redfac == 1.0:
            for p in self.tey_profiles:
                self.reduced_tey_profiles.append(p)
            for p in self.tpy_profiles:
                self.reduced_tpy_profiles.append(p)
            for p in self.raw_tey_profiles:
                self.reduced_raw_tey_profiles.append(p)
            for p in self.raw_tpy_profiles:
                self.reduced_raw_tpy_profiles.append(p)
        else:
            for num, p in enumerate(self.tey_profiles):
                if self.orig_logs[num] is None:
                    steps = len(p)
                    newsteps = int(round(steps/self.redfac))
                    target = np.zeros([newsteps, 3])
                    target[:, 0] = np.linspace(p[:, 0].min(), p[:, 0].max(), newsteps)
                    newone = merge2curves_errors(p,  target)
                    newone[:, 1:] /= self.redfac
                    self.reduced_tey_profiles.append(newone)
                else:
                    avglog, errlog = place_points_in_bins(copy.deepcopy(self.orig_logs[num]),
                                                                           redfac = self.redfac)
                    newone = np.column_stack([avglog['E'], avglog['CURR2'], errlog['CURR2']])
                    self.reduced_tey_profiles.append(newone)
            for num, p in enumerate(self.tpy_profiles):
                if self.orig_logs[num] is None:
                    steps = len(p)
                    newsteps = int(round(steps/self.redfac))
                    target = np.zeros([newsteps, 3])
                    target[:, 0] = np.linspace(p[:, 0].min(), p[:, 0].max(), newsteps)
                    newone = merge2curves_errors(p,  target)
                    newone[:, 1:] /= self.redfac
                    self.reduced_tpy_profiles.append(newone)
                else:
                    avglog, errlog = place_points_in_bins(copy.deepcopy(self.orig_logs[num]),
                                                                            redfac = self.redfac)
                    newone = np.column_stack([avglog['E'], avglog['CURR1'], errlog['CURR1']])
                    self.reduced_tpy_profiles.append(newone)
            for p in self.raw_tey_profiles:
                self.reduced_raw_tey_profiles.append(p)
            for p in self.raw_tpy_profiles:
                self.reduced_raw_tpy_profiles.append(p)
        self.lastreduction = self.redfac
        return True
    @pyqtSlot()
    def take_table_values(self):
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
        for fnum, fname in enumerate([flist]):
            self.temp_path, short_name = os.path.split(fname)
            try:
                p, e, u, d = read_1D_curve_extended(fname, xcol = self.xcolumn, ycol = self.ycolumn, ecol = self.ecolumn, 
                                                          comment = self.comment, sep = self.separator)
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
        self.fullname = fnames[0]
        self.shortname = snames[0]
        self.profile = profiles[0]
        self.prof_count = 1
        self.loaded.emit()
        self.fileparams.emit([self.fullname, self.shortname,  self.profile])
        return self.fullname, self.shortname, self.profile
    @pyqtSlot()
    def clear_profiles(self):
        self.fullname = ""
        self.shortname = ""
        self.profile = []
        self.prof_count = 0
        self.cleared.emit()
    @pyqtSlot(str)
    def save_fit_results(self, fname):
        if self.merged_curve is None:
            self.logger("There is no merged curve to be saved.")
            return None
        else:
            WriteEnergyProfile(fname, self.merged_curve, [])
            return True
            # outname = os.path.join(self.temp_path, self.temp_name+"_HISTOGRAM.txt")
            # WriteEnergyProfile(outname, target, [])
    @pyqtSlot()
    def sequential_fit(self):
        self.fitsworked = False
        if self.profile is []:
            return None
        else:
            temp = self.profile.copy()
            lowlim,  hilim = self.cuts
            temp = temp[np.where(np.logical_and(temp[:, 0] >= lowlim, temp[:, 0]<=hilim))]
            edge = self.useedge > 0.5
            b_order = self.polyorder
            mpeaks = self.maxpeaks
            gauss_fwhm = self.fixedGwidth
            lorentz_fwhm = self.fixedLwidth
            pfactor = self.penalty
            self.logger("Preparing the sequential fit")
            self.logger("edge, b-order, maxpeaks, gauss-width, lorentz-width, penalty")
            self.logger(", ".join([str(x) for x in [edge, b_order, mpeaks, gauss_fwhm, lorentz_fwhm, pfactor]]))
            results,  summary = iterative_fitting_optimiser(temp, maxpeaks = mpeaks, bkg_order = b_order, include_edge = edge,
                                                    one_gauss_fwhm = gauss_fwhm,  one_lorentz_fwhm = lorentz_fwhm, 
                                                    overshoot_penalty = pfactor)
            self.fit_results = results
            self.fit_summary = summary
            self.finished_fitting.emit(results)
            self.fitsworked = True
    @pyqtSlot()
    def global_fit(self):
        self.fitsworked = False
        if self.profile is []:
            return None
        else:
            temp = self.profile.copy()
            lowlim,  hilim = self.cuts
            temp = temp[np.where(np.logical_and(temp[:, 0] >= lowlim, temp[:, 0]<=hilim))]
            edge = self.useedge > 0.5
            b_order = self.polyorder
            mpeaks = self.maxpeaks
            gauss_fwhm = self.fixedGwidth
            lorentz_fwhm = self.fixedLwidth
            pfactor = self.penalty
            self.logger("Preparing the global fit")
            self.logger("edge, b-order, maxpeaks, gauss-width, lorentz-width, penalty")
            self.logger(", ".join([str(x) for x in [edge, b_order, mpeaks, gauss_fwhm, lorentz_fwhm, pfactor]]))
            results,  summary = global_fitting_optimiser(temp, maxpeaks = mpeaks, bkg_order = b_order, include_edge = edge,
                                                    one_gauss_fwhm = gauss_fwhm,  one_lorentz_fwhm = lorentz_fwhm, 
                                                    overshoot_penalty = pfactor)
            self.fit_results = results
            self.fit_summary = summary
            self.finished_fitting.emit(results)
            self.fitsworked = True
    @pyqtSlot()
    def return_fitpars(self):
        pass

def read_fluxcorr_curves(fname):
    rescurves = []
    source = open(fname, 'r')
    curve = []
    for line in source:
        toks = line.split()
        if len(toks) == 1:
            if '---' in toks[0]:
                if len(curve) > 0:
                    rescurves.append(np.array(curve))
                    curve = []
        elif len(toks) == 2:
            curve.append([float(x) for x in toks])
        else:
            continue
    source.close()
    if len(curve) > 0:
        rescurves.append(np.array(curve))
        curve = []
    return rescurves

class XasCorrector(QObject):
    def __init__(self, parent = None):
        if parent is None:
            super().__init__()
        else:
            super().__init__(parent)
        self.curves_harm1 = [] # the actual data in 2 columns
        self.curves_harm3 = []
        self.curves_harm5 = []
        self.curves_harm7 = []
        self.total_harm1 = [] # combined curves for each harmonics
        self.total_harm3 = []
        self.total_harm5 = []
        self.total_harm7 = []
        self.limits_harm1 = [0, 0] # the total limits of data available in our referece file for this harmonics
        self.limits_harm3 = [0, 0]
        self.limits_harm5 = [0, 0]
        self.limits_harm7 = [0, 0]
        self.inner_limits_harm1 = [] # the speific limits of each curve
        self.inner_limits_harm3 = []
        self.inner_limits_harm5 = []
        self.inner_limits_harm7 = []
        self.limits_need_updating = [True, True, True, True]
    def readCurves(self, fname, harmnum = 1):
        rcurves = read_fluxcorr_curves(fname)
        if harmnum == 1:
            self.curves_harm1 = rcurves
            temp = np.row_stack(rcurves)
            sorting = np.argsort(temp[:, 0])
            self.total_harm1 = temp[sorting]
            self.limits_need_updating[0] = True
        elif harmnum == 3:
            self.curves_harm3 = rcurves
            temp = np.row_stack(rcurves)
            sorting = np.argsort(temp[:, 0])
            self.total_harm3 = temp[sorting]
            self.limits_need_updating[1] = True
        elif harmnum == 5:
            self.curves_harm5 = rcurves
            temp = np.row_stack(rcurves)
            sorting = np.argsort(temp[:, 0])
            self.total_harm5 = temp[sorting]
            self.limits_need_updating[2] = True
        elif harmnum == 7:
            self.curves_harm7 = rcurves
            temp = np.row_stack(rcurves)
            sorting = np.argsort(temp[:, 0])
            self.total_harm7 = temp[sorting]
            self.limits_need_updating[3] = True
    def checkLimits(self):
        for nc in range(4):
            if self.limits_need_updating[nc]:
                if nc == 0:
                    allcurves = self.curves_harm1
                elif nc == 1:
                    allcurves = self.curves_harm3
                elif nc == 2:
                    allcurves = self.curves_harm5
                elif nc == 3:
                    allcurves = self.curves_harm7
                temp = []
                abstemp = [1200.0, 0.0]
                for cur in allcurves:
                    temp.append((cur[:, 0].min(), cur[:, 0].max()))
                    maxx = max(abstemp[1], cur[:, 0].max())
                    minx = min(abstemp[0],  cur[:, 0].min())
                    abstemp = [minx, maxx]
                if nc == 0:
                    self.limits_harm1 = abstemp
                    self.inner_limits_harm1 = temp
                elif nc == 1:
                    self.limits_harm3 = abstemp
                    self.inner_limits_harm3 = temp
                elif nc == 2:
                    self.limits_harm5 = abstemp
                    self.inner_limits_harm5 = temp
                elif nc == 3:
                    self.limits_harm7 = abstemp
                    self.inner_limits_harm7 = temp
                self.limits_need_updating[nc] = False
    def returnInterpolatedCurve(self, xarray, kind_covered = 'linear', kind_notcovered = 'cubic'):
        """
        The input should be just the x array of the data we are trying to match.
        The returned curve will be the flux correction as a function of photon energy.
        The interpolation method will be different, depending on where the x values are
        compared to the measured reference data.
        """
        self.checkLimits()
        inarr = xarray[np.argsort(xarray)]
        match = [0.0, 0.0, 0.0, 0.0]
        for nh in range(4):
            if nh == 0:
                abslims = self.limits_harm1
                seplims = self.inner_limits_harm1
            elif nh == 1:
                abslims = self.limits_harm3
                seplims = self.inner_limits_harm3
            elif nh == 2:
                abslims = self.limits_harm5
                seplims = self.inner_limits_harm5
            elif nh == 3:
                abslims = self.limits_harm7
                seplims = self.inner_limits_harm7
            match_deg = np.logical_and(inarr >= abslims[0], inarr <= abslims[1]).sum() / len(inarr)
            match[nh] = match_deg # we check the overlap between the x range of the input data and reference data
        match = np.array(match)
        if np.all(match == 0.0):
            print("None of the data ranges match the input data range. Flux correction is not possible.")
            return None
        harmnum = np.argmax(match)
        if harmnum == 0:
            abslims = self.limits_harm1
            seplims = self.inner_limits_harm1
            total = self.total_harm1
            curves = self.curves_harm1
        elif harmnum == 1:
            abslims = self.limits_harm3
            seplims = self.inner_limits_harm3
            total = self.total_harm3
            curves = self.curves_harm3
        elif harmnum == 2:
            abslims = self.limits_harm5
            seplims = self.inner_limits_harm5
            total = self.total_harm5
            curves = self.curves_harm5
        elif harmnum == 3:
            abslims = self.limits_harm7
            seplims = self.inner_limits_harm7
            total = self.total_harm7
            curves = self.curves_harm7
        # interpolation kinds
        # 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next'
        newcurve = np.zeros((len(xarray), 2))
        newcurve[:, 0] = xarray
        newcurve[:, 1] = interp1d(total[:, 0],  total[:, 1], kind = kind_notcovered,  bounds_error=False, fill_value = 0.0)(xarray)
        for nc, curve in enumerate(curves):
            xref = curve[:, 0]
            xmin,  xmax = xref.min(), xref.max()
            criterion = np.logical_and(xarray >= xmin, xarray <= xmax)
            if not np.any(criterion):
                continue # let's not waste time on the curves out of the range
            tempx = xarray[np.where(criterion)]
            newy = interp1d(curve[:, 0], curve[:, 1],  kind = kind_covered,  bounds_error=False, fill_value = 0.0)(tempx)
            newcurve[:, 1][np.where(criterion)] = newy
        return newcurve

def save_array(nparray,  fname):
    target = open(fname, 'w')
    for ln in nparray:
        line = " ".join([str(x) for x in ln]) + '\n'
        target.write(line)
    target.close()

## this is how it works
#newx1 = np.arange(100.0, 600.0, 0.01)
#newx2 = np.arange(600.0, 1200.0, 0.01)
#save_array(xas.total_harm1, 'rawdata_harm1.txt')
#save_array(xas.total_harm3, 'rawdata_harm3.txt')
#for inkind in ['linear', 'slinear', 'quadratic', 'cubic']:
#    curve_harm1 = xas.returnInterpolatedCurve(newx1, kind_notcovered = inkind)
#    curve_harm3 = xas.returnInterpolatedCurve(newx2, kind_notcovered = inkind)
#    save_array(curve_harm1, 'interpdata_'+inkind+'_harm1.txt')
#    save_array(curve_harm3, 'interpdata_'+inkind+'_harm3.txt')
