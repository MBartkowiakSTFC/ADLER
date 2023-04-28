
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
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer, QThread, QMutex,  QSemaphore, Qt, \
                                        QPersistentModelIndex
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import  QApplication
from ExtendGUI import CustomThreadpool
from DataHandling import DataEntry, DataGroup, RixsMeasurement
# this is a Windows thing
# ctypes.windll.kernel32.SetDllDirectoryW('.')



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


class CustomThreadpool(QObject):
    finished = pyqtSignal()
    def __init__(self,  MAX_THREADS = 1):
        super().__init__()
        self.maxthreads = MAX_THREADS
        self.mutex = QMutex()
        self.sem = QSemaphore(MAX_THREADS)
        self.reset()
        self.finished.connect(self.reset)
    @pyqtSlot()
    def reset(self):
        self.mutex.lock()
        self.counter = 0
        self.tasklist = [] # these are the pending tasks
        self.runlist = []
        self.objlist = []
        self.isrunning = False
        self.mutex.unlock()
    @pyqtSlot()
    def decrement(self):
        self.sem.release(1)
        # print('Semaphore released. Now available: ',  self.sem.available())
        self.mutex.lock()
        self.counter -= 1
        # if self.running == 0:
        #     self.finished.emit()
        self.mutex.unlock()
    @pyqtSlot()
    def run(self):
        if self.isrunning == True:
            return None
        self.isrunning = True
        # print("threadpool: isrunning")
        while len(self.tasklist) > 0:
            self.launchone()
        # print("threadpool: stopped submitting, waiting to complete")
        # print("Semaphore. Available: ",  self.sem.available(),  "asking for",  self.maxthreads)
        self.sem.acquire(self.maxthreads)
        for th in self.runlist:
            th.quit()
        for th in self.runlist:
            th.wait()
        self.finished.emit()
        self.sem.release(self.maxthreads)
        # print("Semaphore. Available: ",  self.sem.available(),  "released",  self.maxthreads)
    def launchone(self):
        self.mutex.lock()
        th = self.tasklist.pop()
        self.runlist.append(th)
        self.mutex.unlock()
        # print("threadpool: launchone: about to acquire")
        self.sem.acquire(1)
        # print('Semaphore acquired. Now available: ',  self.sem.available())
        th.start()
    def make_thread(self, target_function,  args = []):
        # obj = OneshotWorker()  # no parent!
        obj = ThreadpoolWorker(self)  # no parent!
        thread = QThread()  # no parent!
        obj.moveToThread(thread)
        obj.finished.connect(thread.quit)
        thread.started.connect(obj.run_once)
        obj.assignFunction(target_function)
        obj.assignArgs(args)
        # thread.finished.connect(self.decrement)
        return obj, thread
    def populate(self,  target_function,  arglist = []):
        for arginst in arglist:
            obj,  thread = self.make_thread(target_function, args = arginst)
            self.tasklist.append(thread)
            self.objlist.append(obj)
    def add_task(self,  target_function):
        if self.isrunning == False:
            obj,  thread = self.make_thread(target_function)
            self.tasklist.append(thread)
            self.objlist.append(obj)
            # print("threadpool: added a task")


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

