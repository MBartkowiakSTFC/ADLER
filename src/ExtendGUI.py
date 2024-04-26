
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
This is a collection of useful PyQt widgets used in ADLER
and other software projects related to PEAXIS.
"""

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer, QThread, QMutex,  QSemaphore, Qt, \
                                        QPersistentModelIndex
from PyQt6.QtWidgets import QProgressBar,  QWidget,  QTextEdit, QVBoxLayout, QCheckBox, \
                                                QGroupBox, QDialog, QLabel, QTabWidget, QGridLayout, \
                                                QDockWidget, QScrollArea, QSplitter, QSizePolicy, QTableView, QMenu
from PyQt6.QtGui import QFont, QColor, QPainter, QStandardItemModel, QStandardItem

progbar_stylesheet =  "QProgressBar {background-color:rgb(30,10,20); border: 2px solid grey; border-radius: 5px}"
chunk_stylesheet =  "QProgressBar::chunk {background-color:rgb(210,20,20); border-color:rgb(180,20,20); width: 10px; border: 3px; margin: 2px}"

varbox_stylesheet =  "QLineEdit {background-color:rgb(30,250,250); border: 2px solid grey; border-radius: 5px}"
display_stylesheet =  "QLineEdit {background-color:rgb(220,170,250); border: 2px solid grey; border-radius: 2px}"

class PeaxisDataModel(QStandardItemModel):
    pass_data = pyqtSignal(object)
    def __init__(self, parent, column_names):
        super().__init__(parent)
        self.setHorizontalHeaderLabels(column_names)
        self.cnames = column_names
        self.col_order = column_names
        self._file_dict = {}
        self._header_dict = {}
        self._log_dict = {}
    def cleanup(self):
        self.clear()
        self.setHorizontalHeaderLabels(self.cnames)
        self.col_order = self.cnames
    @pyqtSlot(object)
    def add_ListOfLists(self, new_entries):
        templist = []
        for new_entry in new_entries:
            templist = []
            for x in new_entry:
                templist.append(QStandardItem(str(x)))
            self.appendRow(templist)
    def add_entry_list(self, new_entry):
        templist = []
        for x in new_entry:
            templist.append(QStandardItem(str(x)))
        self.appendRow(templist)
    def add_row(self, new_entry):
        templist = []
        for kk in self.cnames:
            try:
                element = new_entry[kk]
            except:
                element = -99.0
            if kk == 'CostFunction':
                if abs(element) < 1e-7:
                    element = 0.0
                else:
                    element = round(element, 7)
            else:
                element = round(element, 3)
            templist.append(QStandardItem(str(element)))
        self.appendRow(templist)
    @pyqtSlot()
    def trigger_values(self):
        results = []
        for row in range(self.rowCount()):
            line = []
            for column in range(self.columnCount()):
                item = self.item(row, column).text()
                line.append(float(item))
            results.append(line)
        self.pass_data.emit(results)
        return results

class PeaxisTableView(QTableView):
    selected_data = pyqtSignal(object)
    def __init__(self, parent, actions = ['Accept', 'Delete']):
        super().__init__(parent)
        self.visible_actions = actions
    def contextMenuEvent(self, event):
        selected_items = self.selectedIndexes()
        selected_rows = np.unique([x.row() for x in selected_items])
        ncols = self.model().columnCount()
        if selected_items:
            menu = QMenu(self)
            accept_line = menu.addAction("Accept")
            delete_line = menu.addAction("Delete")
            if 'Accept' not in self.visible_actions:
                accept_line.setVisible(False)
            if 'Delete' not in self.visible_actions:
                delete_line.setVisible(False)
            action = menu.exec(self.mapToGlobal(event.pos()))
            if action:
                if action == accept_line:
                    data_list = []                                                          
                    for row in selected_rows:
                        line=[]
                        for c in range(ncols):
                            line.append(self.model().index(row, c).data())
                        data_list.append(line)
                        # index = QPersistentModelIndex(model_index)         
                        # data_list.append(index.data())                                             
                    self.selected_data.emit(data_list)
                elif action == delete_line:
                    index_list = []                                                          
                    for model_index in self.selectionModel().selectedRows():       
                        index = QPersistentModelIndex(model_index)         
                        index_list.append(index)                                             
                    for index in index_list:                                      
                         self.model().removeRow(index.row())   

class LightWidget(QWidget):
    def __init__(self, parent, startval = False):
        super().__init__(parent)
        self.isOn = startval
        self.red = QColor(200,0,0)
        self.black = QColor(0,0,0)
        self.yellow = QColor(120,80,0)
        self.green = QColor(0,200,0)
        self.blue = QColor(0,20,200)
        self.color = self.red
        self.setValue(startval)
    @pyqtSlot()
    def toggle(self):
        self.isOn = not self.isOn
        if self.isOn:
            self.color = self.green
        else:
            self.color = self.red
        self.update()
    @pyqtSlot(int)
    def setValue(self, nVal):
        if nVal is True:
            self.isOn = True
            self.color = self.green
        elif nVal is False:
            self.isOn = False
            self.color = self.red
        elif nVal >2:
            self.isOn = 3
            self.color = self.blue
        elif nVal >1:
            self.isOn = 2
            self.color = self.green
        elif nVal >0:
            self.isOn = 1
            self.color = self.yellow
        else:
            self.isOn = 0
            self.color = self.black
        self.update()
    def paintEvent(self, event):
        qpaint = QPainter(self)
        qpaint.setRenderHint(QPainter.Antialiasing)
        qpaint.setBrush(self.color)
        qpaint.drawEllipse(0, 0, self.width(), self.height())

class InfoDialog(QDialog):
    def __init__(self, parent, progname = "", version = "", years = [2019, 2020], pixmap = None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        lab1 = QLabel("This is "+progname+", version "+version, self)
        lab2 = QLabel("Copyright: Maciej Bartkowiak, " + '-'.join([str(y) for y in years]), self)
        lab3 = QLabel("This software is licensed under GNU GPL 3.", self)
        lab4 = QLabel("Developed at Helmholtz Zentrum Berlin.", self)
        self.layout.addWidget(lab1)
        self.layout.addWidget(lab2)
        self.layout.addWidget(lab3)
        self.layout.addWidget(lab4)
        if pixmap is not None:
            lab5 = QLabel("", self)
            lab5.setPixmap(pixmap)
            self.layout.addWidget(lab5)
        self.setLayout(self.layout)
        self.setWindowTitle("Version information for " + progname)

class ManyButtons(QWidget):
    new_values = pyqtSignal(object)
    def __init__(self, parent, names = [], label = "",  ticked = False):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.global_checkbox = QCheckBox("Mark all", self)
        self.group = QGroupBox(label, self)
        self.inner_layout = QVBoxLayout(self.group)
        self.blist = []
        self.bvals = []
        self.global_checkbox.setTristate()
        self.global_checkbox.clicked.connect(self.toggle_all)
        for n in names:
            temp = QCheckBox(n, self)
            self.inner_layout.addWidget(temp)
            self.bvals.append(ticked)
            temp.clicked.connect(self.update_global)
            temp.stateChanged.connect(self.update_boolvals)
            self.blist.append(temp)
        self.layout.addWidget(self.global_checkbox)
        self.layout.addWidget(self.group)
        self.bvals = np.array(self.bvals)
        self.totalnum = len(self.bvals)
    @pyqtSlot()
    def toggle_all(self):
        newstate = self.global_checkbox.checkState()
        boolval = newstate == Qt.CheckState.Checked
        # print("Setting all boxes to ",  boolval)
        for n, b in enumerate(self.blist):
            if boolval:
                b.setCheckState(Qt.CheckState.Checked)
            else:
                b.setCheckState(Qt.CheckState.Unchecked)
            self.bvals[n] = boolval
    @pyqtSlot()
    def update_global(self):
        current_state = self.bvals.sum()
        # print(current_state, "out of ", self.totalnum, "boxes are checked now")
        if current_state == 0:
            self.global_checkbox.setCheckState(Qt.CheckState.Unchecked)
        elif current_state == self.totalnum:
            self.global_checkbox.setCheckState(Qt.CheckState.Checked)
        else:
            self.global_checkbox.setCheckState(Qt.CheckState.PartiallyChecked)
    @pyqtSlot(int)
    def update_boolvals(self, bnum):
        for n, b in enumerate(self.blist):
            newval = b.checkState() == Qt.CheckState.Checked
            self.bvals[n] = newval
        self.new_values.emit(self.bvals)

class FlexBar(QProgressBar):
    def __init__(self,  parent):
        super().__init__(parent)
        self.forward = True
        self.setTextVisible(False)
        self.setStyleSheet(progbar_stylesheet + '\n' + chunk_stylesheet)
    @pyqtSlot()
    def reverse(self):
        if self.forward == True:
            self.forward = False
        else:
            self.forward = True
        self.setInvertedAppearance(not self.forward)

class LogBox(QTextEdit):
    def __init__(self,  parent):
        super().__init__(parent)
        self.setReadOnly(True)
    @pyqtSlot(str)
    def put_logmessage(self,  message):
        self.setReadOnly(False)
        self.append(message)
        self.setReadOnly(True)

class KnightRider(QObject):
    nextvalue = pyqtSignal(int)
    turnaround = pyqtSignal()
    clearbar = pyqtSignal()
    def __init__(self,  progbar,  parent = None, totalsteps = 100):
        super().__init__(parent)
        self.progbar = progbar
        self.progbar.setMaximum(totalsteps)
        self.currval = 0
        self.maxval = totalsteps
        self.minval = 0
        self.forward = True
        self.timer = QTimer(self)
        self.nextvalue.connect(progbar.setValue)
        self.turnaround.connect(progbar.reverse)
        self.clearbar.connect(progbar.reset)
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.step)
    def start(self):
        self.timer.start()
    def stop(self):
        self.timer.stop()
        self.clearbar.emit()
        self.currval = 0
        self.forward = True
        self.progbar.setInvertedAppearance(False)
    @pyqtSlot()
    def step(self):
        if self.forward:
            self.currval += 1
        else:
            self.currval -= 1
        if (self.currval >= self.maxval):
            self.forward = not self.forward
            self.turnaround.emit()
        elif (self.currval <= self.minval):
            self.forward = not self.forward
        self.nextvalue.emit(self.currval)
        # # print("timer ticked")

class OneshotWorker(QObject):
    finished = pyqtSignal()
    intReady = pyqtSignal(int)
    def __init__(self):
        super().__init__()
        self.runThis = None
        self.keeprunnig = True
        self.waittime = 1
        self.func_args = []
    def set_end(self):
        self.keeprunning = False
    def set_waittime(self, wtime):
        self.waittime = wtime
    def assignFunction(self, some_function):
        self.runThis = some_function
    def assignArgs(self, args):
        self.func_args = args
    @pyqtSlot()
    def run_once(self): # A slot takes no params
        try:
            nargs = len(self.func_args)
        except:
            nargs = 0
        if self.runThis is not None:
            if nargs == 0:
                self.runThis()
            elif nargs == 1:
                self.runThis(self.func_args[0])
            elif nargs == 2:
                self.runThis(self.func_args[0],  self.func_args[1])
        # # print("Now the worker object should emit 'finished'")
        self.finished.emit()

class ThreadpoolWorker(QObject):
    finished = pyqtSignal()
    intReady = pyqtSignal(int)
    def __init__(self, tpool):
        super().__init__()
        self.runThis = None
        self.keeprunnig = True
        self.waittime = 1
        self.func_args = []
        self.tpool = tpool
    def set_end(self):
        self.keeprunning = False
    def set_waittime(self, wtime):
        self.waittime = wtime
    def assignFunction(self, some_function):
        self.runThis = some_function
    def assignArgs(self, args):
        self.func_args = args
    @pyqtSlot()
    def run_once(self): # A slot takes no params
        try:
            nargs = len(self.func_args)
        except:
            nargs = 0
        if self.runThis is not None:
            if nargs == 0:
                self.runThis()
            elif nargs == 1:
                self.runThis(self.func_args[0])
            elif nargs == 2:
                self.runThis(self.func_args[0],  self.func_args[1])
        self.tpool.decrement()
        self.finished.emit()

class LogrusTab(QObject):
    new_parameters = pyqtSignal(object)
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        base = QWidget(self.master)
        self.base=base
        self.values = {}
        self.varboxes = []
        self.glayout = QGridLayout(base)
        self.mylock = QMutex()
        self.timer = QTimer()
        self.core = None
        self.thread = None
        self.val_changed = False
        self.temp_path = '.'
        self.timer.timeout.connect(self.trigger_output)
    def add_varbox(self, vbox, rown, coln,  rowspan = 1, colspan = 1):
        self.add_widget(vbox.base, rown, coln,  rowspan, colspan)
        self.varboxes.append(vbox)
        vbox.pass_values.connect(self.read_varboxes)
        # self.new_parameters.connect(vbox.receiveValues)
        vbox.base.setStyleSheet(varbox_stylesheet)
    def add_display(self, disp, rown, coln,  rowspan = 1, colspan = 1):
        self.add_widget(disp.base, rown, coln,  rowspan, colspan)
        # disp.pass_values.connect(self.take_values)
        disp.base.setStyleSheet(display_stylesheet)
    def add_widget(self, widget, rown, coln,  rowspan = 1, colspan = 1):
        self.glayout.addWidget(widget, rown, coln,  rowspan, colspan)
    def setup_timer(self, interval, function):
        self.timer.setInterval(interval)
        self.timer.timeout.connect(function)
    def trigger_readout(self):
        self.mylock.lock()
        for vb in self.varboxes:
            vb.updateOutputs()
        self.mylock.unlock()
    @pyqtSlot(object)
    def read_varboxes(self, something):
        self.mylock.lock()
        for kk in something.keys():
            self.values[kk] = something[kk]
        self.val_changed = True
        self.mylock.unlock()
    @pyqtSlot(object)
    def take_values(self, something):
        self.mylock.lock()
        for vb in self.varboxes:
            # vb.receiveValues(something)
            vb.takeValues(list(something.keys()), something)
        # for kk in something.keys():
        #     self.values[kk] = something[kk]
        # self.val_changed = True
        self.mylock.unlock()
    @pyqtSlot(object)
    def return_values(self):
        tempdict = {}
        self.mylock.lock()
        for kk in self.values.keys():
            tempdict[kk] = self.values[kk]
        self.mylock.unlock()
        return tempdict
    @pyqtSlot()
    def trigger_output(self):
        self.mylock.lock()
        if self.val_changed == True:
            self.new_parameters.emit(self.values)
            self.val_changed = False
        self.mylock.unlock()
    @pyqtSlot()
    def start_core(self):
        if self.core is not None and self.thread is None:
            self.thread = QThread()
            self.core.moveToThread(self.thread)
            # obj.finished.connect(thread.quit)
            # thread.started.connect(obj.run_once)
            # obj.assignFunction(target_function)
            # obj.assignArgs(args)
            self.thread.start()
    def assign_core(self, thecore):
        if self.core is not None and self.thread is None:
            self.core = thecore
            self.new_parameters.connect(self.core.takeParams)
        else:
            print("There is a core in this tab already. There can be only one!")
    def start_timer(self, interval):
        self.timer.setInterval(5)
        self.timer.start()
    def stop_timer(self):
        self.timer.stop()

class LogrusCore(QObject):
    send_results = pyqtSignal(object)
    return_variables = pyqtSignal(object)
    def __init__(self):
        super().__init__()
        self.vars ={}
        self.mylock = QMutex()
    @pyqtSlot(object)
    def takeParams(self, vdict):
        self.mylock.lock()
        for kk in vdict.keys():
            self.vars[kk] = vdict[kk]
        self.return_variables.emit(self.vars)
        self.mylock.unlock()
    @pyqtSlot()
    def produceResults(self):
        pass

class AdlerTab(QObject):
    conf_update = pyqtSignal(object)
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        base = QWidget(self.master)
        # self.splitter = QSplitter(base)
        # self.splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.base=base
        self.progbar = FlexBar(base)
        self.oscillator = KnightRider(self.progbar, parent = base)
        self.current_thread = (None,  None)
    @pyqtSlot()
    def block_interface(self):
        self.oscillator.start()
        self.boxes_base.setEnabled(False)
        self.button_base.setEnabled(False)
    @pyqtSlot()
    def unblock_interface(self):
        self.boxes_base.setEnabled(True)
        self.button_base.setEnabled(True)
    def background_launch(self,  core_function,  args =[]):
        self.oscillator.start()
        self.block_interface()
        # self.core.thread_start(core_function,  args)
        core_function(args)
    @pyqtSlot()
    def flip_buttons(self):
        self.unblock_interface()
        for n in range(len(self.button_list)):
            self.button_list[n].setEnabled(self.active_buttons[n])
        self.current_thread = (None,  None)
        self.oscillator.stop()
    def thread_locknload(self, target_function,  args = []):
        self.oscillator.start()
        self.block_interface()
        # 1 - create Worker and Thread inside the Form
        obj = OneshotWorker()  # no parent!
        thread = QThread()  # no parent!
        # 2 - Connect Worker`s Signals to Form method slots to post data.
        # self.obj.intReady.connect(self.onIntReady)
        # 3 - Move the Worker object to the Thread object
        obj.moveToThread(thread)
        # 4 - Connect Worker Signals to the Thread slots
        obj.finished.connect(thread.quit)
        # 5 - Connect Thread started signal to Worker operational slot method
        thread.started.connect(obj.run_once)
        # custom config
        obj.assignFunction(target_function)
        obj.assignArgs(args)
        # obj.set_waittime(0.1)
        #
        # * - Thread finished signal will close the app if you want!
        thread.finished.connect(self.flip_buttons)
        self.current_thread = (obj,  thread)
        # 6 - Start the thread
        # self.thread.start()
        return obj, thread

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

class PatternDialog(QDialog):
    values_ready = pyqtSignal(object)
    def __init__(self, master,  dialogname = ""):
        super().__init__(master)
        self.layout = QVBoxLayout(self)
        self.tabbar = QTabWidget(self)
        self.progbar = FlexBar(self)
        self.oscillator = KnightRider(self.progbar, parent = self)
        self.current_thread = (None,  None)
        self.layout.addWidget(self.tabbar)
        self.layout.addWidget(self.progbar)
        self.setWindowTitle(dialogname)
    @pyqtSlot()
    def block_interface(self):
        self.oscillator.start()
        self.boxes_base.setEnabled(False)
        self.button_base.setEnabled(False)
    @pyqtSlot()
    def unblock_interface(self):
        self.boxes_base.setEnabled(True)
        self.button_base.setEnabled(True)
    def background_launch(self,  core_function,  args =[]):
        self.oscillator.start()
        self.block_interface()
        # self.core.thread_start(core_function,  args)
        core_function(args)
    @pyqtSlot()
    def flip_buttons(self):
        self.unblock_interface()
        for n in range(len(self.button_list)):
            self.button_list[n].setEnabled(self.active_buttons[n])
        self.current_thread = (None,  None)
        self.oscillator.stop()
