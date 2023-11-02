
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
A collection of GUI components handling variable input in ADLER
and derived software projects.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator, QIntValidator
from PyQt6.QtCore import  QObject, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QLabel,  QGroupBox, QGridLayout, QLineEdit, \
                                                QWidget,  QHBoxLayout,  QSizePolicy, QCheckBox, QApplication, \
                                                QComboBox, QFormLayout, QButtonGroup
import numpy as np
import copy

input_stylesheet =  "QLineEdit {background-color:rgb(30,250,250); border: 2px solid grey; border-radius: 5px}"
output_stylesheet =  "QLineEdit {background-color:rgb(220,170,250); border: 2px solid grey; border-radius: 2px}"
warning_stylesheet =  "QLineEdit {background-color:rgb(250,250,0); border-color:rgb(0,0,0); border: 2px solid grey; border-radius: 2px}"
 #normal_stylesheet = QApplication.styleSheet()

class MyDoubleValidator(QDoubleValidator):
    def __init__(self, mins, maxes, flen, linedit):
        super().__init__( mins, maxes, flen, linedit)
    def fixup(self, s):
        try:
            val = float(s)
        except:
            return str((self.top() + self.bottom())/2.0)
        else:
            if val > self.top():
                return str(self.top())
            elif val < self.bottom():
                return str(self.bottom())
            else:
                return str((self.top() + self.bottom())/2.0)

class MyIntValidator(QIntValidator):
    def __init__(self, mins, maxes, linedit):
        super().__init__( mins, maxes, linedit)
    def fixup(self, s):
        try:
            val = int(s)
        except:
            return str(int((self.top() + self.bottom())/2.0))
        else:
            if val > self.top():
                return str(self.top())
            elif val < self.bottom():
                return str(self.bottom())
            else:
                return str(int((self.top() + self.bottom())/2.0))

class StringVariable(QObject):
    values_changed = pyqtSignal()
    def __init__(self, parent, name, value,  grid = None,  tooltip=""):
        super().__init__(parent)
        self.v_name = name
        self.value = value
        self.tooltip = tooltip
        base = QWidget(parent)
        label = QLabel(self.v_name, base)
        label.setToolTip(self.tooltip)
        flay = QHBoxLayout(base)
        inlin = QLineEdit(str(self.value), base)
        # inlin.setMinimumHeight(20)
        inlin.setText(str(self.value))
        inlin.setToolTip(self.tooltip)
        inlin.returnPressed.connect(self.update)
        inlin.editingFinished.connect(self.update)
        inlin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        flay.addWidget(inlin)
        if grid is None:
            layout = QHBoxLayout(base)
            # infield.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Preferred)
            layout.addWidget(label)
            layout.addWidget(base)
        else:
            nGridRows = grid.rowCount()
            grid.addWidget(label,  nGridRows,  0)
            grid.addWidget(base,  nGridRows,  1)
        self.field = inlin
        self.update()
    def returnValue(self):
        return self.value
    @pyqtSlot()
    def update(self):
        val = self.field.text()
        self.value = val
        self.values_changed.emit()
    def takeValue(self,  newvalue):
        self.value = str(newvalue)
        self.field.setText(str(self.value))
        
        
class InputVariable(QObject):
    values_changed = pyqtSignal()
    def __init__(self, parent, variable,  grid = None,  prec=9):
        super().__init__(parent)
        self.normal_style = ""
        self.prec = prec
        self.v_name = variable['Name']
        self.v_unit = variable['Unit']
        self.v_len = variable['Length']
        self.v_value = variable['Value']
        self.v_minval = variable['MinValue']
        self.v_maxval = variable['MaxValue']
        self.v_type = variable['Type']
        try:
            self.w_zones = variable['WarnZones']
        except:
            self.w_zones = []
        self.tooltip=""
        for k in variable.keys():
            self.tooltip += str(k) + ": " + str(variable[k]) + '\n'
        self.maxismatched = False
        self.minismatched = False
        self.isinteger = False
        if 'int' in self.v_type:
            self.isinteger = True
        self.veclen = -1
        self.minlen = -2
        self.maxlen = -3
        self.value = np.zeros(self.v_len)
        self.minval = np.zeros(self.value.shape)
        self.maxval = np.zeros(self.value.shape)
        veclen = self.v_len
        try:
            maxlen = len(self.v_maxval)
        except:
            maxlen = 1
        try:
            minlen = len(self.v_minval)
        except:
            minlen = 1
        self.veclen = veclen
        self.maxlen = maxlen
        self.minlen = minlen
        if maxlen >= veclen:
            self.maxismatched = True
        if minlen >= veclen:
            self.minismatched = True
        if self.v_len > 1:
            for n,  x in enumerate(self.v_value):
                self.value[n] = x
                try:
                    self.maxval[n] = np.array(self.v_maxval)[n]
                except:
                    try:
                        self.maxval[n] = np.array(self.v_maxval)[0]
                    except:
                        self.maxval[n] = 1e12
                try:
                    self.minval[n] = np.array(self.v_minval)[n]
                except:
                    try:
                        self.minval[n] = np.array(self.v_minval)[0]
                    except:
                        self.minval[n] = 1e12
        else:
            self.value[0] = float(self.v_value)
            self.maxval[0] = float(self.v_maxval)
            self.minval[0] = float(self.v_minval)
        if self.isinteger:
            self.value = self.value.astype(int)
            self.maxval = self.maxval.astype(int)
            self.minval = self.minval.astype(int)
        base = QWidget(parent)
        self.normal_style = base.styleSheet()
        self.normal_font = base.font()
        label = QLabel(self.v_name, base)
        label2 = QLabel(self.v_unit, base)
        label.setToolTip(self.tooltip)
        label2.setToolTip(self.tooltip)
        fields = []
        fbase = QWidget(base)
        flay = QHBoxLayout(fbase)
        for ln in range(self.veclen):
            inlin = QLineEdit(str(self.value[ln]), base)
            if self.isinteger:
                dval = MyIntValidator(self.minval[ln], self.maxval[ln], inlin)
            else:
                dval = MyDoubleValidator(self.minval[ln], self.maxval[ln], self.prec, inlin)
            inlin.setValidator(dval)
            inlin.setMinimumHeight(20)
            inlin.setText(str(self.value[ln]))
            inlin.setToolTip(self.tooltip)
            if self.isinteger:
                inlin.returnPressed.connect(self.updateInt)
                inlin.editingFinished.connect(self.updateInt)
            else:
                inlin.returnPressed.connect(self.update)
                inlin.editingFinished.connect(self.update)
            flay.addWidget(inlin)
            fields.append(inlin)
        self.fields = fields
        self.defvalue = copy.deepcopy(self.value)
        if grid is None:
            layout = QHBoxLayout(base)
            # infield.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Preferred)
            layout.addWidget(label)
            layout.addWidget(fbase)
            layout.addWidget(label2)
        else:
            nGridRows = grid.rowCount()
            grid.addWidget(label,  nGridRows,  0)
            grid.addWidget(fbase,  nGridRows,  1)
            grid.addWidget(label2,  nGridRows,  2)
        if self.isinteger:
            self.updateInt()
        else:
            self.update()
        # self.normal_style = self.fields[0].styleSheet()
    @pyqtSlot()
    def enableChanges(self):
        for n,  tfield in enumerate(self.fields):
            tfield.setReadOnly(False)
    @pyqtSlot()
    def disableChanges(self):
        for n,  tfield in enumerate(self.fields):
            tfield.setReadOnly(True)
    @pyqtSlot()
    def update(self):
        for n,  tfield in enumerate(self.fields):
            val = tfield.text()
            try:
                val = float(val)
            except:
                val = self.defvalue[n]
                tfield.setText(str(val))
#            else:
#                if val > self.maxval[n]:
#                    val = self.maxval[n]
#                    tfield.setText(str(val))
#                elif val < self.minval[n]:
#                    val = self.minval[n]
#                    tfield.setText(str(val))
            self.value[n] = val
            highlight = False
            for wzone in self.w_zones:
                if val >= wzone[0] and val <= wzone[1]:
                    highlight = True
            if highlight:
                tfield.setStyleSheet(warning_stylesheet)
            else:
                tfield.setStyleSheet(self.normal_style)
            tfield.setFont(self.normal_font)
        # print("update was called")
        self.values_changed.emit()
    @pyqtSlot()
    def updateInt(self):
        for n,  tfield in enumerate(self.fields):
            val = tfield.text()
            try:
                val = int(float(val))
            except:
                val = self.defvalue[n]
                tfield.setText(str(val))
#            else:
#                if val > self.maxval[n]:
#                    val = self.maxval[n]
#                    tfield.setText(str(val))
#                elif val < self.minval[n]:
#                    val = self.minval[n]
#                    tfield.setText(str(val))
            self.value[n] = val
            highlight = False
            for wzone in self.w_zones:
                if val >= wzone[0] and val <= wzone[1]:
                    highlight = True
            if highlight:
                tfield.setStyleSheet(warning_stylesheet)
            else:
                tfield.setStyleSheet(self.normal_style)
            tfield.setFont(self.normal_font)
        # print("updateInt was called")
        self.values_changed.emit()
    def returnValue(self):
        return self.value
    def takeValue(self,  newvalue):
        try:
            len(newvalue)
        except:
            newvalue = [newvalue]
        for nn in range(len(newvalue)):
            try:
                tval = float(newvalue[nn])
            except:
                continue
            if self.isinteger:
                tval = int(tval)
            if tval > self.maxval[nn]:
                tval = self.maxval[nn]
            if tval < self.minval[nn]:
                tval = self.minval[nn]
            self.value[nn] = tval
            # print(self.v_name, "about to set the ",  nn,  "field to",  tval)
            self.fields[nn].setText(str(round(tval, self.prec)))

class IOVariable(QObject):
    values_changed = pyqtSignal()
    state_changed = pyqtSignal()
    def __init__(self, parent, variable,  grid = None,  prec=9, blabel = '<- FIXED!'):
        super().__init__(parent)
        self.prec = prec
        self.v_name = variable['Name']
        self.v_unit = variable['Unit']
        self.v_len = variable['Length']
        self.v_value = variable['Value']
        self.v_minval = variable['MinValue']
        self.v_maxval = variable['MaxValue']
        self.v_type = variable['Type']
        self.tooltip=""
        self.box_label = blabel
        self.is_input = True
        for k in variable.keys():
            self.tooltip += str(k) + ": " + str(variable[k]) + '\n'
        self.maxismatched = False
        self.minismatched = False
        self.isinteger = False
        if 'int' in self.v_type:
            self.isinteger = True
        self.veclen = -1
        self.minlen = -2
        self.maxlen = -3
        self.value = np.zeros(self.v_len)
        self.minval = np.zeros(self.value.shape)
        self.maxval = np.zeros(self.value.shape)
        veclen = self.v_len
        try:
            maxlen = len(self.v_maxval)
        except:
            maxlen = 1
        try:
            minlen = len(self.v_minval)
        except:
            minlen = 1
        self.veclen = veclen
        self.maxlen = maxlen
        self.minlen = minlen
        if maxlen >= veclen:
            self.maxismatched = True
        if minlen >= veclen:
            self.minismatched = True
        if self.v_len > 1:
            for n,  x in enumerate(self.v_value):
                self.value[n] = x
                try:
                    self.maxval[n] = np.array(self.v_maxval)[n]
                except:
                    try:
                        self.maxval[n] = np.array(self.v_maxval)[0]
                    except:
                        self.maxval[n] = 1e12
                try:
                    self.minval[n] = np.array(self.v_minval)[n]
                except:
                    try:
                        self.minval[n] = np.array(self.v_minval)[0]
                    except:
                        self.minval[n] = 1e12
        else:
            self.value[0] = float(self.v_value)
            self.maxval[0] = float(self.v_maxval)
            self.minval[0] = float(self.v_minval)
        if self.isinteger:
            self.value = self.value.astype(int)
            self.maxval = self.maxval.astype(int)
            self.minval = self.minval.astype(int)
        base = QWidget(parent)
        label = QLabel(self.v_name, base)
        label2 = QLabel(self.v_unit, base)
        label.setToolTip(self.tooltip)
        label2.setToolTip(self.tooltip)
        tickbox = QCheckBox(self.box_label, base)
        tickbox.setCheckState(Qt.CheckState.Checked)
        tickbox.stateChanged.connect(self.tickerChecked)
        tickbox.setToolTip("For input variables, keep FIXED ticked. For values you want to have calculated, remove the tick.")
        fields = []
        fbase = QWidget(base)
        flay = QHBoxLayout(fbase)
        for ln in range(self.veclen):
            inlin = QLineEdit(str(self.value[ln]), base)
            if self.isinteger:
                dval = MyIntValidator(self.minval[ln], self.maxval[ln], inlin)
            else:
                dval = MyDoubleValidator(self.minval[ln], self.maxval[ln], self.prec, inlin)
            inlin.setValidator(dval)
            inlin.setStyleSheet(input_stylesheet)
            # inlin.setMinimumHeight(20)
            inlin.setText(str(self.value[ln]))
            inlin.setToolTip(self.tooltip)
            if self.isinteger:
                inlin.returnPressed.connect(self.updateInt)
                inlin.editingFinished.connect(self.updateInt)
            else:
                inlin.returnPressed.connect(self.update)
                inlin.editingFinished.connect(self.update)
            flay.addWidget(inlin)
            fields.append(inlin)
        self.fields = fields
        self.defvalue = copy.deepcopy(self.value)
        if grid is None:
            layout = QHBoxLayout(base)
            # infield.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Preferred)
            layout.addWidget(label)
            layout.addWidget(fbase)
            layout.addWidget(label2)
            layout.addWidget(tickbox)
        else:
            nGridRows = grid.rowCount()
            grid.addWidget(label,  nGridRows,  0)
            grid.addWidget(fbase,  nGridRows,  1)
            grid.addWidget(label2,  nGridRows,  2)
            grid.addWidget(tickbox, nGridRows, 3)
        if self.isinteger:
            self.updateInt()
        else:
            self.update()
    @pyqtSlot(int)
    def tickerChecked(self, newstate):
        if newstate >0:
            self.enableChanges()
            self.is_input = True
            self.state_changed.emit()
        else:
            self.disableChanges()
            self.is_input = False
            self.state_changed.emit()
    @pyqtSlot()
    def enableChanges(self):
        for n,  tfield in enumerate(self.fields):
            tfield.setReadOnly(False)
            tfield.setStyleSheet(input_stylesheet)
    @pyqtSlot()
    def disableChanges(self):
        for n,  tfield in enumerate(self.fields):
            tfield.setReadOnly(True)
            tfield.setStyleSheet(output_stylesheet)
    @pyqtSlot()
    def update(self):
        for n,  tfield in enumerate(self.fields):
            val = tfield.text()
            try:
                val = float(val)
            except:
                val = self.defvalue[n]
                tfield.setText(str(val))
#            else:
#                if val > self.maxval[n]:
#                    val = self.maxval[n]
#                    tfield.setText(str(val))
#                elif val < self.minval[n]:
#                    val = self.minval[n]
#                    tfield.setText(str(val))
            self.value[n] = val
        # print("update was called")
        self.values_changed.emit()
    @pyqtSlot()
    def updateInt(self):
        for n,  tfield in enumerate(self.fields):
            val = tfield.text()
            try:
                val = int(float(val))
            except:
                val = self.defvalue[n]
                tfield.setText(str(val))
#            else:
#                if val > self.maxval[n]:
#                    val = self.maxval[n]
#                    tfield.setText(str(val))
#                elif val < self.minval[n]:
#                    val = self.minval[n]
#                    tfield.setText(str(val))
            self.value[n] = val
        # print("updateInt was called")
        self.values_changed.emit()
    def returnValue(self):
        return self.value
    def takeValue(self,  newvalue):
        if newvalue is None:
            return None
        for nn in range(len(newvalue)):
            try:
                tval = float(newvalue[nn])
            except:
                continue
            if self.isinteger:
                tval = int(tval)
            if tval > self.maxval[nn]:
                tval = self.maxval[nn]
            if tval < self.minval[nn]:
                tval = self.minval[nn]
            self.value[nn] = tval
            # print(self.v_name, "about to set the ",  nn,  "field to",  tval)
            if self.fields[nn].isReadOnly():
                self.fields[nn].setReadOnly(False)
                self.fields[nn].setText(str(round(tval, self.prec)))
                self.fields[nn].setReadOnly(True)
            else:
                self.fields[nn].setText(str(round(tval, self.prec)))

class VarBox(QObject):
    values_changed = pyqtSignal()
    pass_values = pyqtSignal(object)
    def __init__(self, parent, setup_variables,  gname,  prec_override = 9):
        super().__init__(parent)
        self.base = QGroupBox(gname,  parent)
        self.base.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Maximum)
        self.glay = QGridLayout(self.base)
        self.sample_vars = {}
        self.var_names = []
        self.value_dict = {}
        for v in setup_variables:
            self.sample_vars[v['Key']] = InputVariable(self.base, v,  self.glay,  prec = prec_override)
            self.var_names.append(v['Key'])
            self.value_dict[v['Key']] = self.sample_vars[v['Key']].returnValue()
            self.sample_vars[v['Key']].values_changed.connect(self.updateValues)
    @pyqtSlot()
    def enableChanges(self):
        for v in self.var_names:
            self.sample_vars[v].enableChanges()
    @pyqtSlot()
    def disableChanges(self):
        for v in self.var_names:
            self.sample_vars[v].disableChanges()
    @pyqtSlot()
    def updateValues(self):
        for vn in self.var_names:
            self.value_dict[vn] = self.sample_vars[vn].returnValue()
        # print("SampleBox just did an updateValues")
        self.values_changed.emit()
        self.pass_values.emit(self.value_dict)
    @pyqtSlot()
    def updateOutputs(self):
        for vn in self.var_names:
            self.sample_vars[vn].takeValue(self.value_dict[vn])
        # print("SampleBox just did an updateValues")
        self.values_changed.emit()
    def returnValues(self):
        return (self.var_names,  self.value_dict)
    @pyqtSlot(object)
    def receiveValues(self, input):
        value_dict = input
        self.blockSignals(True)
        for vn in value_dict.keys():
            newkey = str(vn).split('-')[-1].split()[0].lower()
            # print("New Key is",  newkey, " old key is",  vn,  " value at the old key",  value_dict[vn])
            for altkey in self.var_names:
                if newkey in altkey:
                    self.sample_vars[altkey].takeValue(value_dict[vn])
        self.blockSignals(False)
        self.updateValues()
        self.updateOutputs()
        self.values_changed.emit()
    def takeValues(self,  var_names, value_dict):
        for vn in var_names:
            if vn in self.var_names:
                self.sample_vars[vn].takeValue(value_dict[vn])
        self.values_changed.emit()

class StrBox(QObject):
    values_changed = pyqtSignal()
    pass_values = pyqtSignal(object)
    def __init__(self, parent, setup_variables,  gname):
        super().__init__(parent)
        self.base = QGroupBox(gname,  parent)
        # self.base.setSizePolicy(QSizePolicy.Maximum,QSizePolicy.Maximum)
        self.base.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.glay = QGridLayout(self.base)
        self.sample_vars = {}
        self.var_names = []
        self.value_dict = {}
        for kk in setup_variables.keys():
            kname = str(kk)
            value = str(setup_variables[kk])
            self.sample_vars[kname] = StringVariable(self.base, kname, value,  self.glay)
            self.var_names.append(kname)
            self.value_dict[kname] = value
            self.sample_vars[kname].values_changed.connect(self.updateValues)
    @pyqtSlot()
    def updateValues(self):
        for vn in self.var_names:
            self.value_dict[vn] = self.sample_vars[vn].returnValue()
        # print("SampleBox just did an updateValues")
        self.values_changed.emit()
        self.pass_values.emit(self.value_dict)
    @pyqtSlot()
    def updateOutputs(self):
        for vn in self.var_names:
            self.sample_vars[vn].takeValue(self.value_dict[vn])
        # print("SampleBox just did an updateValues")
        self.values_changed.emit()
    def returnValues(self):
        return (self.var_names,  self.value_dict)
    @pyqtSlot(object)
    def receiveValues(self, input):
        value_dict = input
        self.blockSignals(True)
        for vn in value_dict.keys():
            newkey = str(vn).split('-')[-1].split()[0].lower()
            # print("New Key is",  newkey, " old key is",  vn,  " value at the old key",  value_dict[vn])
            for altkey in self.var_names:
                if newkey in altkey:
                    self.sample_vars[altkey].takeValue(value_dict[vn])
        self.blockSignals(False)
        self.updateValues()
        self.updateOutputs()
        self.values_changed.emit()
    def takeValues(self,  var_names, value_dict):
        for vn in var_names:
            if vn in self.var_names:
                self.sample_vars[vn].takeValue(value_dict[vn])

class TickVar(QWidget):
    def __init__(self, parent, init_var = 0.0, prec_override = 9):
        super().__init__(parent)
        self.blayout = QHBoxLayout(self)
        self.prec = prec_override
        self.val = init_var
        self.isOn = True
        self.tfield = QLineEdit(self)
        self.cbox = QCheckBox(self)
        dval = MyDoubleValidator(-1e60, 1e60, self.prec, self.tfield)
        self.tfield.setValidator(dval)
        self.tfield.setText(str(round(init_var, self.prec)))
        self.blayout.addWidget(self.tfield)
        self.blayout.addWidget(self.cbox)
    def takeValues(self, newval = 0.0):
        self.tfield.setText(str(round(newval, self.prec)))
    def returnValues(self):
        val = float(self.tfield.text())
        flag = self.cbox.isChecked()
        return val, flag

class RefineBox(QObject):
    values_changed = pyqtSignal()
    def __init__(self, parent, setup_variables=[("Some parameters", 1)], 
                              gname="Profile refinement",  prec_override = 9):
        super().__init__(parent)
        self.base = QGroupBox(gname,  parent)
        self.base.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Maximum)
        self.glay = QFormLayout(self.base)
        self.vars = []
        for entry in setup_variables:
            strip = QWidget(self.base)
            slay = QHBoxLayout(strip)
            for n in range(entry[1]):
                temp = TickVar(strip)
                self.vars.append(temp)
                slay.addWidget(temp)
            self.glay.addRow(entry[0], strip)
    def returnValues(self):
        vals, flags = [], []
        for item in self.vars:
            temp = item.returnValues()
            vals.append(temp[0])
            flags.append(temp[1])
        return vals, flags
    @pyqtSlot(object)
    def takeValues(self, vals):
        for n, v in enumerate(vals):
            self.vars[n].takeValues(v)

class CheckGroup(QButtonGroup):
    new_values = pyqtSignal(object)
    def __init__(self, parent, setup_variables=[("Some parameters", 1)], 
                              gname="Profile refinement", max_items_per_row = 2):
        super().__init__(parent)
        self.base = QGroupBox(gname,  parent)
        self.base.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Maximum)
        self.glay = QGridLayout(self.base)
        self.setExclusive(False)
        self.vars = []
        for n, entry in enumerate(setup_variables):
            col_number = n%max_items_per_row
            row_number = n//max_items_per_row
            if entry[1]:
                state = Qt.CheckState.Checked
            else:
                state = Qt.CheckState.Unchecked
            title = QLabel(entry[0], self.base)
            cbox = QCheckBox(self.base)
            cbox.setCheckState(state)
            self.glay.addWidget(cbox, row_number, 2*col_number)
            self.glay.addWidget(title, row_number, 2*col_number + 1)
            self.addButton(cbox)
        self.buttonClicked.connect(self.returnValues)
    @pyqtSlot()
    def returnValues(self):
        result = []
        for cbox in self.buttons():
            result.append(cbox.isChecked())
        self.new_values.emit(result)
        return result

class TwoWayBox(QObject):
    values_changed = pyqtSignal()
    def __init__(self, parent, setup_variables,  gname,  prec_override = 9, blabel = '<- FIXED!'):
        super().__init__(parent)
        self.base = QGroupBox(gname,  parent)
        self.base.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Maximum)
        self.glay = QGridLayout(self.base)
        self.sample_vars = {}
        self.var_names = []
        self.value_dict = {}
        self.is_input = []
        for v in setup_variables:
            self.sample_vars[v['Key']] = IOVariable(self.base, v,  self.glay,  prec = prec_override, 
                                                                         blabel = blabel)
            self.var_names.append(v['Key'])
            self.value_dict[v['Key']] = self.sample_vars[v['Key']].returnValue()
            self.is_input.append(True)
            self.sample_vars[v['Key']].values_changed.connect(self.updateValues)
            self.sample_vars[v['Key']].state_changed.connect(self.updateFlags)
    @pyqtSlot()
    def enableChanges(self):
        for v in self.var_names:
            self.sample_vars[v].enableChanges()
    @pyqtSlot()
    def disableChanges(self):
        for v in self.var_names:
            self.sample_vars[v].disableChanges()
    @pyqtSlot()
    def updateFlags(self):
        for nnn, vn in enumerate(self.var_names):
            self.is_input[nnn] = self.sample_vars[vn].is_input
        # print("SampleBox just did an updateValues")
        # self.values_changed.emit()
        self.updateValues()
    @pyqtSlot()
    def updateValues(self):
        for nnn, vn in enumerate(self.var_names):
            if self.is_input[nnn]:
                self.value_dict[vn] = self.sample_vars[vn].returnValue()
            else:
                self.value_dict[vn] = None
        # print("SampleBox just did an updateValues")
        self.values_changed.emit()
    @pyqtSlot()
    def updateOutputs(self):
        for nnn, vn in enumerate(self.var_names):
            if not self.is_input[nnn]:
                self.sample_vars[vn].takeValue(self.value_dict[vn])
        # print("SampleBox just did an updateValues")
        self.values_changed.emit()
    def returnValues(self):
        return (self.var_names,  self.value_dict)
    def returnFlags(self):
        return (self.var_names,  self.is_input)
    @pyqtSlot(object)
    def receiveValues(self, input):
        value_dict = input
        for vn in value_dict.keys():
            newkey = str(vn).split('-')[-1].split()[0].lower()
            # print("New Key is",  newkey, " old key is",  vn,  " value at the old key",  value_dict[vn])
            for altkey in self.var_names:
                if newkey in altkey:
                    self.sample_vars[altkey].takeValue(value_dict[vn])
        self.updateValues()
        self.updateOutputs()
        self.values_changed.emit()
    def takeValues(self,  var_names, value_dict):
        for vn in var_names:
            if vn in self.var_names:
                self.sample_vars[vn].takeValue(value_dict[vn])
        self.values_changed.emit()

class ComBox(QObject):
    values_changed = pyqtSignal()
    pass_values = pyqtSignal(object)
    def __init__(self, parent, setup_variables,  gname):
        super().__init__(parent)
        self.base = QGroupBox(gname,  parent)
        # self.base.setSizePolicy(QSizePolicy.Maximum,QSizePolicy.Maximum)
        self.base.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.glay = QGridLayout(self.base)
        self.sample_vars = {}
        self.var_names = []
        self.value_dict = {}
        for kk in setup_variables.keys():
            kname = str(kk)
            vallist = str(setup_variables[kk])
            tempBox = QComboBox(self.base)
            tempBox.setEditable(True)
            tempBox.addItems(vallist)
            self.sample_vars[kname] = tempBox
            self.var_names.append(kname)
            self.value_dict[kname] = tempBox.currentText()
            self.sample_vars[kname].currentTextChanged.connect(self.updateValues)
    @pyqtSlot()
    def updateValues(self):
        for vn in self.var_names:
            self.value_dict[vn] = self.sample_vars[vn].currentText()
        # print("SampleBox just did an updateValues")
        self.values_changed.emit()
        self.pass_values.emit(self.value_dict)
    def returnValues(self):
        return (self.var_names,  self.value_dict)
