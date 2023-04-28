
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

__doc__ = """Here we attempt to consolidate the information about out measurements
that is typically coming from different sources.
"""

import numpy as np
import os
import time
import sys
import copy
from PyQt6.QtCore import QAbstractTableModel,  QObject, QVariant, Qt
from PyQt6.QtCore import pyqtSlot, pyqtSignal
from PyQt6.QtGui import QStandardItemModel,  QStandardItem
from ADLERcalc import header_read
from ExperimentTree import SingleFile

LOG_PATH = os.path.join("D:", "Measurements")

# global_file_dict = {}

class PeaxisDataModel(QStandardItemModel):
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
    def add_entry_list(self, new_entry):
        templist = []
        for x in new_entry:
            templist.append(QStandardItem(str(x)))
        self.appendRow(templist)
    def add_entry_dict(self, new_entry):
        temp = len(self.col_order)*[QStandardItem("")]
        # volatile_list = copy.deepcopy(self.col_order)
        matched_list = []
#        for xx in new_entry.keys():
#            tx =str(xx).strip(": #")
#            if tx in volatile_list:
#                try:
#                    temp[n] = QStandardItem(new_entry[xx])
#                except:
#                    continue
        for n, kk in enumerate(self.col_order):
            for xx in new_entry.keys():
                # try:
                #     tempval = float(new_entry[xx])
                # except:
                #     tempval = new_entry[xx]
                tempval = new_entry[xx]
                tx = str(xx).strip(": #")
                if 'emperature' in tx:
                    toks = tempval.split('.')
                    flen = len(toks[0])
                    if flen < 3:
                        toks[0] = toks[0].zfill(3)
                    tempval = ".".join(toks)
                if kk in tx or tx in kk:
                    if tx == kk:
                        matched_list.append(n)
                        try:
                            temp[n] = QStandardItem(tempval)
                        except:
                            continue
                    elif n not in matched_list:
                        try:
                            temp[n] = QStandardItem(tempval)
                        except:
                            continue
        self.appendRow(temp)
    def scanDir(self, dpath):
        outfiles = []
        try:
            flist1 = os.scandir(dpath)
        except:
            print("Could not access the directory: ", dpath)
        else:
            for entry in flist1:
                if entry.is_file():
                    tokens = entry.name.split('.')
                    name, extension = '.'.join(tokens[:-1]), tokens[-1]
                    if extension in ['sif']:
                        outfiles.append(entry.name)
                        fullpath = os.path.join(dpath, entry.name).replace('\\','/').replace('//','/')
                        thead, tlogs = header_read(fullpath)
                        ndict = {'name':entry.name}
                        for kk in thead.keys():
                            newquay = kk.strip('# :')
                            ndict[newquay] = str(thead[kk])
                        self.add_entry_dict(ndict)
                        # global global_file_dict
                        self._file_dict[entry.name] = fullpath
                        self._header_dict[entry.name] = thead
                        self._log_dict[entry.name] = tlogs
        

class UnifiedDataHandler(QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.datapath = None
        self.logpath = None
        self.filelist = []
        self.command_history = []
        
# from PyQt6.QtWidgets import QApplication
# 
# app = QApplication([])
# app.exec_()
