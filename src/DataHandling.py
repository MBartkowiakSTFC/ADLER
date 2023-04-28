
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
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from astropy.io import fits as fits_module

ROAR_LJOKELSOI = 32.518



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
                key, val = toks[0].strip(' '),  toks[1].strip(' ')
                self.fitting_params_channels[key] = float(val)
        for line in pdict['fitting_params_absEnergy']:
            toks = line.split()
            if len(toks) == 2:
                key, val = toks[0].strip(' '),  toks[1].strip(' ')
                self.fitting_params_absEnergy[key] = float(val)
        for line in pdict['fitting_params_eV']:
            toks = line.split()
            if len(toks) == 2:
                key, val = toks[0].strip(' '),  toks[1].strip(' ')
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
                    newln = tkey + " $ " + str(element)
                    target.write(' '.join(['#HEADER#',newln.strip().strip('\n'),'\n']))
                else:
                    for gk in element.keys():
                        item = element[gk]
                        innerkey = str(gk)
                        newln = tkey + " # " + innerkey + " $ " + str(item)
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
        header,  vallog = defaultdict(lambda:[ -1]), defaultdict(lambda: np.array([]))
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
