
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

import numpy as np
import os
import gzip
import copy
from collections import defaultdict

import yaml
from astropy.io import fits as fits_module

ROAR_LJOKELSOI = 32.518


def decompress_array(compdata, shape, dtype):
    bdata = gzip.decompress(compdata)
    linedata = np.frombuffer(bdata, dtype=dtype)
    truedata = linedata.reshape(shape)
    return truedata

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

