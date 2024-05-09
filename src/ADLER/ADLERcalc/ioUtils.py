
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

import numpy as np
import os
import sys

from scipy.fftpack import rfft, irfft
from astropy.io import fits as fits_module

#from ADLER.ADLERcalc.imageUtils import RemoveCosmics
#from ADLER.ADLERcalc.arrayUtils import guess_XAS_xaxis

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

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


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
    numbs = np.array(numbs).astype(int)
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

def save_array(nparray,  fname):
    target = open(fname, 'w')
    for ln in nparray:
        line = " ".join([str(x) for x in ln]) + '\n'
        target.write(line)
    target.close()



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
        count = np.zeros(len(points)).astype(int)
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
        count = np.zeros(len(points)).astype(int)
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

