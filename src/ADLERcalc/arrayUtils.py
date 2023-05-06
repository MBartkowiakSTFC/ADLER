
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
from scipy.optimize import leastsq

from ADLERcalc.qtObjects import MergeCurves
from ADLERcalc.xasUtils import guess_XAS_xaxis

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


def scaling_fit(args, data, to_match):
    temp = args[0] * data + args[1]
    return temp - to_match

  
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


def primitive_rebin(inarray,  redrate = 5):
    tlen = len(inarray)
    nrow = int(math.floor(tlen / redrate))
    newarray = inarray[:nrow*redrate].reshape((nrow, redrate))
    lastone = inarray[nrow*redrate:]
    result = np.concatenate([newarray.sum(1), [lastone.sum()]])
    return result


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
