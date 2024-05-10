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

from scipy.optimize import leastsq

from ADLER.ADLERcalc.fitUtils import (
    gaussian,
    fit_gaussian,
    fit_gaussian_fixedbase,
    chisquared,
    polynomial,
)
from ADLER.ADLERcalc.arrayUtils import discrete_rebin
from ADLER.ADLERcalc.qtObjects import MergeCurves


def elastic_line(profile, background, fixwidth=None, olimits=(None, None)):
    """
    As an initial guess, this function will assume that the
    elastic line is at the point with the highest intensity.
    Otherwise, an override for the peak centre can be provided.
    """
    chi2 = -1.0
    if len(background) > 1:
        zline, zerr = background[:, 1].mean(), background[:, 1].std()
    else:
        zline = profile[:, 1].min()
        zerr = abs(profile[:, 1].max() - profile[:, 1].min()) ** 0.5
    temp = profile.copy()
    temp[:, 1] -= zline
    maxval = temp[:, 1].max()
    nlimits = [0, -1]
    if olimits[0] is not None and olimits[1] is not None:
        boolarr1 = temp[:, 0] > olimits[0]
        if boolarr1.any() and not boolarr1.all():
            nlimits[0] = max(np.argmax(boolarr1) - 1, 0)
        else:
            nlimits[0] = 0
        boolarr2 = temp[:, 0] > olimits[1]
        if boolarr2.any() and not boolarr2.all():
            nlimits[1] = min(np.argmax(boolarr2), len(temp) - 1)
        else:
            nlimits[1] = len(temp) - 1
        if abs(nlimits[1] - nlimits[0]) < 5:
            nlimits[1] += 1
        if abs(nlimits[1] - nlimits[0]) < 5:
            nlimits[0] -= 1
        if abs(nlimits[1] - nlimits[0]) < 5:
            nlimits[1] += 1
        if abs(nlimits[1] - nlimits[0]) < 5:
            nlimits[0] -= 1
        # centre = (temp[nlimits[0],0] + temp[nlimits[1],0])/2
        nmaxval = temp[nlimits[0] : nlimits[1], 1].max()
        centre = temp[nlimits[0] : nlimits[1], 0][
            np.where(temp[nlimits[0] : nlimits[1], 1] == nmaxval)
        ][-1]
    else:
        centre = temp[:, 0][np.where(temp[:, 1] == maxval)][-1]
    tempcrit = np.abs(temp[:, 0] - centre)
    crit = tempcrit.min()
    (cindex,) = np.where(tempcrit == crit)
    if len(cindex) == 1:
        cindex = int(cindex)
    else:
        cindex = int(len(temp) / 2)
    table = np.zeros((90, 2))
    table[:, 0] = np.arange(len(table)) + 1
    for x in np.arange(len(table)):
        try:
            table[x, 1] = temp[cindex - table[x, 0], 1] + temp[cindex + table[x, 0], 1]
        except IndexError:
            break
    criterion = np.abs(table[:, 1] - maxval)
    fwhm = int(table[:, 0][np.where(criterion == criterion.min())].mean())
    step = np.array([fwhm])[0]
    if olimits[0] is not None and olimits[1] is not None:
        subset = profile[nlimits[0] : nlimits[1]]
    else:
        subset = profile[-step + cindex : step + cindex]
        counter, running = 0, 1
        while (len(subset) < 350) and running:
            step += 1
            subset = profile[-step + cindex : step + cindex]
            if (
                (subset[:3, 1].mean() + subset[-3:, 1].mean() - 2 * zline) / zerr
            ) ** 2 < 2.0:
                counter += 1
            if counter > 8:
                running = 0
            # print(((subset[:3,1].mean() + subset[-3:,1].mean() - 2*zline)/zerr)**2)
    if fixwidth is not None:
        subset = profile[-int(fixwidth) + cindex : int(fixwidth) + cindex]
        maxval = subset[:, 1].max()
        zline = subset[:, 1].min()
        fwhm = subset[:, 0].std() / 2.0
        centre = subset[:, 0].mean()
        if len(subset) < 5:
            print("Not enough points in the curve.")
            return None, None, None
        pfit, pcov, infodict, errmsg, success = leastsq(
            fit_gaussian,
            np.array([maxval, fwhm, centre, zline]),
            args=(subset,),
            full_output=1,
        )
        if pcov is not None:
            s_sq = (fit_gaussian(pfit, subset) ** 2).sum() / (len(subset) - len(pfit))
            a, FWHM, centre, zeroline = pfit
            tpeak = gaussian(a, FWHM, centre, subset[:, 0]) + zeroline
            chi2 = chisquared(tpeak, subset, zeroline)
            pcov = pcov * s_sq
    else:
        if len(subset) < 5:
            print("Not enough points in the curve.")
            return None, None, None
        pfit, pcov, infodict, errmsg, success = leastsq(
            fit_gaussian_fixedbase,
            np.array([maxval, fwhm, centre]),
            args=(subset, zline),
            full_output=1,
        )
        if pcov is not None:
            s_sq = (fit_gaussian_fixedbase(pfit, subset, zline) ** 2).sum() / (
                len(subset) - len(pfit)
            )
            a, FWHM, centre = pfit
            tpeak = gaussian(a, FWHM, centre, subset[:, 0]) + zline
            chi2 = chisquared(tpeak, subset, zline)
            pcov = pcov * s_sq
    error = []
    for i in range(len(pfit)):
        try:
            error.append(np.absolute(pcov[i][i]) ** 0.5)
        except:
            error.append(0.00)
    optimised = [pfit, np.array(error)]
    a, b, c = optimised[0][0], optimised[0][1], optimised[0][2]
    newx = np.linspace(subset[0, 0], subset[-1, 0], len(subset) * 10)
    if fixwidth is not None:
        peak = gaussian(a, b, c, newx) + optimised[0][3]
    else:
        peak = gaussian(a, b, c, newx) + zline
    return optimised, np.column_stack([newx, peak]), chi2


def elastic_line_anyx(
    profile, background, fixwidth=None, olimits=(None, None), init_fwhm=0.05
):
    """
    As an initial guess, this function will assume that the
    elastic line is at the point with the highest intensity.
    Otherwise, an override for the peak centre can be provided.
    """
    chi2 = -1.0
    if len(background) > 1:
        zline, zerr = background[:, 1].mean(), background[:, 1].std()
    else:
        zline = profile[:, 1].min()
        zerr = abs(profile[:, 1].max() - profile[:, 1].min()) ** 0.5
    temp = profile.copy()
    temp[:, 1] -= zline
    maxval = temp[:, 1].max()
    if olimits[0] is not None and olimits[1] is not None:
        # centre = ((temp[:, 0][np.where(crit1==min1)] + temp[:, 0][np.where(crit2==min2)])/2)[0]
        if olimits[0] <= temp[:, 0].min():
            if olimits[1] >= temp[:, 0].max():
                subset = profile
            else:
                subset = profile[: np.argmax(temp[:, 0] >= olimits[1])]
        else:
            if olimits[1] >= temp[:, 0].max():
                subset = profile[np.argmax(temp[:, 0] >= olimits[0]) :]
            else:
                subset = profile[
                    np.argmax(temp[:, 0] >= olimits[0]) : np.argmax(
                        temp[:, 0] >= olimits[1]
                    )
                ]
        if len(subset) < 2:
            return None, None, None
        centre = subset[:, 0][np.where(subset[:, 1] == subset[:, 1].max())][-1]
    else:
        subset = None
        centre = temp[:, 0][np.where(temp[:, 1] == maxval)][-1]
    tempcrit = np.abs(temp[:, 0] - centre)
    crit = tempcrit.min()
    (cindex,) = np.where(tempcrit == crit)
    if len(cindex) == 1:
        cindex = int(cindex)
    else:
        cindex = int(cindex.mean())
    table = np.zeros((90, 2))
    table[:, 0] = np.arange(len(table)) + 1
    for x in np.arange(len(table)):
        try:
            table[x, 1] = temp[cindex - table[x, 0], 1] + temp[cindex + table[x, 0], 1]
        except IndexError:
            break
    criterion = np.abs(table[:, 1] - maxval)
    fwhm = int(table[:, 0][np.where(criterion == criterion.min())].mean())
    step = np.array([fwhm])[0]
    if subset is None:
        subset = profile[-step + cindex : step + cindex]
        counter, running = 0, 1
        while (len(subset) < 350) and running:
            step += 1
            subset = profile[-step + cindex : step + cindex]
            if (
                (subset[:3, 1].mean() + subset[-3:, 1].mean() - 2 * zline) / zerr
            ) ** 2 < 2.0:
                counter += 1
            if counter > 8:
                running = 0
            # print(((subset[:3,1].mean() + subset[-3:,1].mean() - 2*zline)/zerr)**2)
    if fixwidth is not None:
        subset = profile[-int(fixwidth) + cindex : int(fixwidth) + cindex]
        maxval = subset[:, 1].max()
        zline = subset[:, 1].min()
        fwhm = subset[:, 0].std() / 2.0
        centre = subset[:, 0].mean()
        if len(subset) < 5:
            print("Not enough points in the curve.")
            return None, None, None
        pfit, pcov, infodict, errmsg, success = leastsq(
            fit_gaussian,
            np.array([maxval, fwhm, centre, zline]),
            args=(subset,),
            full_output=1,
        )
        if pcov is not None:
            s_sq = (fit_gaussian(pfit, subset) ** 2).sum() / (len(subset) - len(pfit))
            a, FWHM, centre, zeroline = pfit
            tpeak = gaussian(a, FWHM, centre, subset[:, 0]) + zeroline
            chi2 = chisquared(tpeak, subset, zeroline)
            pcov = pcov * s_sq
    else:
        if len(subset) < 5:
            print("Not enough points in the curve.")
            return None, None, None
        pfit, pcov, infodict, errmsg, success = leastsq(
            fit_gaussian_fixedbase,
            np.array([maxval, init_fwhm, centre]),
            args=(subset, zline),
            full_output=1,
        )
        if pcov is not None:
            s_sq = (fit_gaussian_fixedbase(pfit, subset, zline) ** 2).sum() / (
                len(subset) - len(pfit)
            )
            a, FWHM, centre = pfit
            tpeak = gaussian(a, FWHM, centre, subset[:, 0]) + zline
            chi2 = chisquared(tpeak, subset, zline)
            pcov = pcov * s_sq
    error = []
    for i in range(len(pfit)):
        try:
            error.append(np.absolute(pcov[i][i]) ** 0.5)
        except:
            error.append(0.00)
    optimised = [pfit, np.array(error)]
    a, b, c = optimised[0][0], optimised[0][1], optimised[0][2]
    newx = np.linspace(subset[0, 0], subset[-1, 0], len(subset) * 10)
    if fixwidth is not None:
        peak = gaussian(a, b, c, newx) + optimised[0][3]
    else:
        peak = gaussian(a, b, c, newx) + zline
    return optimised, np.column_stack([newx, peak]), chi2


def make_profile(
    inarray,
    reduction=1.0,
    xaxis=None,
    tpoolin=None,
    pbarin=None,
    maxthreads=1,
    limits=None,
):
    temparray = inarray[limits[2] : limits[3], limits[0] : limits[1]]
    profile = temparray.sum(1)
    initial_x = np.arange(len(profile)) + 1 + limits[2]
    if xaxis is None:
        final_x = np.linspace(limits[2] + 1, limits[3], int(len(profile) / reduction))
    else:
        final_x = xaxis.copy()
    final_y = np.zeros(len(final_x))
    finalprof = np.column_stack([final_x, final_y])
    profile = np.column_stack([initial_x, profile])
    # pprofile = merge2curves(profile, finalprof)
    # the_object = MergeCurves(profile,  finalprof, tpool = tpoolin, pbar = pbarin)
    the_object = MergeCurves(
        profile, finalprof, tpool=None, pbar=pbarin, mthreads=maxthreads
    )
    the_object.runit()
    pprofile = the_object.postprocess()
    return pprofile


def make_stripe(data, xlims, ylims):
    stripe = data[ylims[0] : ylims[1], xlims[0] : xlims[1]].sum(0)
    stripe = np.column_stack([np.arange(xlims[0], xlims[1]) + 1, stripe])
    return stripe


def make_histogram(data, xlims, ylims):
    temp = data[ylims[0] : ylims[1], xlims[0] : xlims[1]]
    flat = np.ravel(temp)
    ordered = np.sort(flat)
    cons_limits = (ordered[2:10].mean() - 0.5, ordered[-10:-2].mean() + 0.5)
    # nbins = max(int(round(len(flat)**0.5)),  10)
    nbins = max(int(round(cons_limits[1]) - round(cons_limits[0])) + 1, 3)
    hist, binlims = np.histogram(temp, bins=nbins, range=cons_limits)
    return hist, binlims


def curvature_profile(
    data2D, blocksize=16, percentile=70, override=None, olimits=(None, None)
):
    width = data2D.shape[1]
    segnums = int(round(width / blocksize))
    segments = [
        data2D[:, int(n * blocksize) : int((n + 1) * blocksize)].sum(1)
        for n in np.arange(segnums)
    ]
    results = []
    for num, seg in enumerate(segments):
        background = np.where(seg < np.percentile(seg, percentile))
        xaxis = np.arange(len(seg)) + 1
        prof = np.column_stack([xaxis, seg])
        back = prof[background]
        try:
            a, b, c = elastic_line(prof, back, override, olimits=olimits)
        except:
            continue
        else:
            if olimits[0] is None or olimits[1] is None:
                results.append([(num + 0.5) * blocksize, a[0][2], a[1][2]])
            elif a[0][2] > olimits[0] and a[0][2] < olimits[1]:
                results.append([(num + 0.5) * blocksize, a[0][2], a[1][2]])
    return np.array(results)


def SphericalCorrection(
    inarray, params=[1.0, 1.0, 1.0], locut=0, hicut=2048, direct_offsets=None
):
    """This is a dummy function for now.
    Later on, it will apply a function to transform the image,
    so that all the lines are straight instead of curved.
    """
    xaxis = np.arange(locut, hicut, 1) + 1
    if params[0] > 5.0 and params[1] > 5.0 and params[2] > 5.0:
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
        result[:, n] = discrete_rebin(
            inarray[:, n], offset=offsets[n], padding=6, nsubdiv=100
        )
    return result


def apply_offset_to_2D_data(inarray, locut, hicut, offset):
    """This is a dummy function for now.
    Later on, it will apply a function to transform the image,
    so that all the lines are straight instead of curved.
    """
    xaxis = np.arange(locut, hicut, 1) + 1
    result = np.zeros(inarray.shape)
    for n, i in enumerate(xaxis):
        result[:, n] = discrete_rebin(inarray[:, n], offset=offset)
    return result
