
import math
import numpy as np
has_voigt = True
try:
    from scipy.special import voigt_profile
except:
    from scipy.special import wofz
    has_voigt = False
from scipy.optimize import leastsq, shgo, minimize
from scipy.interpolate import interp1d
from scipy.fftpack import rfft, irfft, fftfreq


gauss_denum = 2.0 * (2.0 * math.log(2.0))**0.5
gauss_const = 1/((2.0*math.pi)**0.5)

oldval = 0.0

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
