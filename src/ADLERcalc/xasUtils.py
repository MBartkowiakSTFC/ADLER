
import numpy as np

def single_n2_profile(params, xarr, fixed_lorentz = 0.12, yarr = None, extraweight = False):
    num_peaks = int((len(params) - 3)/2)
    profile = np.zeros(xarr.shape)
    dists = []
    scalefac = 1.0
    for n in range(num_peaks):
        profile += my_voigt(xarr, abs(params[2*n]), abs(params[2*n+1]), abs(params[-3]), fixed_lorentz)
        dists.append(params[2*n+1])
    dists = np.sort(dists)
    dists = dists[1:] - dists[:-1]
    profile += params[-2]*xarr + params[-1]
    if yarr is not None:
        profile -= yarr
        if extraweight:
            if len(dists) > 0:
                scalefac = 1.0 + max(0.0, 0.15 - dists.min())
    return profile * scalefac
    
def single_neon_profile(params, xarr, fixed_lorentz = 0.254, yarr = None, extraweight = False):
    num_peaks = int((len(params) - 5)/2)
    profile = np.zeros(xarr.shape)
    scalefac = 1.0
    for n in range(num_peaks):
        profile += my_voigt(xarr, abs(params[2*n]), abs(params[2*n+1]), abs(params[-5]), fixed_lorentz)
    profile += abs(params[-4])*np.arctan((xarr-params[-3])/params[-2]) + params[-1]
    if yarr is not None:
        profile -= yarr
        if extraweight:
            scalefac = 1.0 + max(0.0, 870 - params[-3])
    return profile * scalefac
    
def single_edge_profile(params, xarr, fixed_lorentz = 0.254, yarr = None):
    profile = np.zeros(xarr.shape)
    profile += params[-4]*np.arctan((xarr-params[-3])/params[-2]) + params[-1]
    if yarr is not None:
        profile -= yarr
    return profile

def fit_edge_profile(num_arr,  nlines = 0):
    mid_y = (num_arr[:, 1].max()+num_arr[:, 1].min())/2.0
    crit = np.abs(num_arr[:, 1] - mid_y)
    ref = crit.min()
    mid_x = num_arr[:, 0][np.where(crit==ref)].mean()
    step = np.abs(num_arr[1:, 0] - num_arr[:-1, 0]).mean()
    params = [num_arr[-1, 1], mid_x, 10*step, num_arr[0, 1]]
    pfit, pcov, infodict, errmsg, success = leastsq(single_edge_profile, params, args = (num_arr[:, 0], 0.254, num_arr[:, 1]), full_output =1)
    final_profile = single_edge_profile(pfit, num_arr[:, 0], 0.254)
    GOF = ((final_profile - num_arr[:, 1])**2).sum()**0.5/ np.abs(num_arr[:, 1]).sum()
    print('GOF: ',  GOF, ((final_profile - num_arr[:, 1])**2).sum()**0.5, np.abs(num_arr[:, 1]).sum())
    return final_profile, pfit[1],  pfit[2],  GOF

def fit_n2_gas(inp_num_arr, fixed_gamma = 0.06):
    peak_list = np.array([400.76, 400.996, 401.225, 401.45, 401.66, 401.88, 402.1, 402.29, 402.48])
    params = []
    # fixed_gamma = 0.012
    seq = np.argsort(inp_num_arr[:, 0])
    num_arr = inp_num_arr[seq]
    temp_prof = interp1d(num_arr[:, 0],  num_arr[:, 1])
    for peak in peak_list:
        if peak >= num_arr[:, 0].min() and peak <= num_arr[:, 0].max():
            params += [temp_prof(peak), peak]
    params += [0.01, 0.0, 0.0]
    pfit, pcov, infodict, errmsg, success = leastsq(single_n2_profile, params, args = (num_arr[:, 0], fixed_gamma, num_arr[:, 1],  True), full_output =1)
    final_profile = single_n2_profile(pfit, num_arr[:, 0], fixed_gamma)
    # determine the fit quality
    GOF = ((final_profile - num_arr[:, 1])**2 ).sum()**0.5/ np.abs(num_arr[:, 1]).sum()
    print('GOF: ',  GOF, ((final_profile - num_arr[:, 1])**2 ).sum()**0.5, np.abs(num_arr[:, 1]).sum() )
    resline = np.concatenate([[GOF], pfit])
    results = [resline]
    if 1.0/GOF < 50:
        for xx in range(3):
            newparams = np.array(params)
            plen = len(newparams)
            rands = np.random.rand(plen)
            newparams[0:-3:2] *= rands[0:-3:2]
            newparams[1:-3:2] += 0.1*(rands[1:-3:2] - 0.5)
            pfit, pcov, infodict, errmsg, success = leastsq(single_n2_profile, params, args = (num_arr[:, 0], fixed_gamma, num_arr[:, 1],  True), full_output =1)
            final_profile = single_n2_profile(pfit, num_arr[:, 0], fixed_gamma)
            # determine the fit quality
            GOF = ((final_profile - num_arr[:, 1])**2 ).sum()**0.5/ np.abs(num_arr[:, 1]).sum()
            resline = np.concatenate([[GOF], pfit])
            results.append(resline)
        results = np.array(results)
        results = results[np.argsort(results[:, 0])]
        pfit = results[0, 1:]
        final_profile = single_n2_profile(pfit, num_arr[:, 0], fixed_gamma)
        # determine the fit quality
        GOF = ((final_profile - num_arr[:, 1])**2 ).sum()**0.5/ np.abs(num_arr[:, 1]).sum()
        print('best GOF: ',  GOF, ((final_profile - num_arr[:, 1])**2 ).sum()**0.5, np.abs(num_arr[:, 1]).sum() )
    # final_profile = single_n2_profile(params, num_arr[:, 0], fixed_gamma)
    # gauss_sigma = params[-3]
    # peak_pos = params[1:-3:2]
    gauss_sigma = pfit[-3]
    lorentz_gamma = fixed_gamma
    peak_pos = pfit[1:-3:2]
    return final_profile, gauss_sigma, lorentz_gamma, peak_pos,  GOF

def fit_neon_gas(inp_num_arr,  fixed_gamma = 0.126):
    peak_list = np.array([867.2, 868.8, 869.36, 869.6,  869.95, 870.23, 870.42, 870.60])
    atan2_params = [870.15, 0.125]
    # y = y0 + A*atan((x-xc)/w)
    params = []
    seq = np.argsort(inp_num_arr[:, 0])
    num_arr = inp_num_arr[seq]
    temp_prof = interp1d(num_arr[:, 0],  num_arr[:, 1])
    for peak in peak_list[:1]:
        if peak >= num_arr[:, 0].min() and peak <= num_arr[:, 0].max():
            params += [temp_prof(peak), peak]
        else:
            params += [num_arr[:, 1].max(), num_arr[:, 0].mean()]
    for peak in peak_list[1:]:
        if peak >= num_arr[:, 0].min() and peak <= num_arr[:, 0].max():
            params += [temp_prof(peak), peak]
    if len(params) == 2:
        params[0] = num_arr[:, 1].max()
        params[1] = num_arr[:, 0][np.where(num_arr[:, 1] == params[0])][0]
        params += [0.01, 0.0,  atan2_params[0], atan2_params[1], 0.0]
    else:
        params += [0.01, num_arr[-1, 1],  atan2_params[0], atan2_params[1], num_arr[0, 1]]
    pfit, pcov, infodict, errmsg, success = leastsq(single_neon_profile, params, args = (num_arr[:, 0], fixed_gamma, num_arr[:, 1],  True), full_output =1)
    final_profile = single_neon_profile(pfit, num_arr[:, 0], fixed_gamma)
    # determine the fit quality
    GOF = ((final_profile - num_arr[:, 1])**2).sum()**0.5/ np.abs(num_arr[:, 1]).sum()
    print('GOF: ',  GOF, ((final_profile - num_arr[:, 1])**2 ).sum()**0.5, np.abs(num_arr[:, 1]).sum() )
    #
    resline = np.concatenate([[GOF], pfit])
    results = [resline]
    if 1.0/GOF < 50:
        for xx in range(3):
            newparams = np.array(params)
            plen = len(newparams)
            rands = np.random.rand(plen)
            newparams[0:-5:2] *= rands[0:-5:2]
            newparams[1:-5:2] += 0.1*(rands[1:-5:2] - 0.5)
            pfit, pcov, infodict, errmsg, success = leastsq(single_neon_profile, params, args = (num_arr[:, 0], fixed_gamma, num_arr[:, 1],  True), full_output =1)
            final_profile = single_neon_profile(pfit, num_arr[:, 0], fixed_gamma)
            # determine the fit quality
            GOF = ((final_profile - num_arr[:, 1])**2 ).sum()**0.5/ np.abs(num_arr[:, 1]).sum()
            resline = np.concatenate([[GOF], pfit])
            results.append(resline)
        results = np.array(results)
        results = results[np.argsort(results[:, 0])]
        pfit = results[0, 1:]
        final_profile = single_neon_profile(pfit, num_arr[:, 0], fixed_gamma)
        # determine the fit quality
        GOF = ((final_profile - num_arr[:, 1])**2 ).sum()**0.5/ np.abs(num_arr[:, 1]).sum()
        print('best GOF: ',  GOF, ((final_profile - num_arr[:, 1])**2 ).sum()**0.5, np.abs(num_arr[:, 1]).sum() )
    gauss_sigma = pfit[-5]
    lorentz_gamma = fixed_gamma
    peak_pos = pfit[1:-5:2]
    return final_profile, gauss_sigma, lorentz_gamma, peak_pos,  GOF
