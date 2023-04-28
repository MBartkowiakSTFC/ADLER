
import numpy as np

def load_file(fname, bpp, cray, poly = None):
    fpath, shortname = os.path.split(fname)
    if "." in shortname:
        nameroot = ".".join(shortname.split(".")[:-1])
    else:
        nameroot = shortname
    data, header = ReadAndor(fname)
    tadded_header, units_for_header = load_datheader(os.path.join(fpath, nameroot+".dat"))
    tadded_log = load_datlog(os.path.join(fpath, nameroot+".xas"))
    # data = data[:,lcut:hcut]
    if cray > 0.0:
        data = RemoveCosmics(data, NStd = cray)
    if bpp is None:
        bpp = np.percentile(data, 10).mean()*0.99
    data -= bpp
    return data, tadded_header,  tadded_log

def load_filelist(flist,  bpp,  cray,  poly):
    if len(flist) == 0:
        return None,  None,  None
    else:
        temps = []
        shortnames = []
        headers,  logs = [],  []
        header,  vallog = {},  {}
        textheader = ""
        for tname in flist:
            tdata, theader, tlog = load_file(tname, bpp, cray, poly = poly)
            temps.append(tdata)
            headers.append(theader)
            logs.append(tlog)
            aaa, bbb = os.path.split(tname)
            shortnames.append(bbb)
        for hd in headers:
            hk = hd.keys()
            for k in hk:
                if k in header.keys():
                    header[k].append(hd[k])
                else:
                    header[k] = []
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
        for hk in header.keys():
            tarr = np.array(header[hk])
            tmin,  tmax,  tmean,  tstd = tarr.min(),  tarr.max(),  tarr.mean(),  tarr.std()
            textheader += " ".join(['#', hk] + [str(round(x,4)) for x in [tmin, tmax, tmean, tstd]] + ['\n'])
        data = np.array(temps)# .sum(0)
        # it is better to return the files separately.
        # we can always sum them up outside of this function
        # or do something else to them first.
        return data,  textheader,  vallog


def precise_merge(curves, outname = 'best_merge.txt'):
    datalist, energylist, unitlist, dictlist = [],[],[], []
    for fname in curves:
        a,b,c,d = read_1D_curve_extended(fname)
        datalist.append(a)
        energylist.append(b)
        unitlist.append(c)
        dictlist.append(d)
    # energies = np.array(energylist).mean()
    # I ignore the units and energies. Let's assume I put the correct files in the list.
    minx, maxx, stepx = 1e5,-1e5,-1.0
    for dat in datalist:
        minx = min(minx, dat[:,0].min())
        maxx = max(maxx, dat[:,0].max())
        step = (dat[1:,0] - dat[:-1,0]).mean()
        stepx = max(stepx, step)
    newx = np.arange(minx, maxx + 0.1*stepx, stepx)
    target = np.zeros((len(newx),2))
    target[:,0] = newx
    for dat in datalist:
        target = merge2curves(dat,target)
    WriteEnergyProfile(outname, target, [])
    return target

