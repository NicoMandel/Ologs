#!/usr/bin/env python3

import os.path
import numpy as np
import h5py

def readh5(fname):
    """
        reading the h5 file, and converting the h5py format to numpy arrays
    """
    datadict = {}
    with h5py.File(fname, 'r') as f:
        for k, v in f.items():
            datadict[k] = v.value
    return datadict


if __name__=="__main__":
   
    # Look in the results directory
    parentDir = os.path.dirname(__file__)
    outputdir = os.path.abspath(os.path.join(parentDir, 'results'))
    files = [f for f in os.listdir(outputdir) if os.path.isfile(os.path.join(outputdir, f))]

    # Setting up the default values:
    best_dyn = 1000
    best_diff = 0
    best_ind = None
    best_diff_ind = None
    for f in files:
        fname = os.path.join(outputdir, f)
        datadict = readh5(fname)
        dyn = datadict["Dynamic Hierarchical"]
        pred = datadict["Predicted"]
        if dyn < best_dyn:
            best_pred = pred
            best_dyn = dyn
            best_ind = int(f.split('.')[0])
            ptu = datadict["Ptu"]
            pu = datadict["Pu"]
        if (pred - dyn) > best_diff:
            dyn_diff = dyn
            pred_diff = pred
            best_diff = (pred - dyn)
            best_diff_ind = int(f.split('.')[0])
            ptu_diff = datadict["Ptu"]
            pu_diff = datadict["Pu"]

    print("============ Results ===========")
    print("Best Dynamic Hierarchical Reproduction: {:.3f} in file {}".format(best_dyn, best_ind))
    print("Pred:{:.3f}".format(best_pred))
    print("Pu:\n{}".format(pu))
    print("Ptu:\n{}".format(ptu))
    print("=======================")
    print("Highest difference: {:.3f} in file: {}".format(best_diff, best_diff_ind))
    print("Dynamic: {:.3f}, Predicted: {:.3f}".format(dyn_diff, pred_diff))
    print("Pu:\n{}".format(pu_diff))
    print("Ptu:\n{}".format(ptu_diff))