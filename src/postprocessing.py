#!/usr/bin/env python3

import h5py
import numpy as np
import pandas as pd
import os
import time
from matplotlib import rcParams
rcParams['font.family'] = "Serif"
rcParams['font.serif'] = ["Times New Roman"]
rcParams['axes.titleweight'] = "bold"
rcParams['axes.titlesize'] = 20
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import deepcopy
import seaborn as sns
# plt.rcParams["font.family"] = "Serif"
# plt.rcParams["font.name"] = "Times New Roman"
# csfont = {'fontname' : 'Times New Roman'}

"""
    Order of the Dimensions of the Big Array  - IN THIS ORDER!
        0. dims = np.asarray([48, 64])
        1. simcases = np.asarray([1, 2])
        2. fov = np.asarray([1, 2, 3])
        3. overh
        4. overv = np.asarray([0.25, 0.5, 0.75, 1.0/3.0, 2.0/3.0])
        5. acc = np.asarray([0.6, 0.7, 0.8, 0.9, 0.95])
        6. probs = np.asarray([0, 1, 2])

        # Boolean options
        7. trans = np.asarray([False, True])
        8. rand = np.asarray([False, True])
        9. testconf = np.asarray([False, True])

        10. Algorithms = Flat, Pred, Hier, Dyn
        11. Metrics = Entropy, Count

"""


# Evaluation
def cross_entropy(vec_true, vec_pred):
    """
        cross entropy loss for a single element. Following the definition of:
        https://youtu.be/ErfnhcEV1O8?t=579
    """
    return np.sum(vec_true*np.log(vec_pred)) * (-1.0)

def cross_entropy_arr(arr_true, arr_pred):
    """
        Cross Entropy for two arrays of the same size, Summed up to asingle value
    """
    cr = 0
    for i in range(arr_true.shape[0]):
        for j in range(arr_true.shape[1]):
            gt = arr_true[i,j,:]
            pred = arr_pred[i,j,:]
            cr += cross_entropy(gt, pred)

    return cr

def recreate_posterior(prior, counts, obs_prob):
    """
        Function to recreate the posterior, with the prior and the number of observations of the classes
        as well as the observation probabilities
    """
    post = prior
    for i in range(counts.size):
        ct = counts[i].astype(np.int)
        for j in range(ct):
            vec = obs_prob[i]
            post = vec*post
            post = post / post.sum()
    return post

def wrongcells(gtmap, predicted):
    """
        Function to return the Relative percentage of wrongly predicted cells
    """
    pred_idxmax = np.asarray(np.unravel_index(np.argmax(predicted, axis=2), predicted.shape))[2]
    gt_idxmax = np.asarray(np.unravel_index(np.argmax(gtmap, axis=2), gtmap.shape))[2]
    diff = pred_idxmax - gt_idxmax
    return np.count_nonzero(diff) / (gtmap.shape[0] * gtmap.shape[1])     

def maxobscells(countsmap):
    """
        returns the maximum number of observations per cell
    """
    return np.max(countsmap.sum(axis=2))

# Reading Files and configurations
def processing(outputdir, dirname):
    """
        Function to call the processing on a single file
    """

    # SplitFilename
    casedict = splitfilename(dirname)

    # Read the h5py File
    fname = os.path.abspath(os.path.join(outputdir, dirname, dirname+".hdf5"))
    datadict = readh5(fname=fname)

    # Read the Excelfile
    xlsxfile = os.path.abspath(os.path.join(outputdir, dirname, "configs.hdf5"))
    configdict = readh5(fname=xlsxfile)

    # Reproduce the results for a single file
    reproduceresults(configdict, datadict, casedict)

def splitfilename(fname, firstsepa='_', secondsepa='-'):
    """
        Function to split the filename into all the keys and values for the simulation
    """
    c_dict = {}
    for spl in fname.split(firstsepa):
        k, v = spl.split(secondsepa)
        c_dict[k] = float(v)
    return c_dict

def readh5(fname):
    """
        reading the h5 file, and converting the h5py format to numpy arrays
    """
    datadict = {}
    with h5py.File(fname, 'r') as f:
        for k, v in f.items():
            datadict[k] = v.value
    return datadict

# Reproducing the Results
def reproduceresults(configdict, datadict, casename, entropy=True, visualise=True):
    """
        Function to Plot the results for a single file 
    """
    # Extract all of the important things from the dictionaries
    gtmap = datadict["Ground Truth"]
    predmap = datadict["Predicted"]
    countsmap = datadict["Counts"]
    hiermap_dyn = datadict["Hierarchical-Dynamic"]
    hiermap = datadict["Hierachical-Pre"]
    flatmap = datadict["Flat"]

    # carr = configdict["Colours"].to_numpy()
    carr = colorarr()
    obs_prob = configdict["Observation"]
    real_dist = configdict["Real_Dist"]
    hier_prior = configdict["Hier_Prior"]
    pred_classes_hierarchical = configdict["Pred_Hier"]
    
    # Run the calculation of the cross_entropy for all of them
    # Create arrays that store all the information
    postmap_dyn = np.zeros_like(gtmap)
    postmap_hier = np.copy(postmap_dyn)
    pred_e = np.zeros((gtmap.shape[0], gtmap.shape[1]))
    flat_e = np.copy(pred_e)
    hier_dyn_e = np.copy(pred_e)
    hier_e =  np.copy(pred_e)

    max_obs = countsmap.max()
    
    # Recalculate the posterior
    for i in range(gtmap.shape[0]):
        for j in range(gtmap.shape[1]):
            
            gt = gtmap[i,j,:]
            # Recreate the posteriors:
            postmap_dyn[i,j,:] = recreate_posterior(hiermap_dyn[i,j,:], countsmap[i,j,:], obs_prob)
            postmap_hier[i,j,:] = recreate_posterior(hiermap[i,j,:], countsmap[i,j,:], obs_prob)

            # Calculate the entropys
            pred_e[i,j] = cross_entropy(gt, predmap[i,j,:])
            flat_e[i,j] = cross_entropy(gt, flatmap[i,j,:])
            hier_dyn_e[i,j] = cross_entropy(gt, postmap_dyn[i,j,:])
            hier_e[i,j] = cross_entropy(gt, postmap_hier[i,j,:])

    valsdict={}   
    if entropy:
        valsdict["Flat"] = flat_e.sum()
        valsdict["Predicted"] = pred_e.sum()
        valsdict["Hierarchical Dynamic"] = hier_dyn_e.sum()
        valsdict["Hierarchical"] = hier_e.sum()
    else:
        valsdict["Flat"] = wrongcells(gtmap, flatmap)
        valsdict["Predicted"] = wrongcells(gtmap, predmap)
        valsdict["Hierarchical"] = wrongcells(gtmap, postmap_hier)
        valsdict["Hierarchical Dynamic"] = wrongcells(gtmap, postmap_dyn)
    if visualise:
        reprdict = {}
        reprdict["Ground Truth"] = vis_idx_map(gtmap, carr)
        reprdict["Flat"] = lookupColorFromPosterior(carr, flatmap)
        reprdict["Hierarchical"] = lookupColorFromPosterior(carr, postmap_hier)
        reprdict["Hierarchical Dynamic"] = lookupColorFromPosterior(carr, postmap_dyn)
        reprdict["Predicted"] = lookupColorFromPosterior(carr, predmap)
        plot_reproduction(reprdict, valsdict, max_obs, casename)
    else:
        for k, v in valsdict.items():
            print("Values for: {} = {}".format(k,v))
        print("Maximum number of observations per cell: {}".format(max_obs))
    
# Plotting Functions
def plot_entropy(e_dict, configdict, settings):
    """
        Function to plot the entropy dictionary
    """
    fig = plt.figure()
    ax = fig.gca()
    x = np.arange(e_dict["Flat"].size)
    for k, v in e_dict.items():
        ax.plot(x, v, alpha=0.5, label=k)
    for k,v in configdict.items():
        pass
        # TODO: Processing on the configurations dictionary
    title = ""
    for k,v in settings.items():
        title = title +"{}:{}; ".format(k,v)
    ax.set(title=title)
    ax.legend()
    plt.show() 

# Color lookup
def colorarr():
    """
        Function to return the same color vectors as the df, but only through numpy indices.
        Indices:
            0 - house
            1 - pavement
            2 - grass
            3 - tree
            4 - vehicle
    """
    house = [0.921, 0.117, 0.137, 1.0]
    pavement = [0.662, 0.674, 0.647, 1.0]
    grass= [0.384, 0.976, 0.411, 1.0]
    tree = [0.164, 0.576, 0.219, 1.0]
    vehicle = [0.172, 0.533, 0.866, 1.0]
    return np.asarray([house, pavement, grass, tree, vehicle])

def lookupColorFromPosterior(carr, post):
    """
        Carr is an array with the appropriate color codes, Post a 3-dim. array with the posterior probabilities over the classes
        for each 2-dim cell
    """     
    col = np.empty((post.shape[0], post.shape[1], 4))
    idxmax = np.asarray(np.unravel_index(np.argmax(post, axis=2), post.shape))[2]
    for i in range(idxmax.shape[0]):
        for j in range(idxmax.shape[1]):
            col[i,j] = carr[idxmax[i,j]]
    # alternative: col = cdf[classlist[np.argmax(post, axis=2)]]
    return col

def vis_idx_map(gtmap, carr):
    """
        Function to turn a one-hot vector map into something displayable by imshow.
        Indices:
            0 - house
            1 - pavement
            2 - grass
            3 - tree
            4 - vehicle
    """
    outmap = np.zeros((gtmap.shape[0], gtmap.shape[1], 4))
    for i in range(outmap.shape[0]):
        for j in range(outmap.shape[1]):
            outmap[i,j,:] = carr[np.nonzero(gtmap[i,j])]

    return outmap

# Plotting the Reproduction
def plot_reproduction(reprdict, valsdict, max_obs, casename=None):
    """
        Function to plot the reproductions
    """
    noPlots = int(np.ceil(len(reprdict) / 2))
    fig, axes = plt.subplots(noPlots, 2)
    for i, (k, v) in enumerate(reprdict.items()):
        row = int(np.floor(i/2))
        if i % 2 ==0:
            col = 0
        else:
            col = 1
        axes[row,col].imshow(v)
        title = k
        try:
            title += " {:.1f}".format(valsdict[k])
        except KeyError:
            pass
        axes[row,col].set(title=title)
    plt.suptitle(casename+" Max Obs: {}".format(max_obs))
    plt.show()

# Function to plot a single specific case
def plotcase(casename, resultsdir):
    """
        Function that plots a specific case in the results directory
    """
    try:
        d = os.path.abspath(os.path.join(resultsdir, casename))
        f = os.path.join(d, casename+".hdf5")
        conff = os.path.join(d, "configs.hdf5")
        datadict = readh5(f)
        confdict = readh5(conff)
    except OSError:
        raise OSError("File or Folder does not exist. Check filename again: {}".format(casename))
    
    reproduceresults(confdict, datadict, casename)
    print("Testline")


# Functions for manipulating the output data
# function to collect ALL data in the output directory 
def collectalldata(outdir):
    """
        Function to collect ALL cases in the output directory. Cases and the directory need to be specified
    """
    # Prelims for all the cases that we have
    algos = np.asarray(["Flat", "Predicted", "Hierarchical", "Hierarchical Dynamic"])
    metrics = np.asarray(["Entropy", "Count"])

    cdict = getcasesdict()
    arr = getcasesarr(algos.size, metrics.size)

    dirs = [d for d in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, d))]
    alldirs = len(dirs)
    t1 = time.time()
    for i, d in enumerate(dirs):
        casedict = splitfilename(d)
        fname = os.path.abspath(os.path.join(outputdir, d, d+".hdf5"))
        datadict = readh5(fname=fname)

        cfgf = os.path.abspath(os.path.join(outputdir, d, "configs.hdf5"))
        configdict = readh5(fname=cfgf)

        # Find the right index now, using the cdict and casedict for the filename
        ind = findindex(casedict, cdict)
        edict, wdict = getMetrics(datadict, configdict)
        for k, v in edict.items():
            wc = wdict[k]
            algo_ind = np.where(algos == k)[0]
            wind = tuple(np.concatenate((ind, algo_ind, [1]), axis=None))
            eind = tuple(np.concatenate((ind, algo_ind, [0]), axis=None))
            arr[wind] = wc
            arr[eind] = v

        if i % (alldirs/10) == 0:
            t2 = time.time() - t1
            tperit = t2 / (i+1)
            print("Finished with {} % \of the {} cases. Took {} s, ETA: {}".format( (i/alldirs) * 10, alldirs, tperit, (alldirs-i)*tperit))

    return arr

def getMetrics(datadict, configdict):
    """
        Function to write the information into the array at the desired position.
        Array is the big, multidimensional array containing all the information.
        Ind is the index of the current simulation. Still requires 2 additional arguments for:
            * Algorithm: Flat, predicted, Hierarchical, Dynamic
            * Metric: Count of wrong cells, entropy
    """
     # Extract all of the important things from the dictionaries
    gtmap = datadict["Ground Truth"]
    predmap = datadict["Predicted"]
    countsmap = datadict["Counts"]
    hiermap_dyn = datadict["Hierarchical-Dynamic"]
    hiermap = datadict["Hierachical-Pre"]
    flatmap = datadict["Flat"]

    carr = colorarr()
    obs_prob = configdict["Observation"]
    real_dist = configdict["Real_Dist"]
    hier_prior = configdict["Hier_Prior"]
    pred_classes_hierarchical = configdict["Pred_Hier"]
    
    # Run the calculation of the cross_entropy for all of them
    # Create arrays that store all the information
    postmap_dyn = np.zeros_like(gtmap)
    postmap_hier = np.copy(postmap_dyn)
    
    # Recalculate the posterior
    for i in range(gtmap.shape[0]):
        for j in range(gtmap.shape[1]):
            gt = gtmap[i,j,:]
            # Recreate the posteriors:
            postmap_dyn[i,j,:] = recreate_posterior(hiermap_dyn[i,j,:], countsmap[i,j,:], obs_prob)
            postmap_hier[i,j,:] = recreate_posterior(hiermap[i,j,:], countsmap[i,j,:], obs_prob)

    edict = {}
    edict["Flat"] = cross_entropy_arr(gtmap, flatmap)
    edict["Predicted"] = cross_entropy_arr(gtmap, predmap)
    edict["Hierarchical"] = cross_entropy_arr(gtmap, postmap_hier)
    edict["Hierarchical Dynamic"] = cross_entropy_arr(gtmap, postmap_dyn)
   
    wcells = {}
    wcells["Flat"] = wrongcells(gtmap, flatmap)
    wcells["Predicted"] = wrongcells(gtmap, predmap)
    wcells["Hierarchical"] = wrongcells(gtmap, postmap_hier)
    wcells["Hierarchical Dynamic"] = wrongcells(gtmap, postmap_dyn)
    return edict, wcells
    
def findindex(casedict, cdict):
    """
        Function to find the index where the corresponding dictionary entries match
    """
    inddict = {}
    for k, v in casedict.items():
        c = cdict[k]
        ind = np.where(c == v)[0]
        inddict[k] = ind
    
    indices = np.asarray([inddict["Dim"], inddict["Sim"], inddict["Fov"], inddict["HOver"], inddict["VOver"], inddict["Acc"],
                inddict["Ptu"], inddict["Transp"], inddict["Rand"], inddict["Test"]])
    return indices 

# Setup functions for the array
def getcasesarr(algos=4, metrics=2):
    """
        Function to get all the simulation cases that we are running through / have been running through
    """
    dims = np.asarray([48, 64])
    simcases = np.asarray([1, 2])
    fov = np.asarray([1, 2, 3])
    overh = overv = np.asarray([0.25, 0.5, 0.75, 1.0/3.0, 2.0/3.0])
    acc = np.asarray([0.6, 0.7, 0.8, 0.9, 0.95])
    probs = np.asarray([0, 1, 2])

    # The boolean options
    trans = np.asarray([False, True])
    rand = np.asarray([False, True])
    testconf = np.asarray([False, True])
    
    arr = np.zeros((dims.size, simcases.size, fov.size, overh.size, overv.size, acc.size, probs.size, trans.size, rand.size, testconf.size, algos, metrics))
    return arr

def getcasesdict():
    """
        Function to get the dictionary with all the cases
    """
    cdict = {}
    cdict["Dim"] = np.asarray([48, 64])
    cdict["Sim"] = np.asarray([1, 2])
    cdict["Fov"] = np.asarray([1, 2, 3])
    cdict["HOver"] = np.asarray([0.25, 0.5, 0.75, 1.0/3.0, 2.0/3.0])
    cdict["VOver"] = np.asarray([0.25, 0.5, 0.75, 1.0/3.0, 2.0/3.0])
    cdict["Acc"] = np.asarray([0.6, 0.7, 0.8, 0.9, 0.95])
    cdict["Transp"] = np.asarray([0, 1])
    cdict["Rand"] = np.asarray([0, 1])
    cdict["Ptu"] = np.asarray([0, 1, 2])
    cdict["Test"] = np.asarray([0, 1])
    return cdict

# unnecessary - since each case has ALL algorithms and ALL metrics
# def augmentcasesdict(cdict):
#     """
#         Function to augment the case dictionary with the Algorithms and the metrics
#     """
#     cdict["Algo"] = np.asarray([0, 1, 2, 3])
#     cdict["Metric"] = np.asarray([0, 1])
#     return cdict

# For Manipulation of collected results
def savebigarr(arr, outputdir):
    """
        Function to save the big array after calculating entropy and so forth
    """

    fname = os.path.join(outputdir, "CollResults.hdf5")
    with h5py.File(fname, "w") as f:
        dset=f.create_dataset("CollectedResults", data=arr)

def readColResults(path, fname):
    """
        Function to load the collected results into a numpy array and return them
    """
    p = os.path.join(path, fname)
    with h5py.File(p, 'r') as f:
        arr = f["CollectedResults"].value

    return arr

def getcasefromindex(ind, cases):
    """
        Function to get the case from an index of the array
        Does not care about algorithm or metric, since all of them are contained in a case
    """
    cdict = {}
    cdict["Dim"] = cases["Dim"][ind[0]]
    cdict["Sim"] = cases["Sim"][ind[1]]
    cdict["Fov"] = cases["Fov"][ind[2]]
    cdict["HOver"] = cases["HOver"][ind[3]]
    cdict["VOver"] = cases["VOver"][ind[4]]
    cdict["Acc"] = cases["Acc"][ind[5]]
    cdict["Ptu"] = cases["Ptu"][ind[6]]
    # Boolean Options
    cdict["Transp"] = cases["Transp"][ind[7]]
    cdict["Rand"] = cases["Rand"][ind[8]]
    cdict["Test"] = cases["Test"][ind[9]]

    return cdict

# Functions to evaluate the collected results - TODO: Continue here with formatting the indices correctly
def findsmallest(arr, algos, algorithm="Hierarchical Dynamic", metric="entropy"):
    """
        Function to find the smallest value of a given algorithm.
    """

    algind = np.where(algos == algorithm)[0][0]
    if metric=="entropy":
        mind = 0
    elif metric == "wrong":
        mind = 1
    else:
        raise ValueError("Wrong index specified, metric does not exist")
    subarr = arr[...,algind, mind]
    # ind = np.unravel_index(np.argmin(subarr, axis=None), subarr.shape)
    ind = np.where( subarr==np.min(subarr[np.nonzero(subarr)]))

    # Append the algorithm and metric index
    ind = tuple(np.concatenate((ind, algind, mind), axis=None))
    return ind

def findtotalsmallest(arr, metric="entropy"):
    """
        Function to find the index of the total smallest, either entropy or wrong cells.
        TODO: For wrong cells this returns multiple entries
    """
    if metric=="entropy":
        m = 0
    elif metric=="wrong":
        m = 1
    else:
        raise ValueError("Wrong index specified, metric does not exist")
    subarr = arr[...,m]
    ind = np.where(subarr==np.min(subarr[np.nonzero(subarr)]))
    ind = tuple(np.concatenate((ind, m), axis=None))
    return ind

def findsmallestsimcase(arr, sim_case=2, algo=None, metric="entropy"):
    """
        Function to find the smallest value for a specific simulation case
    """
    if metric=="entropy":
        m = 0
    elif metric=="wrong":
        m=1
    else:
        raise ValueError("Wrong index specified, metric does not exist")

    sc = sim_case-1
    subarr = arr[:,sc,...]
    ind = np.where(subarr==np.min(subarr[np.nonzero(subarr)]))
    # TODO: use this if clause for other cases
    if len(ind[0]) > 1:
        sc_ind = np.full(len(ind[0]), sc)
        m_ind = np.full(len(ind[0]), m)
        ind = np.asarray(ind).T
        nind = np.insert(ind, 1, sc_ind, axis=1)
        ind = np.zeros((nind.shape[0], nind.shape[1]+1))
        ind[:,:-1] = nind
        ind[:,-1] = m_ind
    # TODO: push the sc and the metric in there
    return ind.astype(np.int)

def createnamefromidxdict(idxdict):
    """
        Function to turn an index dictionary into an appropriate filename for the h5py module to load
    """
    outname = "Ptu-{}_Sim-{}_Dim-{}_Fov-{}_Acc-{}_HOver-{}_VOver-{}_Transp-{}_Rand-{}_Test-{}".format(
         int(idxdict["Ptu"]), int(idxdict["Sim"]), int(idxdict["Dim"]), int(idxdict["Fov"]),
          idxdict["Acc"], idxdict["HOver"], idxdict["VOver"], int(idxdict["Transp"]), int(idxdict["Rand"]),
          int(idxdict["Test"])
    )
    return outname

def getAxisDict():
    """
        Function to get an axis lookup for use with np.take()
        TODO: Check what position "Test" gets read into!
    """
    a = {}
    a["Dim"] = 0
    a["Sim"] = 1
    a["Fov"] = 2
    a["HOver"] = 3
    a["VOver"] = 4
    a["Acc"] = 5
    a["Ptu"] = 6
    a["Transp"] = 7
    a["Rand"] = 8
    a["Test"] = 9
    a["Algo"] = 10
    a["Metric"] = 11
    return a

# https://numpy.org/doc/stable/reference/generated/numpy.take.html
def usenumpytake(arr, axisdict, target_axis, target_case):
    """
        Function to use numpy take to index the value in the huge array.
        Parameters:
            * The array which to look at
            * The lookup for the axes, which axes to return for target_axis
            * The lookup for which case from that axis to take: the casesdict
            * The target case 
                CAREFUL: Target case has to be specified as the right integer to look for
    """
    ax = axisdict[target_axis]
    subarr = np.take(arr, target_case, axis=ax)

    # Good ways to find the indices:
    ind = np.where(subarr == np.min(subarr[np.nonzero(subarr)]))
    # Alternative Option:
    # TODO: Turn the array into a np.nan array arr.fill(np.nan) OR: replace(0 with nan) and use:
    # np.unravel_index(np.nanargmin(subarr), subarr.shape)

    # TODO: Append the indices to the back here
    if len(ind[0]) < 2:
        nind = np.insert(ind, ax, target_case)
    else:
        tgt = np.full(len(ind[0]), target_case)
        nind = np.asarray(ind).T
        nind = np.insert(nind, ax, tgt, axis=1)
    return nind

def comparearray(arr):
    """
        Comparison of a subarray for one metric.
    """
    a_f = arr[...,0]
    a_p = arr[...,1]
    a_h = arr[...,2]
    a_d = arr[...,3]

    # Test: replace all 0 values in an array with nan
    # a_f[a_f == 0 ] = np.nan
    # a_p

    diff_fp = comparetwoarrays(a_f, a_p)
    diff_ah = comparetwoarrays(a_f, a_h)
    diff_pd = comparetwoarrays(a_p, a_d)
    pos_fp, neg_fp = getposnegcounts(diff_fp)
    pos_hf, neg_hf = getposnegcounts(diff_ah)
    pos_pd, neg_pd = getposnegcounts(diff_pd)
    # print("Done with the counts where condition is fulfilled.")

    # Find a sample index where the condition is true
    ind = np.argwhere(diff_pd > 0)

    return ind

def getcounts(arr, axes=(0,1)):
    """
        Function similar to comparearray, but instead of returning an index, this returns:
            Params:
            * Axes to compare. Values:
            * 0 - Flat
                * 1 - Predicted
                * 2 - Hierarchical
                * 3 - Dynamic Hierarchical
    """
    a1 = arr[...,axes[0]]
    a2 = arr[...,axes[1]]

    diff_arr = comparetwoarrays(a1, a2)
    pos, neg = getposnegcounts(diff_arr)
    return pos, neg

def getrelative(arr, axes=(0,1)):
    """  
        Function again similar to comparearray, but instead of returning an index, this returns:
        A relative percentage.
        Calls getcounts()
        axes:   0 - flat
                1- predicted
                2 - hierarchical
                3 - dynamic hierarchical
    """
    pos, neg = getcounts(arr, axes=axes)
    # return the percentage of cases where pos is better
    rel = pos / (pos+neg)
    return rel

def getposnegcounts(diff_arr):
    """
        Function to get the counts of positive and negative elements of an array of differences
    """
    pos = np.count_nonzero(diff_arr > 0)
    neg = np.count_nonzero(diff_arr < 0)
    return pos, neg

def comparetwoarrays(a1, a2):
    """
        Function to subtract a2 from a1. a1 should be the bigger one (Higher value) then
    """
    return a1 - a2

def getbigger(diff_arr):
    """
        Returns a list of indices where the input array is bigger than 0
    """

def getsmaller(diff_arr):
    """
        Returns a list of indices where the input array is smaller than 0
    """
    res = np.where(diff_arr > 0)
    listofcoord = list(zip([r for r in res]))
    return listofcoord

# def manipulatesubarray(ent, axisdict):
#     """
#         Manipulate the subarray with the appropriate subindexing
#     """
#     ax0 = axisdict["Ptu"]
#     ax0_ind = 0
#     ptu0 = np.take(ent, ax0_ind, axis=ax0)

#     ax1 = axisdict["Sim"]
#     ax1_ind = 1
#     ptu1 = np.take(ent, ax1_ind, axis=ax0)
#     if ax0 < ax1:
#         ax1 -= 1
#     ptu1sim1 = np.take(ptu1, ax1_ind, axis=ax1)

#     indices_ptu = comparearray(ptu0)
#     indices_ptusim1 = comparearray(ptu1sim1)

#     # Insert the index of what we took out back in
#     tgt = np.full(indices_ptu.shape[0], ax0_ind)
#     nind = np.insert(indices_ptu, ax0, tgt, axis=1)
    
#     # If we took out multiple things, walk back up.
#     tgt1 = np.full(indices_ptusim1.shape[0], ax1_ind)
#     nnind = np.insert(indices_ptusim1, ax1, tgt1, axis=1)
#     nnnind = np.insert(nnind, ax0, tgt1, axis=1)

def getrelcountsforcase(arr, axisdict, keytup, valtup, axes=(0,1)):
    """
        For a specific case defined by keytup, valtup, get the relative counts.
        Similar structure to manarr
        Calling getrel() instead of indices.
        axes=() defines which two axes should be subtracted from one another
            0 is flat
            1 is pred
            2 is hier
            3 is dyn
    """
    l = []
    ndict = {}
    for i,k in enumerate(keytup):
        v = axisdict[k]
        l.append(v)
        ndict[v] = valtup[i]
    la = np.asarray(l)
    las = np.sort(la, kind='stable')
    lasf = np.flip(las)
    
    # Sorting through the array and selecting the subarray we are looking for
    a = arr
    for axind in lasf:
        validx = ndict[axind]
        a = np.take(a, validx, axis=axind)
    
    perc = getrelative(a, axes=axes)
    return perc


def comparearray_select(arr, select_index=True):
    """
        Comparison of a subarray for one metric. Has a Boolean to choose whether good or bad indices should be selected
    """
    a_f = arr[...,0]
    a_p = arr[...,1]
    a_h = arr[...,2]
    a_d = arr[...,3]

    # Test: replace all 0 values in an array with nan
    # a_f[a_f == 0 ] = np.nan
    # a_p

    diff_fp = comparetwoarrays(a_f, a_p)
    diff_ah = comparetwoarrays(a_f, a_h)
    diff_pd = comparetwoarrays(a_p, a_d)
    pos_fp, neg_fp = getposnegcounts(diff_fp)
    pos_hf, neg_hf = getposnegcounts(diff_ah)
    pos_pd, neg_pd = getposnegcounts(diff_pd)
    # print("Done with the counts where condition is fulfilled.")

    # Find a sample index where the condition is true
    if select_index:
        ind = np.argwhere(diff_pd > 0)
    else:
        ind = np.argwhere(diff_pd < 0)
    return ind

def manarr(e, axisdict, keytup, valtup, select_index=True):
    """
        Manipulate an array and do the operations on the key and value tuples
    """
    # l = np.asarray([axisdict[k] for k in keytup])
    l = []
    ndict = {}
    for i, k in enumerate(keytup):
        v = axisdict[k]
        l.append(v)
        ndict[v] = valtup[i]
    la = np.asarray(l)
    las = np.sort(la, kind='stable')
    lasf = np.flip(las)
    
    # Sort through the array and select the subarray we are looking for
    a = e
    for axind in lasf:
        validx = ndict[axind]
        a = np.take(a, validx, axis=axind)
    
    # Do the actual operation on
    indices = comparearray_select(a, select_index)
    
    # Recreate the indices - walking back up the list
    nind = indices
    for axind in las:
        val = ndict[axind]
        tgt = np.full(indices.shape[0], val)
        nind = np.insert(nind, axind, tgt, axis=1)

    return nind

def testindices(keytup, valtup, arr, axisdict, casesdict, outputdir):
    """
        Function to test the indices
    """
    indicestrialrundonotusethisforanything = manarr(arr, axisdict, keytup, valtup)

    for _ in range(min(5, indicestrialrundonotusethisforanything.shape[0])):
        r = np.random.randint(indicestrialrundonotusethisforanything.shape[0])
        somecasesdictionary = getcasefromindex(indicestrialrundonotusethisforanything[r], casesdict)
        casenameforrandomthingy = createnamefromidxdict(somecasesdictionary)
        plotcase(casenameforrandomthingy, outputdir)

def collectedEval(arr, axisdict, casesdict):
    """
        Function to run a collection of evaluations on array.
        Only for Simulation case 2
    """
    axes = (1,3)    # to compare predicted with dynamic
    # Simulation case 1 is kept static
    cdict = deepcopy(casesdict)
    del cdict["Sim"]
    outerdict = {}
    for k, vals in cdict.items():
        innerdict = {}
        for v in vals:       
            keytup = tuple(["Sim", k])
            iddx = np.where(vals == v)[0][0]
            valtup = tuple([1, iddx])
            try:
                perc = getrelcountsforcase(arr, axisdict, keytup, valtup, axes=axes)
                innerdict[v] = perc
            except ZeroDivisionError:
                pass
        outerdict[k] = innerdict
    return outerdict
    # print("Testline")

def barplotdict(coldict):
    """
        Function to plot the collected dictionary.
    """
    reform = {(outerKey, innerKey): values for outerKey, innerDict in coldict.items() for innerKey, values in innerDict.items()}
    # df = pd.DataFrame.from_dict(coldict)
    mi = pd.MultiIndex.from_tuples(reform.keys(), names=('Parameter', 'Value'))
    df = pd.DataFrame(reform.values(), index=mi, columns=["Ratio"])
    df2 = df.reset_index()
    print(df2.head())
    testbarplot(df, coldict)
    print("Testline")

def testbarplot2(df, coldict):
    # Trial 5:
    # sns.set_theme(style="whitegrid")
    # df["Value"] = df["Value"].astype(dtype=)
    g = sns.catplot(
        data=df, kind="bar", x="Parameter", y="Ratio", hue="Value"
    )
    g.despine(left=True)
    g.set_axis_labels("Relative Counts", "Parameters")
    # g.legend.set_title("")
    plt.show()

def testbarplot(df, coldict):
    """
        Function to test the plotting of a pandas df
    """
    # Trial 1:
    # sns.catplot(x=df["Value"], col=df["Parameter"], data=df["Ratio"])
    
    # Trial2:
    # data = df.set_index(["Parameter", "Value"])
    # data.unstack().plot(kind= "bar", rot=90)
    
    # Trial3:
    # fig,ax = plt.subplots()
    # y_pos = np.arange(df.shape[0])
    # ax.barh(y_pos, df["Ratio"], align='center')
    alph = 0.85
    c1 = (0.5, 0, 0.5, alph)
    c2 = (0.662, 0.674, 0.647, alph)
    c3 = (0.384, 0.976, 0.411, alph)
    c4 = (235/255, 103/255, 52/255, alph)
    c5 = (0.172, 0.533, 0.866, alph)
    c = [c5, c4, c3, c2, c1]
    # Trial4:
    # TODO: Adapt the names that are 0, 1 - find better way!
    def adaptdictnames():
        return tuple([False, True])
    
    h_ratios = []
    for i in coldict.values():
        h_ratios.append(len(i.keys()))
    gs_kws =dict(height_ratios=h_ratios)

    # cols = ["blue", "black", "green", "red", "cyan"]
    # cm = plt.get_cmap('winter')
    fig, axs = plt.subplots(len(coldict.keys()), sharex=True, figsize=(14,7), gridspec_kw=gs_kws)
    for i, k in enumerate(coldict.keys()):
        subdf = df.loc[k]
        y_pos = np.arange(subdf.shape[0])
        axs[i].barh(y_pos, subdf["Ratio"], align='center', color=c)
        axs[i].set_yticks(y_pos)

        axs[i].set_yticklabels(coldict[k], fontsize=16, fontweight="bold")
        axs[i].invert_yaxis()       # labels read top to bottom - says matplotlib
        if i < (len(coldict.keys()) - 1):
            axs[i].get_xaxis().set_visible(False)
            axs[i].spines["bottom"].set_visible(False)
        else:
            axs[i].tick_params(axis="x", labelsize=20)
            plt.xticks(weight="bold")

            # axs[i].xaxis.label.set_fontsize(24)
            # axs[i].tick_params(axis='x', fontsize=14)
            # pass
        axs[i].set_ylabel(k, rotation=45, va="center", ha="right", fontsize=20, fontweight="bold")

        r = axs[i].spines["right"].set_visible(False)
        t = axs[i].spines["top"].set_visible(False)
        axs[i].axvline(0.5, ls='--', color='k', linewidth=4)
    # plt.axvline(0.5, ls='--')
    # plt.tight_layout()
    plt.show()

def testbarplot3(df, coldict):
    """
        Function to test the plotting of a pandas df
    """

    alph = 0.85
    c1 = (0.1, 0.1, 0.1, alph)
    c2 = (0.662, 0.674, 0.647, alph)
    c3 = (0.384, 0.976, 0.411, alph)
    c4 = (235/255, 103/255, 52/255, alph)
    c5 = (0.172, 0.533, 0.866, alph)
    c = [c5, c4, c3, c2, c1]


    h_ratios = []
    for i in coldict.values():
        h_ratios.append(len(i.keys()))
    
    fig = plt.figure(constrained_layout=True, figsize=(15,7))
    spec = fig.add_gridspec(ncols=1, nrows=len(h_ratios), height_ratios=h_ratios)

    for i, k in enumerate(coldict.keys()):
        ax = fig.add_subplot(spec[i])
        subdf = df.loc[k]
        y_pos = np.arange(subdf.shape[0])
        ax.barh(y_pos, subdf["Ratio"], align='center', color=c)
        ax.set_yticks(y_pos)

        ax.set_yticklabels(coldict[k], fontsize=16, fontweight="bold")
        ax.invert_yaxis()       # labels read top to bottom - says matplotlib
        if i < (len(coldict.keys()) - 1):
            ax.get_xaxis().set_visible(False)
            ax.spines["bottom"].set_visible(False)
        else:
            ax.tick_params(axis="x", labelsize=20)
            # ax.xaxis.label.set_fontsize(24)
            # ax.tick_params(axis='x', fontsize=14)
            # pass
        ax.set_ylabel(k, rotation=45, va="center", ha="right", fontsize=20, fontweight="bold")

        r = ax.spines["right"].set_visible(False)
        t = ax.spines["top"].set_visible(False)
        ax.axvline(0.5, ls='--', color='k', linewidth=4)
    # plt.axvline(0.5, ls='--')
    # plt.tight_layout()
    plt.show()


    
def plotmulticases(outputdir):
    """
        Function to plot Variations over 1 single case
    """        

    somecasedict = {}
    somecasedict["Dim"] = 64
    somecasedict["Sim"] = 2
    somecasedict["Fov"] = 2
    somecasedict["HOver"] = 0.5
    somecasedict["VOver"] = 0.5
    somecasedict["Acc"] = 0.8
    somecasedict["Ptu"] = 0
    # These ones need to be varied
    somecasedict["Transp"] = 0
    somecasedict["Test"] = 0
    somecasedict["Rand"] = 0
    carr = colorarr()
    
    # first case 
    casename1 = createnamefromidxdict(somecasedict)
    gt1 = getgtdata(casename1, outputdir, carr)

    # Second case
    somecasedict["Transp"] = 1
    casename2 = createnamefromidxdict(somecasedict)
    gt2 = getgtdata(casename2, outputdir, carr)

    # Third case
    somecasedict["Rand"] = 1
    somecasedict["Transp"] = 0
    casename3 = createnamefromidxdict(somecasedict)
    gt3 = getgtdata(casename3, outputdir, carr)

    # Fourth case
    somecasedict["Rand"] = 0
    somecasedict["Test"] = 1
    casename4 = createnamefromidxdict(somecasedict)
    gt4 = getgtdata(casename4, outputdir, carr)

    # Simulation case 5
    somecasedict["Test"] = 0
    somecasedict["Sim"] = 1
    casename5 = createnamefromidxdict(somecasedict)
    gt5 = getgtdata(casename5, outputdir, carr)

    # Loading the path planning thing
    tdir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'tmp')
    tf = os.path.join(tdir, "49.hdf5")
    with h5py.File(tf, 'r') as f:
        data = f["path"].value
        wps = f["wps"].value

    print(data.shape)
    i = 47
    fig, axs = plt.subplots(2,3)
    axs[0,0].imshow(gt5)
    axs[0,0].get_xaxis().set_visible(False)
    axs[0,0].get_yaxis().set_visible(False)
    axs[0,0].set(title="a) Simulation Case 1")
    axs[1,0].imshow(data)
    axs[1,0].set(title="d) Example Path")
    axs[1,0].get_xaxis().set_visible(False)
    axs[1,0].get_yaxis().set_visible(False)
    axs[1,0].scatter(wps[i,1], wps[i,0], s=20, c='black', marker='x')
    axs[1,0].scatter(wps[:i,1], wps[:i,0], s=10, c='blue', marker='.', alpha=0.5)
    axs[1,0].scatter(wps[i+1:,1], wps[i+1:,0], s=10, c='black', marker='.', alpha=0.5)
    axs[0,1].imshow(gt1)
    axs[0,1].get_xaxis().set_visible(False)
    axs[0,1].get_yaxis().set_visible(False)
    axs[0,1].set(title="b) Simulation Case 2")
    axs[0,2].imshow(gt2)
    axs[0,2].get_xaxis().set_visible(False)
    axs[0,2].get_yaxis().set_visible(False)
    axs[0,2].set(title="c) Transposed")
    axs[1,1].imshow(gt3)
    axs[1,1].get_xaxis().set_visible(False)
    axs[1,1].get_yaxis().set_visible(False)
    axs[1,1].set(title="e) Randomised")
    axs[1,2].imshow(gt4)
    axs[1,2].get_xaxis().set_visible(False)
    axs[1,2].get_yaxis().set_visible(False)
    axs[1,2].set(title="f) Test Configuration")
    plt.show()

    print("Testline for Debugging")
    
def getgtdata(cname, resultsdir, carr):
    try:
        d = os.path.abspath(os.path.join(resultsdir, cname))
        f = os.path.join(d, cname+".hdf5")
        datadict = readh5(f)
    except OSError:
        raise OSError("File or Folder does not exist. Check filename again: {}".format(cname))
    # what to do with the datadict here?
    gtmap = datadict["Ground Truth"]
    gt = vis_idx_map(gtmap, carr)
    return gt
    

def plottwocasestoptobottom_outer(arr, axisdict, casesdict, outputdir):
    """
        Function to get two cases and plot them above each other
    """

    # Define two keytuples and valuetuples to select specific cases - one with 0.9 and one with 0.7 detector accuracy?
    keytup=tuple(["Sim", "Dim", "Rand", "Acc", "Ptu"])
    valtup1 = tuple([1, 1, 0, 2, 0])
    valtup2 = tuple([1, 1, 1, 1, 2])

    # Use the manarr function to get the indices of the cases
    ind1 = manarr(arr, axisdict, keytup, valtup1, select_index=False)
    ind2 = manarr(arr, axisdict, keytup, valtup2)

    # Get the case dictionary
    case1 = getcasefromindex(ind1[0], casesdict)
    case2 = getcasefromindex(ind2[0], casesdict)

    # Get two index dictionaries
    c1name = createnamefromidxdict(case1)
    c2name = createnamefromidxdict(case2)

    plottwocasestoptobottom_inner(outputdir, c1name, c2name)



def plottwocasestoptobottom_inner(outputdir, c1name, c2name):
    """
        Inner function for reproducing the results and loading the files
    """
    try:
        d1, conf1 = getdatadict(outputdir, c1name)
    except OSError:
        raise OSError("File or Folder does not exist. Check filename again: {}".format(c1name))

    try:
        d2, conf2 = getdatadict(outputdir, c2name)
    except OSError:
        raise OSError("File or Folder does not exist. Check filename again: {}".format(c2name))
    
    # TODO: Do the reproduction of the results
    carr = colorarr()
    # Extract all of the important things from the dictionaries
    gt1 = d1["Ground Truth"]
    pr1 = d1["Predicted"]
    cts1 = d1["Counts"]
    hd1 = d1["Hierarchical-Dynamic"]
    flat1 = d1["Flat"]
    obs_prob1 = conf1["Observation"]
    hier_prior1 = conf1["Hier_Prior"]
    pred_classes_hierarchical1 = conf1["Pred_Hier"]

    # for the second case
    gt2 = d2["Ground Truth"]
    pr2 = d2["Predicted"]
    cts2 = d2["Counts"]
    hd2 = d2["Hierarchical-Dynamic"]
    flat2 = d2["Flat"]
    obs_prob2 = conf2["Observation"]
    hier_prior2 = conf2["Hier_Prior"]
    pred_classes_hierarchical2 = conf2["Pred_Hier"]
    
    # Create arrays that store the reproduction
    postmap_dyn1 = np.zeros_like(gt1)
    postmap_dyn2 = np.zeros_like(gt2)

    # Create the values that store the entropies

    max_obs1 = cts1.max()
    max_obs2 = cts2.max()
    # Recalculate the posterior
    for i in range(gt1.shape[0]):
        for j in range(gt1.shape[1]):
            
            gtv1 = gt1[i,j,:]
            gtv2 = gt2[i,j,:]
            # Recreate the posteriors:
            post1 = recreate_posterior(hd1[i,j,:], cts1[i,j,:], obs_prob1)
            post2 = recreate_posterior(hd2[i,j,:], cts2[i,j,:], obs_prob2)
            postmap_dyn1[i,j,:] = post1
            postmap_dyn2[i,j,:] = post2
    
    # Entropies
    e_pr1 = cross_entropy_arr(gt1, pr1)
    e_fl1 = cross_entropy_arr(gt1, flat1)
    e_hd1 = cross_entropy_arr(gt1, postmap_dyn1)
    e1 = {}
    e1["Ground Truth"] = ""
    e1["Naive Bayes"] = e_fl1
    e1["Algorithm 2"] = e_pr1
    e1["Algorithm 4"] = e_hd1
    
    e_pr2 = cross_entropy_arr(gt2, pr2)
    e_fl2 = cross_entropy_arr(gt2, flat2)
    e_hd2 = cross_entropy_arr(gt2, postmap_dyn2)
    e2 = {}
    e2["Ground Truth"] = ""
    e2["Naive Bayes"] = e_fl2
    e2["Algorithm 2"] = e_pr2
    e2["Algorithm 4"] = e_hd2

    # Visual Reproductions
    reprdict1 = {}
    reprdict1["Ground Truth"] = vis_idx_map(gt1, carr)
    reprdict1["Naive Bayes"] = lookupColorFromPosterior(carr, flat1)
    reprdict1["Algorithm 2"] = lookupColorFromPosterior(carr, pr1)
    reprdict1["Algorithm 4"] = lookupColorFromPosterior(carr, postmap_dyn1)

    reprdict2 = {}
    reprdict2["Ground Truth"] = vis_idx_map(gt2, carr)
    reprdict2["Naive Bayes"] = lookupColorFromPosterior(carr, flat2)
    reprdict2["Algorithm 2"] = lookupColorFromPosterior(carr, pr2)
    reprdict2["Algorithm 4"] = lookupColorFromPosterior(carr, postmap_dyn2)
    
    # TODO: Plot the reproduction top and bottom.
    fig, axes = plt.subplots(2, len(reprdict2), figsize=(14,8))
    ds = tuple([reprdict1, reprdict2])
    es = tuple([e1, e2])
    cs = tuple([c1name, c2name])
    maxs = tuple([max_obs1, max_obs2])

    Rom = tuple(["I", "II"])
    alph = tuple(["a) ", "b) ", "c) ", "d) "])

    tit = "{}: {:.1f}"
    for i, reprdict in enumerate(ds):
        r =  Rom[i]
        for j, (k, v) in enumerate(reprdict.items()):
            a = alph[j]
            axes[i,j].imshow(v)
            e = es[i][k]
            if "Ground" not in k:
                title = r + " " + a + tit.format(k,e)
            else: 
                title = r + " " + a + k
            axes[i,j].set(title=title)
            axes[i,j].get_xaxis().set_visible(False)
            axes[i,j].get_yaxis().set_visible(False)
        print("Casename: {}. Max observations: {}".format(cs[i],maxs[i]))

    # plt.tight_layout()
    plt.show()

    print("Testline")


def getdatadict(outputdir, cname):
    """
        Function to only get the datadict. should be wrapped with OSError
    """
    d = os.path.abspath(os.path.join(outputdir, cname))
    f = os.path.join(d, cname+".hdf5")
    conff = os.path.join(d, "configs.hdf5")
    datadict = readh5(f)
    confdict = readh5(conff)
    return datadict, confdict


if __name__=="__main__":

    # Prelims for all the cases that we have
    casesdict = getcasesdict()
    algos = np.asarray(["Flat", "Predicted", "Hierarchical", "Hierarchical Dynamic"])
    metrics = np.asarray(["Entropy", "Count"])
    
    # Look in the 'tmp' directory
    parentDir = os.path.dirname(__file__)
    outputdir = os.path.abspath(os.path.join(parentDir, 'tmp', 'results'))
    # arr = collectalldata(outputdir)
    # savebigarr(arr, outputdir)
    # Loading the big array of all the processed results
    # collectalldata(outputdir)
    try:
        arr = readColResults(parentDir, fname="CollResults.hdf5")
    except OSError:
        print("File does not exist yet. Have to create. This may take a while...")
        arr = collectalldata(outputdir)
        savebigarr(arr, outputdir)


    # Do the processing here:
    axisdict = getAxisDict()

    # Split by the two metrics that we have
    entr = arr[...,0]
    wrong = arr[...,1]

    # ====================
    # TODO: WHICH RESULTS DO I ACTUALLY WANT!!!!
    # 1. The number of simulations where:
    #  1.1 Hierarchical is better than Flat,
    #  1.1.1. for Ptu = 0 - which is our predefined case
    #  1.1.2. for Ptu = 1 AND all simcases == 2
    # 2. The number of simulations where:
    # 2.1. predicted is better than flat
    # 2.2 Hierarchical Dynamic is better than predicted
    # # ===================

    # The plotting case
    keytup = ("Ptu", "Sim", "Acc", "Test", "Transp", "Dim")
    valtup = (2, 1, 2, 0, 0, 1)
    plotmulticases(outputdir)
    # TODO: This is the actual target 
    coldict = collectedEval(entr, axisdict, casesdict)
    # barplotdict(coldict)

    plottwocasestoptobottom_outer(entr, axisdict, casesdict, outputdir)

    keytup=tuple(["Sim"])
    valtup = tuple([1])
    testindices(keytup, valtup, entr, axisdict, casesdict, outputdir)
    

    # manipulatesubarray(entr, axisdict)    
    indices_e = comparearray(entr)
    indices_w = comparearray(wrong)

    # Pick a random index for both
    rand_case_e = indices_e[np.random.randint(indices_e.shape[0])]
    rand_case_w = indices_w[np.random.randint(indices_w.shape[0])]
    rce_idxd = getcasefromindex(rand_case_e, casesdict)
    rcw_idxd = getcasefromindex(rand_case_w, casesdict)
    rce_cn = createnamefromidxdict(rce_idxd)
    rcw_cn = createnamefromidxdict(rcw_idxd)
    plotcase(rce_cn, outputdir)
    plotcase(rcw_cn, outputdir)
    # ===================================
    # Old stuff
    # idx1 = usenumpytake(arr, axisdict, target_axis="Metric", target_case=0)
    idx = findsmallest(arr, algos)
    
    # TODO: consider if cases where more than one return index comes out
    print(arr[idx])
    # print(arr[idx1])
    idxdict = getcasefromindex(idx, casesdict)
    
    casename = createnamefromidxdict(idxdict)
    # Plot a single case
    plotcase(casename, outputdir)

    for k, v in idxdict.items():
        print("{}, {}".format(k,v))
        




    # old function for walking through the directory    
    dirs = [d for d in os.listdir(outputdir) if os.path.isdir(os.path.join(outputdir, d))]
    for d in dirs:
        processing(outputdir, d)

