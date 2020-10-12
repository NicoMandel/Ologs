#!/usr/bin/env python3

import h5py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

"""
    Order of the Dimensions of the Big Array:
        1. dims = np.asarray([48, 64])
        2. simcases = np.asarray([1, 2])
        3. fov = np.asarray([1, 2, 3])
        4. overh
        5. overv = np.asarray([0.25, 0.5, 0.75, 1.0/3.0, 2.0/3.0])
        6. acc = np.asarray([0.6, 0.7, 0.8, 0.9, 0.95])
        7. probs = np.asarray([0, 1, 2])

        # Boolean options
        8. trans = np.asarray([False, True])
        9. rand = np.asarray([False, True])
        -> Currently Unused: testconf = np.asarray([False, True])

        10. Algorithms = Flat, Pred, Hier, Dyn
        11. Metrics = Entropy, Count - IN THIS ORDER!

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
def reproduceresults(configdict, datadict, entropy=True, visualise=True):
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
    
    if entropy:
        e_dict = {}
        e_dict["Flat"] = flat_e.flatten()
        e_dict["Predicted"] = pred_e.flatten()
        e_dict["Hierarchical Dynamic"] = hier_dyn_e.flatten()
        e_dict["Hierarchical"] = hier_e.flatten()
        for k,v in e_dict.items():
            print("Entropy for {}: {}".format(k,v.sum()))
        # plot_entropy(e_dict, configdict, casedict)
    else:
        wcells = {}
        wcells["Flat"] = wrongcells(gtmap, flatmap)
        wcells["Predicted"] = wrongcells(gtmap, predmap)
        wcells["Hierarchical"] = wrongcells(gtmap, postmap_hier)
        wcells["Hierarchical Dynamic"] = wrongcells(gtmap, postmap_dyn)
        for k, v in wcells.items():
            print("{}: {}".format(k,v))
        print("Maximum number of observations per cell: {}".format(maxobscells(countsmap)))
    if visualise:
        reprdict = {}
        reprdict["Ground Truth"] = vis_idx_map(gtmap, carr)
        reprdict["Flat"] = lookupColorFromPosterior(carr, flatmap)
        reprdict["Hierarchical"] = lookupColorFromPosterior(carr, postmap_hier)
        reprdict["Hierarchical Dynamic"] = lookupColorFromPosterior(carr, postmap_dyn)
        reprdict["Predicted"] = lookupColorFromPosterior(carr, predmap)
        plot_reproduction(reprdict)
    
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
def plot_reproduction(reprdict):
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
        axes[row,col].set(title=k)
    plt.show()

# TODO: Function to plot a single specific case
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
    
    reproduceresults(confdict, datadict)
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
    arr = getcasesarr()

    dirs = [d for d in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, d))]
    for d in dirs:
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
                inddict["Ptu"], inddict["Transp"], inddict["Rand"]])
    return indices 

# Setup functions for the cases
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
    
    # TODO: Add fields here for the a) different metrics b) different simulation cases
    # TODO: include the testconf.size here as well! 
    arr = np.zeros((dims.size, simcases.size, fov.size, overh.size, overv.size, acc.size, probs.size, trans.size, rand.size, algos, metrics))
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
    return cdict

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

def getcasefromindex(ind, cases, algos, metrics):
    """
        Function to get the case from an index of the array 
    """
    cdict = {}
    cdict["Dim"] = cases["Dim"][ind[0]]
    cdict["Sim"] = cases["Sim"][ind[1]]
    cdict["Fov"] = cases["Fov"][ind[2]]
    cdict["HOver"] = cases["HOver"][ind[3]]
    cdict["VOver"] = cases["VOver"][ind[4]]
    cdict["Acc"] = cases["Acc"][ind[5]]
    cdict["Transp"] = cases["Transp"][ind[6]]
    cdict["Rand"] = cases["Rand"][ind[7]]
    cdict["Ptu"] = cases["Ptu"][ind[8]]
    # Algorithms and metrics are kept separate?
    cdict["Algo"] = algos[ind[9]]
    cdict["Metric"] = metrics[ind[10]]
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
    # TODO: Currently unsused: _Test-{}
    outname = "Ptu-{}_Sim-{}_Dim-{}_Fov-{}_Acc-{}_HOver-{}_VOver-{}_Transp-{}_Rand-{}".format(
         int(idxdict["Ptu"]), int(idxdict["Sim"]), int(idxdict["Dim"]), int(idxdict["Fov"]),
          idxdict["Acc"], idxdict["HOver"], idxdict["VOver"], int(idxdict["Transp"]), int(idxdict["Rand"])
    )
    # TODO: Currently Unused: idxdict["Test"]
    return outname


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
    arr = readColResults(outputdir, fname="CollResults.hdf5")
    idx = findsmallest(arr, algos)
    
    # TODO: consider if cases where more than one return index comes out
    print(arr[idx])
    #TODO: Continue here. The problem is, that the idxs do not come back in an acceptable format
    idxdict = getcasefromindex(idx, casesdict, algos, metrics)
    
    casename = createnamefromidxdict(idxdict)
    # Plot a single case
    plotcase(casename, outputdir)

    for k, v in idxdict.items():
        print("{}, {}".format(k,v))
        




    # old function for walking through the directory    
    dirs = [d for d in os.listdir(outputdir) if os.path.isdir(os.path.join(outputdir, d))]
    for d in dirs:
        processing(outputdir, d)

