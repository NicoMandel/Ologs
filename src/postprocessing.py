#!/usr/bin/env python3

import h5py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Evaluation
def cross_entropy(vec_true, vec_pred):
    """
        cross entropy loss for a single element. Following the definition of:
        https://youtu.be/ErfnhcEV1O8?t=579
    """
    return np.sum(vec_true*np.log(vec_pred)) * (-1.0)

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
    xlsxfile = os.path.abspath(os.path.join(outputdir, dirname, "configs.xlsx"))
    configdict = readxlsx(fname=xlsxfile)

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

def readxlsx(fname):
    """
        Reading the xlsx file
    """
    configdict = {}
    with pd.ExcelFile(fname) as f:
        configdict=pd.read_excel(f, sheet_name=None, index_col=0)
    return configdict

# Reproducing the Results
def reproduceresults(configdict, datadict, casedict, entropy=False, visualise=False):
    """
        Function to reproduce the results for a single file 
    """
    # Extract all of the important things from the dictionaries
    gtmap = datadict["Ground Truth"]
    predmap = datadict["Predicted"]
    countsmap = datadict["Counts"]
    hiermap_dyn = datadict["Hierarchical-Dynamic"]
    hiermap = datadict["Hierachical-Pre"]
    flatmap = datadict["Flat"]

    carr = configdict["Colours"].to_numpy()
    obs_prob = configdict["Observation"].to_numpy()
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
        plot_entropy(e_dict, configdict, casedict)
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

if __name__=="__main__":
    
    # Look in the results directory
    parentDir = os.path.dirname(__file__)
    outputdir = os.path.abspath(os.path.join(parentDir, '..', 'tmp', 'results'))
    dirs = [d for d in os.listdir(outputdir) if os.path.isdir(os.path.join(outputdir, d))]
    for d in dirs:
        processing(outputdir, d)

