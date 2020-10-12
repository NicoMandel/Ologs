#!/usr/bin/env python3

"""
    File to test the hyperparameters on the DEFAULT simulation - see function parse_args for default settings.
    Reduced version of one-hot-sim.py
    Accepts the two following arrays (should be sampled from a Dirichlet):
        * Distribution over classes p(u)
        * Distribution over objects for classes p(t|u)
    Returns the total cross-entropy for:
        * Flat estimation
        * Predicted estimation
        * Hierarchical Prediction
        * Dynamic Hierarchical prediction
"""

import numpy as np
import pandas as pd
# Imports for file workings
import sys
from argparse import ArgumentParser
import os.path
import errno
import h5py

def save_results(args, datadict, configs):
    """
        Function to save the results.
            Requires: Output directory, arguments for case discriminiation
            dictionary, with the names being the keys and the values the data 
            We should save:
            All results
            The filename represents the arguments of the simulation, with the tuple being splitable by:
            ParamName-Value_ParamName-Value_ etc.
    """
    parentDir = os.path.dirname(__file__)
    outputdir = os.path.abspath(os.path.join(parentDir, 'tmp', 'results'))

    # Setting up the outputdirectory
    if args.transposed:
        transp = 1
    else:
        transp=0
    if args.random:
        rando = 1
    else:
        rando=0
    
    # With this string formatting it can be split by _ and by -
    outname = "Ptu-{}_Sim-{}_Dim-{}_Fov-{}_Acc-{}_HOver-{}_VOver-{}_Transp-{}_Rand-{}_Test-{}".format(
         args.ptu, args.simcase, args.dim, args.fov, args.accuracy, 1-args.overlaph, 1-args.overlapv, transp, rando, args.testconfig
    )
    outdir = os.path.abspath(os.path.join(outputdir, outname))
    try:
        os.mkdir(outdir)
        # pass
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    # Now write the files in that output directory
    # Use Syntax like:
    # for k,v in dictionary.items() - for all the different arrays that I have
    fname = os.path.join(outdir, outname+".hdf5")
    with h5py.File(fname, "w") as f:
        for k,v in datadict.items():
            dset=f.create_dataset(k, data=v)

    # Write the configs to an excel file. use PD.ExcelWriter here
    fname =  os.path.abspath(os.path.join(outdir,"configs"+".hdf5"))
    with h5py.File(fname, "w") as f:
        for k,v in configs.items():
            dset = f.create_dataset(k, data=v.astype(np.float64))

# Creating the map
def scenario1(xmax, ymax, classlist, transpose=False):
    """
        Helper function to create the ground truth map, with the classlist indicies
        Indices:
            0 - house
            1 - pavement
            2 - grass
            3 - tree
            4 - vehicle
    """

    tree =      np.asarray([0, 0, 0, 1, 0])
    vehicle =   np.asarray([0, 0, 0, 0, 1])
    house =     np.asarray([1, 0, 0, 0, 0])
    pavement =  np.asarray([0, 1, 0, 0, 0])

    n = xmax
    m = ymax
    k = classlist.size
    gt = np.zeros((n,m,k))

    # Divide columns into 4 sections
    fourth = m//4

    # First: Everything is grass
    gt[...,2] = 1
    # second fourth is "pavement"
    gt[:,1*fourth+1:2*fourth,:] = pavement

    # in Fourth fourth, divide rows into 8 block
    eigth = n//8
    for i in range(eigth):
        if i%2==0:
            # Put houses into the even blocks
            r_idx = i*eigth
            gt[r_idx:r_idx+eigth,3*fourth:3*fourth+3,:] = house
    
    # In third block, put a few trees there
    x = np.asarray(range(0,n,5))
    gt[x,2*fourth+3,:] = tree

    # In second Block, put two vehicles there
    quat = m//4
    gt[quat:quat+3,fourth+int(0.5*fourth)-2:fourth+int(0.5*fourth),:] = vehicle
    gt[2*quat:2*quat+3,fourth+int(0.5*fourth)+1:fourth+int(0.5*fourth)+3,:] = vehicle

    if transpose:
        return np.swapaxes(gt,0,1)
    else:
        return gt

def scenario2(xmax, ymax, classlist, random=False, roadwidth=2, carcount=4, h_size=3, h_counts=5, testconfig=False, proportion=0.5, transpose=False):
    """

    """
    gt = np.zeros((xmax, ymax, classlist.size))
    tree =      np.asarray([0, 0, 0, 1, 0])
    vehicle =   np.asarray([0, 0, 0, 0, 1])
    house =     np.asarray([1, 0, 0, 0, 0])
    pavement =  np.asarray([0, 1, 0, 0, 0])
    grass =     np.asarray([0, 0, 1, 0, 0])
    
    quatx = xmax // 4
    quaty = ymax // 4
    gt[...,:] = grass

    # Choose random indices from the second half. Roughly 50%
    if random:
        areasize = int(2*proportion*quatx*ymax)
        xidcs= np.random.randint(2*quatx,xmax,size=areasize)
        yidcs = np.random.randint(ymax,size=areasize)
        idcs = np.asarray((xidcs, yidcs)).T
    else:    # Make the idcs nonrandom
        xrang = np.arange(2*quatx,xmax,2)
        yrang = np.arange(0,ymax,3)
        idcs = np.array([np.array([x,y]) for x in xrang for y in yrang])
        # yidcs = [y for y in yrang for x in xrang]
    gt[idcs[:,0], idcs[:,1], :] = tree
    
    # 2 Roads
    halfx = xmax //2
    halfy = ymax //2
    gt[:,halfx-roadwidth:halfx+roadwidth,:] = pavement
    gt[halfy-roadwidth:halfy+roadwidth,:,:] = pavement
    
    # sets of cars
    if random:
        cars = np.random.randint(1,carcount+1)
        caridx = np.random.randint(0,ymax-3,size=cars)
        caridcs = [np.arange(idx, idx+3) for idx in caridx]
        gt[halfx-roadwidth:halfx,caridcs,:] = vehicle

        caridx = np.random.randint(0,ymax-3,size=carcount-cars)
        caridcs = [np.arange(idx, idx+3) for idx in caridx]
        gt[caridcs,halfx:halfx+roadwidth,:] = vehicle
    else:
        caridx = np.arange(0,xmax, 12)
        caridcs = [np.arange(idx, idx+3) for idx in caridx]
        gt[halfx-roadwidth:halfx,caridcs,:] = vehicle
        gt[caridcs,halfx:halfx+roadwidth,:] = vehicle

    # Houses:
    # Top left indices
    tlx = halfx - roadwidth - h_size
    ty = halfy - roadwidth - h_size
    trx = halfx + roadwidth + 1
    
    if random:
        tlh_idx = np.random.randint(0, tlx, size=h_counts)
        trh_idx = np.random.randint(trx, xmax, size=h_counts)
        tly_idx = np.random.randint(0, ty, size=h_counts)
        try_idx = np.random.randint(0, ty, size=h_counts)
        y_idcs = np.concatenate((tly_idx, try_idx), axis=None)
        x_idcs = np.concatenate((tlh_idx, trh_idx), axis=None)
        h_yidcs = np.array([np.arange(y,y+h_size) for y in y_idcs])
        h_xidcs = np.array([np.arange(x, x+h_size) for x in x_idcs])
        pts = []
        h_idx = np.array([np.array([x,y]) for i, x_arr in enumerate(h_xidcs) for x in x_arr for y in h_yidcs[i]])

    else:
        tlh_idx = np.arange(0, tlx, (h_size+2)*2)
        trh_idx = np.arange(trx, xmax-h_size, (h_size+2)*2)
        tly_idx = np.arange(0, ty-h_size, (h_size+2)*2)
        x_idcs = np.concatenate((tlh_idx, trh_idx), axis=None)
        tlx_idcs = [np.arange(idx, idx+h_size) for idx in x_idcs]
        tly_idcs = [np.arange(idx, idx+h_size) for idx in tly_idx]
        h_idx = np.array([np.array([x, y]) for x_coord in tlx_idcs for x in x_coord for y_coord in tly_idcs for y in y_coord])
    
    # Switching around the cases
    if testconfig:
        gt[h_idx[:,0], h_idx[:,1], :] = house
    else:
        gt[h_idx[:,1],h_idx[:,0],:] = house
    
    if transpose:
        return np.swapaxes(gt,0,1)
    else:
        return gt
    return gt
    
def get_map_counts(map1):
    """
        Function to return the (relative) counts of each class available in the map.
        For evaluation with priors
        Requires a 3D map, where the 3rd dimension is a vector of 0s and ones and it counts the 1s
    """
    n_cells = map1.shape[0] * map1.shape[1]
    out = np.count_nonzero(map1, axis=(0,1)) / n_cells
    return out

def makeObs(gt, obs_probab, classlist):
    """
        Returns an observation based on the Ground truth and the Observation probability
    """

    obs = np.empty((gt.shape[0], gt.shape[1]), dtype='object')
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            cl_name = gt[i,j]
            prob = obs_probab.loc[cl_name,:]
            obs[i,j] = np.random.choice(classlist, p=prob.to_numpy())
    return obs

def gensampleidx(gt, observation_probability):
    """
        generates an index with probability proportional to the row in obs_prob
    """
    sam = np.arange(gt.shape[2])
    samples = np.zeros((gt.shape[0], gt.shape[1]))
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):    
            idx = np.nonzero(gt[i,j])[0]
            p = observation_probability[idx] 
            samples[i,j] = np.random.choice(sam, p=p[0])
    return samples.astype(np.int)

# Prediction Functions
def pred_flat(fut_states, alpha=0.5):
    """
        Function to do a flat prediction. See File "bayes-discr-1D.py" in folder ../tmp for details
    """

    # find out how many cells still have a uniform prior
    num_states = fut_states.shape[2]
    unif_list = []
    alr_obs_list = []
    unif_vec = np.ones(num_states, dtype=float)/num_states
    for i in range(fut_states.shape[0]):
        for j in range(fut_states.shape[1]):
            fut_state = fut_states[i,j]
            # if the vector is uniform: 
            if np.array_equal(unif_vec, fut_state):
                unif_list.append(tuple([i,j]))
            else:
                alr_obs_list.append(tuple([i,j]))
    # unif_ct is now the amount of cells that still have a uniform prior
    map_size = fut_states.shape[0] * fut_states.shape[1]
    unif_ct = len(unif_list)
    rel_unif = unif_ct/map_size
    # if the relative amount of uniform cells is small, the weight of the prior is small
    new_pr = np.copy(fut_states)
    unif_list = np.asarray(unif_list)
    alr_obs_list = np.asarray(alr_obs_list)
    n_vec = np.zeros(num_states)
    for o_val in alr_obs_list:
        new_val = fut_states[o_val[0], o_val[1]]
        n_vec += (1.0/len(alr_obs_list)) * new_val.astype(float)
    # old_states = fut_states[alr_obs_list]
    # new_states = fut_states[unif_list]
    # new_pr[alr_obs_list] = fut_states[alr_obs_list]
    for upd_wp in unif_list:
        # Find way to update this
        new_pr[upd_wp[0], upd_wp[1]] = (1.0-alpha) * new_pr[upd_wp[0], upd_wp[1]] + alpha*n_vec
    
    return new_pr

def assign_prior(map1, areadist_vec, area_class_mat):
    """
        function to assign a more informed prior - sum over the assumed distribution of areas multiplied by the observation probability of that class
        p(t|u) = p(u|t) * p(t) / p(t) || with p(t) = sum(p(u|t) p(t))
    """
    vec = areadist_vec.T @ area_class_mat
    map1[...,:] = vec
    return map1

# Updating Functions:
def updateMap(x_min, x_max, y_min, y_max, posterior, lmap):
    """
        Function that takes the new probabilities with the observations and the original maps and update the function on the right indices
    """
    lmap[x_min:x_max, y_min:y_max] = posterior

def updateprobab(obs, obs_probab, prior):
    """
        Prior: Prior probabilities over the map elements:
        obs: Observation made
        obs_probab: probability of making the observations
        Returns: posterior over the map
    """
    # Prior is a 3D array
    post = np.empty_like(prior)
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            pr = prior[i,j]
            vec = obs_probab[obs[i,j]]
            po = vec*pr
            po = po/po.sum()
            post[i,j] = po

    return post

# Path Planning Functions
def retrieveVisibleFields(wp, fov=1):
    """
        Retrieves the indices of visible fields from the given [x, y] index of the UAV.
        Use the fov + in each direction. Assumes index 0,0 to be the corner between 0,0 and 1,1! 
    """

    x_min = wp[0]-fov+1
    x_max = wp[0]+fov+1
    y_min = wp[1]-fov+1
    y_max = wp[1]+fov+1
    # x_vals = np.arange(wp[0]-fov+1, wp[0]+fov+1)
    # y_vals = np.arange(wp[1]-fov+1, wp[1]+fov+1)
    return x_min, x_max, y_min, y_max
   
def getflightpattern(xmax, ymax, fov=1, overlap=(0.5, 0.5)):
    overlap_x = overlap[0]
    overlap_y = overlap[1]
    stride_x = int(2*fov*overlap_x)
    stride_y = int(2*fov*overlap_y)
    iteration=1
    x = fov-1
    y = fov-1
    # print("X max: {}, Y max: {}, Stride_x: {}, Stride_y: {}".format(xmax, ymax, stride_x, stride_y))
    wps = []
    while(x+fov < xmax):
        while(y+fov < ymax):
            if iteration%2==1:
                # leave the order
                y_n = y
            else:
                # invert the order
                y_n = ymax-y-fov
            wp = tuple([x, y_n])
            wps.append(wp)
            y += stride_y
        y = fov-1
        x += stride_x
        iteration+=1
    return np.asarray(wps)

# Hierarchical functions
def getareaprior(arealist):
    """
        Function to get a uniform prior over the areas
    """
    return np.ones_like(arealist).astype(np.float) / arealist.size

def calchierprob(pu, ptu_df):
    """
        Function to calculate the total probability of observing something
    """
    x = pu @ ptu_df
    return x/x.sum()

def updatearea(ptu, pu, obs):
    p_tu = ptu.T[obs]
    pos = pu * p_tu
    pos = pos / pos.sum()
    return pos

def updatecounts(counts, samples):
    """
        Function to update the count array
    """
    for i in range(samples.shape[0]):
        for j in range(samples.shape[1]):
            counts[i,j,samples[i,j]] += 1
    # counts[...,samples] +=1
    return counts

# Hierarchical Dynamic Prediction for the cells:
def dynamic_prediction(cts, ptu, prior_u, classlist):
    """
        A function to dynamically estimate which area we are in
        Returns a new prior, which should be used for all cells where the counts are zero
        Arguments: 
            The counts of observations
            The observation functions:
                p(u|t) = p(t|u) p(u)
                with p(u) being the flat prior over the areas
    """

    # Step 1: find cells that have already been observed
    obs = np.sum(cts, axis=(0,1)).astype(np.int)
    # Step 2: for each of the observations, run "updatearea()"  With the appropriate values
    post_u = prior_u
    for i in range(obs.size):
        n_obs = obs[i]
        for j in range(n_obs):
            post_u = updatearea(ptu, post_u, i)
    # Step 3: recalculate p(t|u) with the new p(u)
    class_probab = calchierprob(post_u, ptu)

    return class_probab

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

# Evaluation
def cross_entropy(vec_true, vec_pred):
    """
        cross entropy loss for a single element. Following the definition of:
        https://youtu.be/ErfnhcEV1O8?t=579
    """
    return np.sum(vec_true*np.log(vec_pred)) * (-1.0)

def wrongcells(gtmap, predicted):
    """
        Function to return the Relative percentage of wrongly predicted cells
    """
    pred_idxmax = np.asarray(np.unravel_index(np.argmax(predicted, axis=2), predicted.shape))[2]
    gt_idxmax = np.asarray(np.unravel_index(np.argmax(gtmap, axis=2), gtmap.shape))[2]
    diff = pred_idxmax - gt_idxmax
    return np.count_nonzero(diff) / (gtmap.shape[0] * gtmap.shape[1])    

# Actually running the simulation
def runsimulation(args, pu, ptu, obs_prob, arealist, classlist):

    # ================================
    #### Section 1 - Setup work
    # ================================

    max_map_size = args.dim
    n1 = m1 = max_map_size
    fov = args.fov
    h_overlap = 1-args.overlaph
    v_overlap = 1-args.overlapv
    overlap = (h_overlap, v_overlap)
    likelihood = args.accuracy
    simcase = args.simcase
    transposed = args.transposed
    rand = args.random
    testconfig = args.testconfig
    
    # Ground Truth map
    if simcase == 1:
        gtmap = scenario1(max_map_size, max_map_size, classlist, transpose=transposed)
    elif simcase == 2:
        gtmap = scenario2(max_map_size, max_map_size, classlist, transpose=transposed, random=rand, testconfig=testconfig)

    real_distribution = get_map_counts(gtmap)
    pred_classes_hierar = pu @ ptu
    
    # ================================
    # SECTION 2: creating the reproduction maps
    # ================================

    # A map to store the counts
    countsmap = np.zeros_like(gtmap)

    # Maps that are used for predictions:
    predmap = np.ones_like(gtmap) / gtmap.shape[2]
    flatmap = np.copy(predmap)

    # two additional maps used for prediction 
    hiermap = np.copy(predmap)          # One that uses the flat prior prediction from our model
    hiermap[:,:] = np.asarray(pred_classes_hierar)
    hiermap_dyn = np.copy(hiermap)      # One that updates the p(u) dynamically
       
    # Observation probabilites and waypoints
    wps = getflightpattern(n1, m1, fov=fov, overlap=overlap)      # Flight pattern

    # ================================
    # Section 3: Running the simulation
    # ================================
    for i in range(wps.shape[0]-1):
        
        # indices that are currently visible
        x_min, x_max, y_min, y_max = retrieveVisibleFields(wps[i], fov=fov)
        gt = gtmap[x_min:x_max, y_min:y_max]    #  Ground Truth area
        obs = gensampleidx(gt, obs_prob)        # make Observations

        # Getting the priors for the maps        
        pr_flat = flatmap[x_min:x_max, y_min:y_max,:]
        pr_pred = predmap[x_min:x_max, y_min:y_max,:]
        # For the hierarchical function
        counts = countsmap[x_min:x_max, y_min:y_max,:]

        # Updating the counts
        counts = updatecounts(counts, obs)
        countsmap[x_min:x_max, y_min:y_max, :] = counts

        # Update the probabilities
        post_flat = updateprobab(obs, obs_prob, pr_flat)
        post_pred = updateprobab(obs, obs_prob, pr_pred)

        # Re-incorporate the information into the map
        flatmap[x_min:x_max, y_min:y_max] = post_flat
        predmap[x_min:x_max, y_min:y_max] = post_pred

        # Predict the next step
        xmin_pred, xmax_pred, ymin_pred, ymax_pred = retrieveVisibleFields(wps[i+1], fov=fov)
        fustates = predmap[xmin_pred:xmax_pred, ymin_pred:ymax_pred]
        nst_pred = pred_flat(fustates)
        # Hierarchical prediction:
        dyn_pr = hiermap_dyn[xmin_pred:xmax_pred, ymin_pred:ymax_pred, :]
        cts_fut = countsmap[xmin_pred:xmax_pred, ymin_pred:ymax_pred, :]
        pred_dyn = dynamic_prediction(cts_fut, ptu, pu, classlist) 

        # Re-incorporate prediction-values into the map 
        predmap[xmin_pred:xmax_pred, ymin_pred:ymax_pred] = nst_pred
        # hierarchical incorporation - where nothing has been observed yet
        zer_idcs = np.where(np.sum(cts_fut, axis=2) == 0)
        dyn_pr[zer_idcs] = pred_dyn
        hiermap_dyn[xmin_pred:xmax_pred, ymin_pred:ymax_pred, :] = dyn_pr

    # ================================
    ## SECTION 4: Save values
    # ================================
    datadict = {}
    datadict["Counts"] = countsmap
    datadict["Ground Truth"] = gtmap
    datadict["Hierarchical-Dynamic"] = hiermap_dyn
    datadict["Hierachical-Pre"] = hiermap
    datadict["Predicted"] = predmap
    datadict["Flat"] = flatmap

    configs = {}
    configs["Hierarch"] = ptu
    configs["Hier_Prior"] = pu
    configs["Observation"] = obs_prob
    configs["Real_Dist"] = real_distribution
    configs["Pred_Hier"] = pred_classes_hierar

    save_results(args, datadict, configs)
