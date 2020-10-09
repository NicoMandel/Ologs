#!/usr/bin/env python3

"""
    Example file to run an Experiment.
    Everything  is lowercase!!!

    TODO: Indexing into the right higher level 
    block will require indexing like CUDA! blockDim.x * blockidx.X + ThreadIdx.x

    # TODO: Idea: what if we omit the boundaries from the reproduction - does the quality improve?

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation
# Imports for file workings
from argparse import ArgumentParser
import sys
import os.path
import h5py
import errno

# Argument parsing and Saving 
def parse_args():
    """
        Function to parse the arguments
    """
    parser = ArgumentParser(description="Path planning module Parameters")
    parser.add_argument("-d", "--dim", default=64, help="Dimensions of the (square) map in fields. Default is 64", type=int)
    parser.add_argument("-f", "--fov", default=1, help="1/2 of the FOV of the UAV. Is used forward AND backward. Default is 1", type=int)
    parser.add_argument("--overlaph", default=0.5, help="Horizontal desired overlap. Default is 0.5", type=float)
    parser.add_argument("--overlapv", default=0.5, help="Vertical desired overlap. Default is 0.5", type=float)
    parser.add_argument("-a", "--accuracy", default=0.8, help="Detection accuracy. Default is 0.8", type=float)
    parser.add_argument("-t", "--transposed", default=False, help="Whether the map should be transposed. Default is false", action="store_true")
    parser.add_argument("-s", "--simcase", default=1, help="Which simulation case to run. Default is 1", type=int)
    parser.add_argument("-r", "--random", default=False, action="store_true", help="Whether object locations should be randomly generated or not")
    args = parser.parse_args()
    return args

def checkarguments(args):
    """
        Function to check the arguments for admissible/inadmissible stuff
    """

    fov = args.fov
    h_overlap = 1-args.overlaph
    v_overlap = 1-args.overlapv
    overlap = (h_overlap, v_overlap)
    for over in overlap:
        if (over % (1.0 / (2*fov)) != 0):
            sys.exit("Error: Overlap {} Not a multiple of: {}".format(
                over, (1.0 / (2*fov))
            ))
    print("All checks passed. Continuing with case:")
    ar = vars(args)
    for k,v in ar.items():
        print("{}: {}".format(
            k,v
        ))

def save_results(outputdir, args, datadict, configs):
    """
        Function to save the results.
            Requires: Output directory, arguments for case discriminiation
            dictionary, with the names being the keys and the values the data 
            We should save:
            All results
            The filename represents the arguments of the simulation, with the tuple being splitable by:
            ParamName-Value_ParamName-Value_ etc.
    """

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
    outname = "Sim-{}_Dim-{}_Fov-{}_Acc-{}_HOver-{}_VOver-{}_Transp-{}_Rand-{}".format(
        args.simcase, args.dim, args.fov, args.accuracy, 1-args.overlaph, 1-args.overlapv, transp, rando
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
    fname =  os.path.abspath(os.path.join(outdir,'configs'+'.xlsx'))
    with pd.ExcelWriter(fname) as writer:
        for shname, df in configs.items():
            shname = (shname[:29]) if len(shname) > 30 else shname
            if df.shape[0] > 2:
                df.to_excel(writer, sheet_name=shname)

def plotresults(datadict, args):
    """
        Alternative to saving the results: Plotting them:
    """
    fig = plt.figure()
    ax = fig.gca()
    x = np.arange(args.dim*args.dim)
    for k, v in datadict.items():
        ax.plot(x, v.flatten(), alpha=0.5, label=k)
    ax.legend()
    ax.set(title="Sim-{}_Dim-{}_Fov-{}_Acc-{}_HOver-{}_VOver-{}".format(
        args.simcase, args.dim, args.fov, args.accuracy, 1-args.overlaph, 1-args.overlapv,
    ))
    ax.set_ylim(0,0.15)
    # for i in range(max_map_size):
    #     plt.axvline(i * max_map_size, ls='--')
    plt.show()

# Map creation and indexing functions
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

def fillmap_idx(gt, classlist, scenario=1, transpose=False):
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

    n = gt.shape[0]
    m = gt.shape[1]
    k = classlist.size
    gt = np.zeros((n,m,k))
    if scenario==1:
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
    # Alternative one-liner, not tested
    # outmap[...-1]=carr[np.nonzero(gtmap)]

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
    
def get_map_counts(map1):
    """
        Function to return the (relative) counts of each class available in the map.
        For evaluation with priors
        Requires a 3D map, where the 3rd dimension is a vector of 0s and ones and it counts the 1s
    """
    n_cells = map1.shape[0] * map1.shape[1]
    out = np.count_nonzero(map1, axis=(0,1)) / n_cells
    return out

# Sampling Functions
def observation_probabilities(classlist, maxim=0.8):
    """
        Returns an array with the observation probabilities for each class.
        The observation probabilities are calculated using maxim as p(z|q) and a uniform distribution over all other values
    """

    num_classes = classlist.size
    conf_probab = (1.0-maxim)/(num_classes-1)
    arr = np.empty([num_classes, num_classes])
    np.fill_diagonal(arr, maxim)
    off_diag = np.where(~np.eye(num_classes,dtype=bool))
    arr[off_diag] = conf_probab
    return arr

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

def gensampleidx(gt, obs_probab):
    """
        generates an index with probability proportional to the row in obs_probab
        TODO: turn this into a numpy-esque programming thing
    """
    sam = np.arange(gt.shape[2])
    samples = np.zeros((gt.shape[0], gt.shape[1]))
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):    
            idx = np.nonzero(gt[i,j])[0]
            p = obs_prob[idx] 
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

def pred_dir(fut_states):
    """
        A more informed version to predict the states of the next cell, using the dirichlet alpha values
        See "bayes-discr-1D.py" in folder ../tmp for details
        inverse indexing following this example: https://stackoverflow.com/questions/25330959/how-to-select-inverse-of-indexes-of-a-numpy-array
    """
    num_states = fut_states.shape[2]
    num_cells = fut_states.shape[0] * fut_states.shape[1]
    test_vec = np.ones(num_states)
    # n_obs_cells = np.logical_and.reduce(fut_states == test_vec, axis = -1).nonzero()
    # x = np.array_equal(fut_states, test_vec)
    n_obs_cells = np.all(np.isin(fut_states, test_vec), axis=-1).nonzero()
    obs_cells = np.invert(np.all(np.isin(fut_states, test_vec), axis=-1)).nonzero()
    # _z = np.isin(fut_states, test_vec)
    # n_obs_cells = np.transpose(np.all(_z,axis=2).nonzero())    
    # obs_cells = np.transpose(np.all(~_z, axis=2).nonzero())
    x = fut_states[obs_cells].sum(axis=0) / len(obs_cells)
    _x = fut_states[n_obs_cells].sum(axis=0) / len(n_obs_cells)
    r_obs = len(obs_cells) / num_cells          # This is the lam mixing parameter 
    pred_states = r_obs * x + (1-r_obs) * _x
    fut_states[n_obs_cells,-1] = pred_states
    return fut_states

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
            # TODO: Adapt this line!
            vec = obs_probab[obs[i,j]]
            po = vec*pr
            po = po/po.sum()
            post[i,j] = po

    return post

def updateDir(obs, prior):
    """
        Function to update the Dirichlet distribution for an array
    """
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            prior[i,j,obs[i,j].astype(int)] += 1
    return prior
    # Alternative:
    # prior[...,obs[i,j]]+=1

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
   
def getpattern(x_max, y_max, fov=1):
    """
        Function to return the pattern the agent will be moving. The agent sees x,y+fov and -fov+1  
        TODO: Invert the order here
    """
    pattern = []
    for i in range(fov-1, y_max-fov, fov):
        if i%2==0:
            for j in range(fov-1, x_max-fov+1, fov):
                next_wp = tuple([i,j])
                pattern.append(next_wp)
        else:
            for j in range(x_max-fov-1, fov-2, fov*-1):
                next_wp = tuple([i,j])
                pattern.append(next_wp)

    # Desired output: List of the form: [0,0], [1,0], [2, 0]
    return np.asarray(pattern)

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

def gethierarchprobab(arealist, objectlist):
    """
        Function to define the probabilities p(t|u) with u being the areas.
        Rows sum to 1
        Indices (columns):
            0 - house
            1 - pavement
            2 - grass
            3 - tree
            4 - vehicle
        Indices (rows):
            0 - urban
            1 - road
            2 - forest
    """
    df = pd.DataFrame(index=arealist, columns=objectlist)
    df.at["urban", "house"] = 0.5
    df.at["urban", "pavement"] = 0.2
    df.at["urban", "grass"] = 0.05
    df.at["urban", "tree"] = 0.1
    df.at["urban", "vehicle"] = 0.15

    df.at["road", "house"] = 0.05
    df.at["road", "pavement"] = 0.5
    df.at["road", "grass"] = 0.10
    df.at["road", "tree"] = 0.05
    df.at["road", "vehicle"] = 0.3

    df.at["forest", "house"] = 0.05
    df.at["forest", "pavement"] = 0.1
    df.at["forest", "grass"] = 0.4
    df.at["forest", "tree"] = 0.4
    df.at["forest", "vehicle"] = 0.05
    return df

def calchierprob(pu, ptu_df):
    """
        Function to calculate the total probability of observing something
    """
    x = pu @ ptu_df
    return x/x.sum()

def updatearea(probab_tu, pr_u, obs):
    p_tu = probab_tu[obs]
    pos = pr_u * p_tu
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

def updateHierProbab(obs, obs_prob, pr_hier, counts, ptu_df):
    """
        Function to update the hierarchical Probabiltities, Given:
            Obs - Observations made as indices
            Obs_prob - Observation probabilities p(z|x)
            pr_hier - hierarchical prior used for recalculating
            counts  - number of observations per cell. 
            ptu_df - the dataframe containing the p(t|u)
    """
    pass

def updateHierDynProbab(obs, obs_prob, pr_hier_dyn, counts, ptu_df):
    """
        Function to update the hierarchical DYNAMIC probabilities, given:
            Obs - Observations made, as indices according to the order
            Obs_prob = Observation Probabilities p(z|x)
            pr_hier_dyn - Dynamical prior used for recalculating p(u|t)
            counts - number of observations per cell. Also used for recalculating p(u|t)         
            ptu_df - the dataframe containing the p(t|u)
    """
    pass

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
    # print("Counts:\n{}".format(cts))
    # print("P(t|u):\n {}".format(ptu))
    # print("Prior p(u):\n{}".format(prior_u))
    # Step 1: find cells that have already been observed
    obs = np.sum(cts, axis=(0,1)).astype(np.int)
    # Step 2: for each of the observations, run "updatearea()"  With the appropriate values
    post_u = prior_u
    for i in range(obs.size):
        n_obs = obs[i]
        for j in range(n_obs):
            post_u = updatearea(ptu, post_u, classlist[i])
    # print(post_u)
    # Step 3: recalculate p(t|u) with the new p(u)
    class_probab = calchierprob(post_u, ptu)
    # print("Observations:\n{}".format(obs))
    # print("Posterior Area Probabilities:\n{}".format(post_u))
    # print("Updated Object class prior:\n{}".format(class_probab))
    # print("Testline")
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

# Dirichlet functions
def dirichlet_mode(vec_a):
    """
        Calculation of the mode of the dirichlet, coming from Wikipedia
    """
    mode = np.zeros_like(vec_a)
    a0 = vec_a.sum()
    k = vec_a.size
    mode = (vec_a - 1) / (a0 - k)
    return mode

def dirichlet_expected(vec_a):
    """
        Expected value of the dirichlet distribution - coming from Wikipedia
    """
    return vec_a / vec_a.sum()

# Testing functions
def testonehot():
    """
        Just to test if the one-hot representation for displaying works
    """
    carr = colorarr()
    n1 = m1 = max_map_size = 64
    classlist = np.asarray(["house", "pavement", "grass", "tree", "vehicle"])
    gtmap = np.empty((n1,m1))
    gtmap = fillmap_idx(gtmap, classlist)
    img = vis_idx_map(gtmap, carr)
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(img)
    plt.show()

def testhierarchical():
    """
        Function to test whether the hierarchical stuff works
    """
    arealist = np.asarray(["urban", "road", "forest"])
    objectlist = np.asarray(["house", "pavement", "grass", "tree", "vehicle"])
    df = gethierarchprobab(arealist, objectlist)
    print(df)

    gtmap = np.empty((64,64))
    gtmap = fillmap_idx(gtmap, objectlist)

    # Real prior distribution
    out = get_map_counts(gtmap)
    print("Relative Class distribution in the map:\n{}".format(out))

    # Guess the prior distribution over the areas:
    pr_hier = np.asarray([0.1, 0.2, 0.7])
    pred_classes_hier = pr_hier @ df
    print("Predicted Class distribution:\n{}".format(pred_classes_hier))

    ctsmap = np.zeros_like(gtmap)


if __name__=="__main__":

    # testhierarchical()
    # Filename Writing Stuff
    args = parse_args()
    checkarguments(args)
    parentDir = os.path.dirname(__file__)
    outputdir = os.path.abspath(os.path.join(parentDir,'..','tmp', 'results'))

    #### Section 1 - Setup work
    max_map_size = args.dim
    n1 = m1 = max_map_size
    fov = args.fov
    h_overlap = 1-args.overlaph
    v_overlap = 1-args.overlapv
    overlap = (h_overlap, v_overlap)
    likelihood = args.accuracy
    simcase = args.simcase
    transposed = args.transposed

    # TODO: Configs to write into a new file 
    carr = colorarr()
    classlist = np.asarray(["house", "pavement", "grass", "tree", "vehicle"])
    arealist = np.asarray(["urban", "road", "forest"])
    df = gethierarchprobab(arealist, classlist)
    obs_prob = observation_probabilities(classlist, maxim=likelihood)

    # First Level map - TODO: include the random argument
    gtmap=np.empty((n1,m1))
    gtmap = fillmap_idx(gtmap, classlist, scenario=simcase, transpose=transposed)

    # Second Level map - TODO: include these also into the config file!
    real_distribution = get_map_counts(gtmap)
    prior_hierarchical = np.asarray([0.1, 0.2, 0.7]) # best guess of how the distribution of our areas looks like
    pred_classes_hierar = prior_hierarchical @ df
    
    # A map to store the counts
    countsmap = np.zeros_like(gtmap)

    # Maps that are used for predictions:
    dirmap = np.ones_like(gtmap)
    predmap = dirmap / gtmap.shape[2]
    flatmap = np.copy(predmap)

    # two additional maps used for prediction 
    hiermap = np.copy(predmap)          # One that uses the flat prior prediction from our model
    hiermap[:,:] = pred_classes_hierar.to_numpy()
    hiermap_dyn = np.copy(hiermap)      # One that updates the p(u) dynamically

    # Maps that are used for visualisation
    gt_vis = vis_idx_map(gtmap, carr)
    dirmap_vis = np.ones_like(gt_vis)
    map_vis = np.copy(dirmap_vis)
    pred_vis = np.copy(dirmap_vis)
    # Two additional maps for visualization
    hiermap_vis = np.copy(dirmap_vis)
    hiermap_dyn_vis = np.copy(dirmap_vis)
       
    # Observation probabilites and waypoints
    wps = getflightpattern(n1, m1, fov=fov, overlap=overlap)      # Flight pattern

    # SECTION 2: Visualisation prelims
    # fig, axes = plt.subplots(2, 2)
    t1 = "Ground Truth Map"
    t2 = "Reconstructed Map"
    t3 = "Prediction Map"
    t4 = "Dirichlet Map"
    t5 = "Hierarchical prediction Map"
    t6 = "Dynamical Hierarchical prediction Map"
    # axes[0,0].title.set_text(t1)
    # axes[0,1].title.set_text(t2)
    # axes[1,0].title.set_text(t3)
    # axes[1,1].title.set_text(t4)
    hundredpercent = wps.shape[0]
    tenpercent = hundredpercent // 10
    percentcounter = 0
    for i in range(wps.shape[0]-1):
        if i % tenpercent == 0:
            print("Got {} Percent".format(percentcounter*10))
            percentcounter += 1
        # def animate(i):
        # indices that are currently visible
        x_min, x_max, y_min, y_max = retrieveVisibleFields(wps[i], fov=fov)
        gt = gtmap[x_min:x_max, y_min:y_max]    #  Ground Truth area
        obs = gensampleidx(gt, obs_prob)        # make Observations

        # Getting the priors for the maps        
        pr_flat = flatmap[x_min:x_max, y_min:y_max,:]
        pr_dir = dirmap[x_min:x_max, y_min:y_max,:]
        pr_pred = predmap[x_min:x_max, y_min:y_max,:]
        # For the hierarchical function
        counts = countsmap[x_min:x_max, y_min:y_max,:]
        # pr_hier = hiermap[x_min:x_max, y_min:y_max,:]         # May not need those
        # pr_hier_dyn = hiermap_dyn[x_min:x_max, y_min:y_max,:] # May not need them
        # Updating the counts
        counts = updatecounts(counts, obs)
        countsmap[x_min:x_max, y_min:y_max, :] = counts

        # Update the probabilities
        post_flat = updateprobab(obs, obs_prob, pr_flat)
        post_pred = updateprobab(obs, obs_prob, pr_pred)
        post_dir =  updateDir(obs, pr_dir)
        # Udpate the hierarchical probabilities # TODO: Continue here - never really need to do this?
        # post_hier = updateHierProbab(obs, obs_prob, pr_hier, counts, df)
        # post_hier_dyn = updateHierDynProbab(obs, obs_prob, pr_hier_dyn, counts, df)

        # Re-incorporate the information into the map
        flatmap[x_min:x_max, y_min:y_max] = post_flat
        predmap[x_min:x_max, y_min:y_max] = post_pred
        dirmap[x_min:x_max, y_min:y_max] = post_dir

        # Predict the next step
        xmin_pred, xmax_pred, ymin_pred, ymax_pred = retrieveVisibleFields(wps[i+1], fov=fov)
        fustates = predmap[xmin_pred:xmax_pred, ymin_pred:ymax_pred]
        fustates_dir = dirmap[xmin_pred:xmax_pred, ymin_pred:ymax_pred]
        nst_pred = pred_flat(fustates)
        nst_dir = pred_dir(fustates_dir)
        # Hierarchical prediction:
        dyn_pr = hiermap_dyn[xmin_pred:xmax_pred, ymin_pred:ymax_pred, :]
        cts_fut = countsmap[xmin_pred:xmax_pred, ymin_pred:ymax_pred, :]
        pred_dyn = dynamic_prediction(cts_fut, df, prior_hierarchical, classlist) 

        # Re-incorporate prediction-values into the map 
        predmap[xmin_pred:xmax_pred, ymin_pred:ymax_pred] = nst_pred
        dirmap[xmin_pred:xmax_pred, ymin_pred:ymax_pred] = nst_dir
        # hierarchical incorporation - where nothing has been observed yet
        zer_idcs = np.where(np.sum(cts_fut, axis=2) == 0)
        dyn_pr[zer_idcs] = pred_dyn
        hiermap_dyn[xmin_pred:xmax_pred, ymin_pred:ymax_pred, :] = dyn_pr

        # Do the visibility lookup
        post_vis = lookupColorFromPosterior(carr, post_flat)
        map_vis[x_min:x_max, y_min:y_max] = post_vis
        post_pred_vis = lookupColorFromPosterior(carr, post_pred)
        pred_vis[x_min:x_max, y_min:y_max] = post_pred_vis
        post_dir_vis = lookupColorFromPosterior(carr, post_dir)
        dirmap_vis[x_min:x_max, y_min:y_max] = post_dir_vis

        #     # # Plotting section
        #     axes[0,0].clear()
        #     axes[0,0].imshow(gt_vis)
        #     axes[0,0].set(title=t1)
        #     axes[0,1].clear()
        #     axes[0,1].imshow(map_vis)
        #     axes[0,1].scatter(wps[i,1], wps[i,0], s=20, c='red', marker='x')
        #     axes[0,1].set(title=t2+" Waypoint: {}, at x: {}, y: {}".format(i, wps[i,1], wps[i,0]))
        #     # axes[1].scatter(wps[0:i,1], wps[0:i,0], s=15, c='blue', marker='x')
        #     # axes[1].scatter(wps[i+1:-1,1], wps[i+1:-1,0], s=15, c='black', marker='x')
        #     # Plotting the predicted classes
        #     axes[1,0].clear()
        #     axes[1,0].imshow(pred_vis)
        #     axes[1,0].set(title=t3)
        #     axes[1,1].clear()
        #     axes[1,1].imshow(dirmap_vis)
        #     axes[1,1].set(title=t4)

    # ani = matplotlib.animation.FuncAnimation(fig, animate, frames=wps.shape[0]-1, interval=10, repeat=False)
    # plt.show()      

    # Setting up the entropy arrays
    dir_mode = np.ones_like(gtmap)
    dir_exp = np.ones_like(gtmap)
    
    dir_e_e = np.zeros((gtmap.shape[0], gtmap.shape[1]))
    dir_m_e = np.copy(dir_e_e)
    pred_e = np.copy(dir_e_e)
    flat_e = np.copy(dir_e_e)
    
    # Hierarchical stuff for recalculating
    postmap_dyn = np.zeros_like(gtmap)
    postmap_hier = np.zeros_like(gtmap)
    dyn_e = np.copy(dir_e_e)
    hier_e = np.copy(dir_e_e)

    # Looping over all map elements
    for i in range(gtmap.shape[0]):
        for j in range(gtmap.shape[1]):
            dir_mode[i,j] = dirichlet_mode(dirmap[i,j,:])
            dir_exp[i,j] = dirichlet_expected(dirmap[i,j,:])

            # getting the Ground truth vector for the entropy
            gt = gtmap[i,j,:]
            dir_e_e[i,j] = cross_entropy(gt, dir_exp[i,j,:])
            dir_m_e[i,j] = cross_entropy(gt, dir_mode[i,j,:])
            pred_e[i,j] = cross_entropy(gt, predmap[i,j,:])
            flat_e[i,j] = cross_entropy(gt, flatmap[i,j,:])

            # For the hierarchical maps: Recreate the posterior
            postmap_dyn[i,j,:] = recreate_posterior(hiermap_dyn[i,j,:], countsmap[i,j,:], obs_prob)
            postmap_hier[i,j,:] = recreate_posterior(hiermap[i,j,:], countsmap[i,j,:], obs_prob)
            dyn_e[i,j] = cross_entropy(gt, postmap_dyn[i,j,:])
            hier_e[i,j] = cross_entropy(gt, postmap_hier[i,j,:])

    # print("Total Entropy for all: ")
    # print("Flat Updates: {}".format(flat_e.flatten().sum()))
    # print("Predicted Updates: {}".format(pred_e.flatten().sum()))
    # print("Hierarchical Prediction: {}".format(hier_e.flatten().sum()))
    # print("Dynamic Predictions: {}".format(dyn_e.flatten().sum()))

    datadict = {}
    datadict["Counts"] = countsmap
    datadict["Ground Truth"] = gtmap
    datadict["Hierarchical-Dynamic"] = hiermap_dyn
    datadict["Hierachical-Pre"] = hiermap
    datadict["Predicted"] = predmap
    datadict["Flat"] = flatmap

    # Use pandas.ExcelWriter for this?
    configs = {}
    configs["Colours"] = pd.DataFrame(data=carr, index=classlist)
    configs["Hierarch"] = df
    configs["Observation"] = pd.DataFrame(data=obs_prob, index=classlist)
    configs["Real_Dist"] = pd.DataFrame(data=real_distribution, index=classlist)
    configs["Hier_Prior"] = pd.DataFrame(data=prior_hierarchical, index=arealist)
    configs["Pred_Hier"] = pd.DataFrame(data=pred_classes_hierar, index=classlist)

    # Write datadict or Plot
    save_results(outputdir, args, datadict=datadict, configs=configs)
    # plotresults(datadict, args)
   
