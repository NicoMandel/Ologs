#!/usr/bin/env python3

"""
    Example file to run an Experiment.
    Everything  is lowercase!!!

    TODO: Indexing into the right higher level 
    block will require indexing like CUDA! blockDim.x * blockidx.X + ThreadIdx.x

    # TODO: Color mapping of imshow of matplotlib through [this](https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.imshow.html)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation


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

def fillmap_idx(gt, classlist, scenario=1):
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
            idx = np.nonzero(gt[i,j])
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

# Evaluation
def cross_entropy(vec_true, vec_pred):
    """
        cross entropy loss for a single element. Following the definition of:
        https://youtu.be/ErfnhcEV1O8?t=579
    """
    return np.sum(vec_true*np.log(vec_pred)) * (-1.0)

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


if __name__=="__main__":

    #### Section 1 - Setup work
    carr = colorarr()
    max_map_size = 64
    n1 = m1 = max_map_size
    fov = 1

    # First Level map
    classlist = np.asarray(["house", "pavement", "grass", "tree", "vehicle"])
    gtmap=np.empty((n1,m1))
    gtmap = fillmap_idx(gtmap, classlist)

    # Maps that are used for predictions:
    dirmap = np.ones_like(gtmap)
    predmap = dirmap / gtmap.shape[2]
    flatmap = np.copy(predmap)   

    # Maps that are used for visualisation
    gt_vis = vis_idx_map(gtmap, carr)
    dirmap_vis = np.ones_like(gt_vis)
    map_vis = np.copy(dirmap_vis)
    pred_vis = np.copy(dirmap_vis)
       
    # Observation probabilites and waypoints
    obs_prob = observation_probabilities(classlist)
    wps = getpattern(n1, m1, fov)      # Flight pattern

    # SECTION 2: Visualisation prelims
    fig, axes = plt.subplots(2, 2)
    t1 = "Ground Truth Map"
    t2 = "Reconstructed Map"
    t3 = "Prediction Map"
    t4 = "Dirichlet Map"
    axes[0,0].title.set_text(t1)
    axes[0,1].title.set_text(t2)
    axes[1,0].title.set_text(t3)
    axes[1,1].title.set_text(t4)

    def animate(i):
        # indices that are currently visible
        x_min, x_max, y_min, y_max = retrieveVisibleFields(wps[i], fov=fov)
        gt = gtmap[x_min:x_max, y_min:y_max]    #  Ground Truth area
        obs = gensampleidx(gt, obs_prob)        # make Observations

        # Getting the priors for the maps        
        pr_flat = flatmap[x_min:x_max, y_min:y_max,:]
        pr_dir = dirmap[x_min:x_max, y_min:y_max,:]
        pr_pred = predmap[x_min:x_max, y_min:y_max,:]

        # Update the probabilities
        post_flat = updateprobab(obs, obs_prob, pr_flat)
        post_pred = updateprobab(obs, obs_prob, pr_pred)
        post_dir =  updateDir(obs, pr_dir) 

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
        # Re-incorporate prediction-values into the map 
        predmap[xmin_pred:xmax_pred, ymin_pred:ymax_pred] = nst_pred
        dirmap[xmin_pred:xmax_pred, ymin_pred:ymax_pred] = nst_dir

        # Do the visibility lookup
        post_vis = lookupColorFromPosterior(carr, post_flat)
        map_vis[x_min:x_max, y_min:y_max] = post_vis
        post_pred_vis = lookupColorFromPosterior(carr, post_pred)
        pred_vis[x_min:x_max, y_min:y_max] = post_pred_vis
        post_dir_vis = lookupColorFromPosterior(carr, post_dir)
        dirmap_vis[x_min:x_max, y_min:y_max] = post_dir_vis

        # Plotting section
        axes[0,0].clear()
        axes[0,0].imshow(gt_vis)
        axes[0,0].set(title=t1)
        axes[0,1].clear()
        axes[0,1].imshow(map_vis)
        axes[0,1].scatter(wps[i,1], wps[i,0], s=20, c='red', marker='x')
        axes[0,1].set(title=t2+" Waypoint: {}, at x: {}, y: {}".format(i, wps[i,1], wps[i,0]))
        # axes[1].scatter(wps[0:i,1], wps[0:i,0], s=15, c='blue', marker='x')
        # axes[1].scatter(wps[i+1:-1,1], wps[i+1:-1,0], s=15, c='black', marker='x')
        # Plotting the predicted classes
        axes[1,0].clear()
        axes[1,0].imshow(pred_vis)
        axes[1,0].set(title=t3)
        axes[1,1].clear()
        axes[1,1].imshow(dirmap_vis)
        axes[1,1].set(title=t4)

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=wps.shape[0]-1, interval=10, repeat=False)
    plt.show()        

    #TODO: Evaluation function for the cross-entropy
    # 
    #TODO: plotting function for cross-entropy
    print("Test Done")