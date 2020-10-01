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

def colordf():
    """
        Helper Function to turn the colors into a DatFrame - for ease of lookup with np
        Returns a df object
    """
    df = pd.DataFrame()
    df["house"] = [0.921, 0.117, 0.137, 1.0]
    df["pavement"] = [0.662, 0.674, 0.647, 1.0]
    df["grass"] = [0.384, 0.976, 0.411, 1.0]
    df["tree"] = [0.164, 0.576, 0.219, 1.0]
    df["vehicle"] = [0.172, 0.533, 0.866, 1.0]
    df["urban"] = [0.713, 0.207, 0.305, 1.0]
    df["forest"] = [0.149, 0.380, 0.141, 1.0]
    df["road"] = [0.49, 0.49, 0.49, 1.0]
    df["grasslands"] = [0.713, 0.89, 0.631, 1.0]
    return df

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
    
def createMap(n, m, l):
    """
        Function to create a map of size n x m x l
        returns a numpy array
        row-major
    """
    return np.empty((n, m, l), dtype=object)

def createGTMap(n,m):
    """
        Function to create a ground truth map
    """
    return np.empty((n,m), dtype=object)

def fillmap(gt, classlist, scenario=1):
    """
        Helper function to create the ground truth map according to the Scenario and the classlist
    """
    # Dimensions - n rows, m cols
    n = gt.shape[0]
    m = gt.shape[1]
    if scenario==1:
        # Divide columns into 4 sections
        fourth = m//4

        # First: Everything is grass
        gt.fill("grass")
        # second fourth is "pavement"
        gt[:,1*fourth+1:2*fourth] = "pavement"

        # in Fourth fourth, divide rows into 8 block
        eigth = n//8
        for i in range(eigth):
            if i%2==0:
                # Put houses into the even blocks
                r_idx = i*eigth
                gt[r_idx:r_idx+eigth,3*fourth:3*fourth+3] = "house"
        
        # In third block, put a few trees there
        x = np.asarray(range(0,n,5))
        gt[x,2*fourth+3] = "tree"

        # In second Block, put two vehicles there
        quat = m//4
        gt[quat:quat+3,fourth+int(0.5*fourth)-2:fourth+int(0.5*fourth)] = "vehicle"
        gt[2*quat:2*quat+3,fourth+int(0.5*fourth)+1:fourth+int(0.5*fourth)+3] = "vehicle"
        
    return gt

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

def visualize_map(gtmap, cdf, returnarray=False):
    """
        Function to visualize the map using matplotlib and the color codes defined in cd  
    """
    vis_map = np.empty((gtmap.shape[0],gtmap.shape[1], 4))
    # # Fill the word with the color
    for j in range(gtmap.shape[0]):
        for k in range(gtmap.shape[1]):
            vis_map[j,k] =cdf[gtmap[j,k]]

    # visualize it with matplotlib imshow if returnarray is False, else return it
    if not returnarray:
        fig, ax = plt.subplots()
        ax.imshow(vis_map)
        plt.show()
    else:
        return vis_map

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

def observation_probabilities(classlist, maxim=0.8):
    """
        Returns an array with the observation probabilities for each class.
        The observation probabilities are calculated using maxim as p(o|y) and a uniform distribution over all other values
    """

    num_classes = len(classlist)
    conf_probab = (1.0-maxim)/(num_classes-1)
    arr = np.empty([num_classes, num_classes])
    np.fill_diagonal(arr, maxim)
    off_diag = np.where(~np.eye(num_classes,dtype=bool))
    arr[off_diag] = conf_probab
    df = pd.DataFrame(arr, columns=classlist, index=classlist)
    return df
    
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

def updateprobab(obs, obs_probab, prior):
    """
        Prior: Prior probabilities over the maps:
        obs: Observations made
        obs_probab: probability of making the observations
        Returns: posterior over the map
    """
    # Prior is a 3D array
    post = np.empty_like(prior)
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            pr = prior[i,j]
            vec = obs_probab.loc[obs[i,j],:]
            po = vec*pr
            po = po/po.sum()
            post[i,j] = po

    return post

def lookupColorFromPosterior(cdf, post, classlist):
    """
        TODO: Fix the argmax thing
        Cdf is a dataframe with the color codes, Post a 3-dim. array with the posterior probabilities over the classes
        for each 2-dim cell
    """     
    col = np.empty((post.shape[0], post.shape[1], cdf.shape[0]))
    # print(post)
    # print(col)
    idxmax = np.asarray(np.unravel_index(np.argmax(post, axis=2), post.shape))[2]
    for i in range(idxmax.shape[0]):
        for j in range(idxmax.shape[1]):
            col[i,j] = cdf[classlist[idxmax[i,j]]]
    # alternative: col = cdf[classlist[np.argmax(post, axis=2)]]
    return col

def updateMap(x_min, x_max, y_min, y_max, posterior, lmap):
    """
        Function that takes the new probabilities with the observations and the original maps and update the function on the right indices
    """
    lmap[x_min:x_max, y_min:y_max] = posterior

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

def gensampleidx(gt, pmax=0.8):
    """
        generates an index with p=0.8 of the same kind and 1-0.8 /n-1 otherwise
    """
    idx = np.nonzero(gt)
    p = (np.ones(gt.size) - pmax) / (gt.size - 1) 
    p[idx] = pmax
    sam = np.arange(gt.size)
    sample = np.random.choice(sam,p=p)
    return sample

def definel2probabilities(l2_classlist, l1_classlist):
    """
        Hand-Designed function for the second level
    """
    df = pd.DataFrame(index=l2_classlist, columns=l1_classlist)

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

# TODO in here!
def pred_informed(fut_states, init_vector):
    """
        A more informed version to predict the states of the next cell.
        See "bayes-discr-1D.py" in folder ../tmp for details
        Record an extra value for the number of observations per cell at the end of the map vector. This number is then used to weight the prediction of the next cell
        requires the initial vector used to construct the array
        inverse indexing following this example: https://stackoverflow.com/questions/25330959/how-to-select-inverse-of-indexes-of-a-numpy-array
    """
    num_states = fut_states.shape[2]-1
    num_cells = fut_states.shape[0] * fut_states.shape[1]
    # unif_vec = np.ones(num_states, dtype=float)/num_states
    _z = np.isin(fut_states[...,:-1], init_vector, True)
    n_obs_cells = np.transpose(np.all(_z,axis=2).nonzero())
    obs_cells = np.transpose(np.all(~_z, axis=2).nonzero())
    r_obs = obs_cells.shape[0] / num_cells
    # obs = fut_states[~x]
    num_obs = fut_states[obs_cells[:,0], obs_cells[:,1], -1]

    ### Alternative for finding the cells that have not been observed:
    m_counts = fut_states[...,-1]
    tot_counts = m_counts.sum()
    zer_cells = np.argwhere(m_counts < 1e-3)
    # nzer_cells = np.nonzero(m_counts)
    # TODO: Continue here
    # use the value of the counts to multiply the probability and then normalise it. Also use the amount of unobserved cells.
    mc_norm = m_counts / tot_counts         # m_counts for the unobserved cells are 0, therefore we get 0 for these anyways
    pr = mc_norm * fut_states
    fut_states[zer_cells,:-1] = pr
    return fut_states 
    
def get_map_counts(map1):
    """
        Function to return the (relative) counts of each class available in the map.
        For evaluation with priors
        Requires a 3D map, where the 3rd dimension is a vector of 0s and ones and it counts the 1s
    """
    n_cells = map1.shape[0] * map1.shape[1]
    out = np.count_nonzero(map1, axis=(0,1)) / n_cells
    return out

def assign_prior(map1, areadist_vec, area_class_mat):
    """
        function to assign a more informed prior - sum over the assumed distribution of areas multiplied by the observation probability of that class
        p(t|u) = p(u|t) * p(t) / p(t) || with p(t) = sum(p(u|t) p(t))
    """
    vec = areadist_vec.T @ area_class_mat
    map1[...,:] = vec
    return map1

# Helper function - on the side
def entropy(vec):
    """
        Function that returns the Entropy as the -1 * sum(p(x) * ln(p(x)))
    """
    lnvec = np.log(vec)
    return np.sum(np.dot(vec, lnvec)) * -1.0

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
    gtmap = createGTMap(n1, m1)
    gtmap = fillmap_idx(gtmap, classlist)
    img = vis_idx_map(gtmap, carr)
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(img)
    plt.show()



if __name__=="__main__":

    # testonehot()

    #### Section 1 - Setup work
    cdf = colordf()
    carr = colorarr()

    max_map_size = 64
    blockDim = 2            # TODO: Indexing in the inverse hierarchical structure, similar to CUDA
    n1 = m1 = max_map_size
    n2 = m2 = max_map_size//2
    fov = 1

    # First Level map
    l1_classlist = np.asarray(["house", "pavement", "grass", "tree", "vehicle"])
    gtmap = createGTMap(n1, m1)            # Ground Truth Map
    l1map = createMap(n1, m1, l1_classlist.size)            # belief map
    l1map.fill(1.0/l1_classlist.size)
    l1map_vis = np.ones((l1map.shape[0], l1map.shape[1], 4))

    # Dirichlet Map:
    dirmap = np.ones_like(l1map)
    dirmap_u = np.copy(dirmap)
    dirmap_vis = np.copy(l1map_vis)

    # First level map used for prediction:
    l1pred = np.copy(l1map)
    l1pred_vis = np.copy(l1map_vis)

    # Making a Map
    gtmap = fillmap(gtmap, l1_classlist, 1)
    gt_vismap = visualize_map(gtmap, cdf, True)
    # indexed map
    gtmap_idx = fillmap_idx(gtmap, l1_classlist)
    gtmap_idx_vis = vis_idx_map(gtmap_idx, carr)
    
    # Second Level Map:
    # l2_classlist = np.asarray(["urban", "forest", "road", "grasslands"])
    # l2map = createMap(n2, m2, l2_classlist.size)
    # l2probabs = definel2probabilities(l2_classlist, l1_classlist)
       
    # Observation probabilites and waypoints
    obs_prob = observation_probabilities(l1_classlist)
    wps = getpattern(n1, m1, fov)      # Flight pattern

    # SECTION 2: Visualisation
    fig, axes = plt.subplots(2, 2)
    t1 = "Ground Truth Map"
    t2 = "Reconstructed Map"
    t3 = "Prediction Map"
    axes[0,0].title.set_text(t1)
    axes[0,1].title.set_text(t2)
    axes[1,0].title.set_text(t3)

    def animate(i):
        # indices that are currently visible
        x_min, x_max, y_min, y_max = retrieveVisibleFields(wps[i], fov=fov)
        gt = gtmap[x_min:x_max, y_min:y_max]    #  Ground Truth
        prior = l1map[x_min:x_max, y_min:y_max,:] # Prior

        # Prior for the prediction map
        prior_pred = l1pred[x_min:x_max, y_min:y_max,:]

        # Make observation
        obs = makeObs(gt, obs_prob, l1_classlist)

        # how does the observed thing get incorporated into it?
        posterior = updateprobab(obs, obs_prob, prior)
        # Posterior for the prediction map:
        post_pred = updateprobab(obs, obs_prob, prior_pred)
        
        # Re-incorporate the information into the map
        l1map[x_min:x_max, y_min:y_max] = posterior
        # Re-incorporate into the prediction map
        l1pred[x_min:x_max, y_min:y_max] = post_pred

        # Prediction step
        xmin_pred, xmax_pred, ymin_pred, ymax_pred = retrieveVisibleFields(wps[i+1], fov=fov)
        fut_states = l1pred[xmin_pred:xmax_pred, ymin_pred:ymax_pred]
        nstates = pred_flat(fut_states)
        # Re-incorporate prediction-values into the map 
        l1pred[xmin_pred:xmax_pred, ymin_pred:ymax_pred] = nstates

        # Do the visibility lookup
        post_vis = lookupColorFromPosterior(cdf, posterior, l1_classlist)
        l1map_vis[x_min:x_max, y_min:y_max] = post_vis

        # Do the visibility lookup for the prediction - map
        post_pred_vis = lookupColorFromPosterior(cdf, post_pred, l1_classlist)
        l1pred_vis[x_min:x_max, y_min:y_max] = post_pred_vis

        # Plotting section
        axes[0,0].clear()
        axes[0,0].imshow(gt_vismap)
        axes[0,0].set(title=t1)

        # Plotting the predicted classes
        axes[0,1].clear()
        axes[0,1].imshow(l1map_vis)
        axes[0,1].scatter(wps[i,1], wps[i,0], s=20, c='red', marker='x')
        axes[0,1].set(title=t2+"\tWaypoint: {}, at x: {}, y: {}".format(i, wps[i,1], wps[i,0]))
        # axes[1].scatter(wps[0:i,1], wps[0:i,0], s=15, c='blue', marker='x')
        # axes[1].scatter(wps[i+1:-1,1], wps[i+1:-1,0], s=15, c='black', marker='x')

        axes[1,0].clear()
        axes[1,0].imshow(l1pred_vis)
        axes[1,0].set(title=t3)


    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=wps.shape[0], interval=10, repeat=False)
    plt.show()        
        
    print("Test Done")