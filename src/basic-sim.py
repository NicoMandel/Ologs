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

def fillmap(gt, classlist, scenario):
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

def definel2probabilities(l2_classlist, l1_classlist):
    """
        Hand-Designed function for the second level
    """
    df = pd.DataFrame(index=l2_classlist, columns=l1_classlist)

# Helper function - on the side
def entropy(vec):
    """
        Function that returns the Entropy as the -1 * sum(p(x) * ln(p(x)))
    """
    lnvec = np.log(vec)
    return np.sum(np.dot(vec, lnvec)) * -1.0

if __name__=="__main__":

    #### Section 1 - Setup work
    cdf = colordf()

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
    # Making a Map
    gtmap = fillmap(gtmap, l1_classlist, 1)
    gt_vismap = visualize_map(gtmap, cdf, True)
    
    # Second Level Map:
    l2_classlist = np.asarray(["urban", "forest", "road", "grasslands"])
    l2map = createMap(n2, m2, l2_classlist.size)
    l2probabs = definel2probabilities(l2_classlist, l1_classlist)
       
    # Observation probabilites and waypoints
    obs_prob = observation_probabilities(l1_classlist)
    wps = getpattern(n1, m1, fov)      # Flight pattern

    # SECTION 2: Visualisation
    fig, axes = plt.subplots(1, 2)
    t1 = "Ground Truth Map"
    t2 = "Reconstructed Map"
    axes[0].title.set_text(t1)
    axes[1].title.set_text(t2)

    def animate(i):
        # indices that are currently visible
        x_min, x_max, y_min, y_max = retrieveVisibleFields(wps[i], fov=fov)
        gt = gtmap[x_min:x_max, y_min:y_max]    #  Ground Truth
        prior = l1map[x_min:x_max, y_min:y_max,:] # Prior

        # Make observation
        obs = makeObs(gt, obs_prob, l1_classlist)

        # how does the observed thing get incorporated into it?
        posterior = updateprobab(obs, obs_prob, prior)
        # Re-incorporate the information into the map
        l1map[x_min:x_max, y_min:y_max] = posterior


        post_vis = lookupColorFromPosterior(cdf, posterior, l1_classlist)
        l1map_vis[x_min:x_max, y_min:y_max] = post_vis

        # Plotting section
        axes[0].clear()
        axes[0].imshow(gt_vismap)
        axes[0].set(title=t1)

        # Plotting the predicted classes
        axes[1].clear()
        axes[1].imshow(l1map_vis)
        axes[1].scatter(wps[i,1], wps[i,0], s=20, c='red', marker='x')
        axes[1].set(title=t2+"\tWaypoint: {}, at x: {}, y: {}".format(i, wps[i,1], wps[i,0]))
        # axes[1].scatter(wps[0:i,1], wps[0:i,0], s=15, c='blue', marker='x')
        # axes[1].scatter(wps[i+1:-1,1], wps[i+1:-1,0], s=15, c='black', marker='x')

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=wps.shape[0], interval=10, repeat=False)
    plt.show()        
        
    print("Test Done")