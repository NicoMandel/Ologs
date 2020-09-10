#!/usr/bin/env python3

"""
    Example file to run an Experiment.
    Everything  is lowercase!!!

    TODO: Indexing into the right higher level 
    block will require indexing like CUDA! blockDim.x * blockidx.X + ThreadIdx.x

    TODO: Color Mapping as a pandas dataframe - use the transpose method after creation

    # TODO: Color mapping of imshow of matplotlib through [this](https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.imshow.html)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
    return np.empty([n, m, l], dtype=object)

def createGTMap(n,m):
    """
        Function to create a ground truth map
    """
    return np.empty([n,m], dtype=object)

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
    

def visualize_map(gtmap, cd, returnarray=False):
    """
        Function to visualize the map using matplotlib and the color codes defined in cd  
    """
    vis_map = np.empty((gtmap.shape[0],gtmap.shape[1], 4))
    # Fill the word with the color
    for j in range(gtmap.shape[0]):
        for k in range(gtmap.shape[1]):
            vis_map[j,k] = cd.lookupcolor(gtmap[j,k])

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
    """

    pattern = []
    for i in range(y_max-fov):
        for j in range(x_max-fov):
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
    for i in obs.shape[0]:
        for j in obs.shape[1]:
            pr = prior[i,j]
            vec = obs_probab.loc[obs[i,j],:]
            po = vec*prior
            po = po/po.sum()
            post[i,j] = po

    return post
            
    
def updateMap(x_min, x_max, y_min, y_max, posterior, lmap):
    """
        Function that takes the new probabilities with the observations and the original maps and update the function on the right indices
    """
    lmap[x_min:x_max, y_min:y_max] = posterior

def makeObs(gt, obs_probab, classlist):
    """
        Returns an observation based on the Ground truth and the Observation probability
    """

    obs = np.empty((obs_probab.shape[0], obs_probab.shape[1]))
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            cl_name = gt[i,j]
            prob = obs_probab.loc[cl_name,:]
            obs[i,j] = np.random.choice(classlist, p=prob)
    return obs

def entropy(vec):
    """
        Function that returns the Entropy as the -1 * sum(p(x) * ln(p(x)))
    """
    lnvec = np.log(vec)
    return np.sum(np.dot(vec, lnvec)) * -1.0

if __name__=="__main__":
    cdf = colordf()

    max_map_size = 64
    blockDim = 2            # TODO: Indexing in the inverse hierarchical structure, similar to CUDA
    n1 = m1 = max_map_size
    n2 = m2 = max_map_size/2
    fov = 2

    # n3 = m3 = max_map_size/(2)            Keep to 2 levels for now
    l1_classlist = np.asarray(["house", "pavement", "grass", "tree", "vehicle"])
    # l2_classlist = ["urban", "forest", "road", "grasslands"]
    gtmap = createGTMap(n1, m1)            # Ground Truth Map
    l1map = createMap(n1, m1, l1_classlist.size)            # belief map
    
    # Making a Map
    cd = Colordict()
    gtmap = fillmap(gtmap, l1_classlist, 1)
    gt_vismap = visualize_map(gtmap, cd, True)
    
    # Observation probabilites and waypoints
    obs_prob = observation_probabilities(l1_classlist)
    wps = getpattern(n1, m1, fov)      # Flight pattern

    # iterative procedure from here on out
    # last_wp = wps[-1]
    # x_min, x_max, y_min, y_max = retrieveVisibleFields(last_wp, fov=fov)
    # vis_fields = l1map[x_min:x_max, y_min:y_max]

    # Filling the PRIOR with values from the FUCKING SHIT FUCK FUCK FUCK
    l1map.fill(1.0/l1_classlist.size)
    l1map_vis = np.empty((l1map.shape[0], l1map.shape[1]), dtype="object")

    # TODO: Visualisation, updating incrementally:
        # Waypoints
        # FoV
        # Belief Map
    fig, axes = plt.subplots(2, 2)
    for wp in wps:
        """
            This is where the iterative updates take place. Using the Ground Truth data (gt), the observation likelihood and the prior over the previous map
        """
        # indices that are currently visible
        x_min, x_max, y_min, y_max = retrieveVisibleFields(wp, fov=fov)
        gt = gtmap[x_min:x_max, y_min:y_max]    #  Ground Truth
        prior = l1map[x_min:x_max, y_min,y_max] # Prior

        # Make observation
        obs = makeObs(gt, obs_prob, l1_classlist)

        # how does the observed thing get incorporated into it?
        posterior = updateprobab(obs, obs_prob, prior)
        # Re-incorporate the information into the map
        l1map[x_min:x_max, y_min:y_max] = posterior


        # Plotting section
        axes[0].imshow(gt_vismap)
        axes[0].set(title="Ground Truth Map")

        # clear the plots
         
        
    print("Test Done")