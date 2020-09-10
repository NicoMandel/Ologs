#!/usr/bin/env python3

"""
    Example file to run an Experiment.

    TODO: Indexing into the right higher level 
    block will require indexing like CUDA! blockDim.x * blockidx.X + ThreadIdx.x
    # TODO: resolution dependant on distance

    # TODO: Color mapping of imshow of matplotlib through [this](https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.imshow.html)
"""

import numpy as np
import matplotlib.pyplot as plt 

class Colordict():
    """
        Helper Class to color code the class names for visualisation
    """

    def __init__(self):
        cdict = {}
        cdict["House"] = tuple([0.921, 0.117, 0.137])   # Red
        cdict["Pavement"] = tuple([0.662, 0.674, 0.647])   # Gray
        cdict["Grass"] = tuple([0.384, 0.976, 0.411])
        cdict["Tree"] = tuple([0.164, 0.576, 0.219])
        cdict["Vehicle"] = tuple([0.172, 0.533, 0.866])
        cdict["Urban"] = tuple([0.713, 0.207, 0.305])
        cdict["Forest"] = tuple([0.149, 0.380, 0.141])
        cdict["Road"] = tuple([0.49, 0.49, 0.49])
        cdict["Grasslands"] = tuple([0.713, 0.89, 0.631])
        self.cdict = cdict

    def lookupcolor(self, cname):
        """
            Function to return the RGB representation of a color, specified by the class name
        """
        try:
            return self.cdict[cname]
        except LookupError as e:
            raise e

def createMap(n, m):
    """
        Function to create a map of size n x m
        returns a numpy array
        row-major
    """

    return np.empty([n, m], dtype=object)

def fillmap(gt, cd, scenario):
    """
        Helper function to create the ground truth map according to the 
    """
    pass

def visualize_map(gtmap, cd):
    """
        Function to visualize the map using matplotlib and the color codes defined in cd  
    """
    pass

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

def obs_probab(classlist, maxim=0.8):
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
    return arr
    
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

    pass
    # TODO: Continue here

def updateMap(x_min, x_max, y_min, y_max, posterior, lmap):
    """
        Function that takes the new probabilities with the observations and the original maps and update the function on the right indices
    """
    lmap[x_min:x_max, y_min:y_max] = posterior

def makeObs(gt, obs_probab):
    """
        Returns an observation based on the Ground truth and the Observation probability
    """
    pass
    # TODO:  

def entropy(vec):
    """
        Function that returns the Entropy as the -1 * sum(p(x) * ln(p(x)))
    """
    lnvec = np.log(vec)
    return np.sum(np.dot(vec, lnvec)) * -1.0

if __name__=="__main__":

    max_map_size = 64
    blockDim = 2            # TODO: Indexing in the inverse hierarchical structure, similar to CUDA
    n1 = m1 = max_map_size
    n2 = m2 = max_map_size/2
    fov = 2

    # n3 = m3 = max_map_size/(2)            Keep to 2 levels for now
    gtmap = createMap(n1,m1)            # Ground Truth Map
    l1map = createMap(n1,m1)            # belief map
    l1_classlist = ["House", "Pavement", "Grass", "Tree", "Vehicle"]
    l2_classlist = ["Urban", "Forest", "Road", "Grasslands"]
    
    # Making a Map
    cd = Colordict()
    gtmap = fillmap(gtmap, "1")
    visualize_map(gtmap, cd)
    
    # Observation probabilites and waypoints
    obs = obs_probab(l1_classlist)
    wps = getpattern(n1,m1, fov)      # Flight pattern

    # iterative procedure from here on out
    last_wp = wps[-1]
    x_min, x_max, y_min, y_max = retrieveVisibleFields(last_wp, fov=fov)
    vis_fields = l1map[x_min:x_max, y_min:y_max]
    
    # TODO: Assign a prior distribution to the probability of observing the classes

    for wp in wps:
        """
            This is where the iterative updates take place. Using the Ground Truth data (gt), the observation likelihood and the prior over the previous map
        """
        x_min, x_max, y_min, y_max = retrieveVisibleFields(wp, fov=fov)
        
        gt = gtmap[x_min:x_max, y_min:y_max]    #  Ground Truth
        prior = l1map[x_min:x_max, y_min,y_max] # Prior

        # Make observation
        obs = makeObs(gt, obs_probab)

        # TODO: how does the observed thing get incorporated into it?
        posterior = updateprobab(obs, obs_probab, prior)

        # Re-incorporate the information into the map


        



    print("Test Done")