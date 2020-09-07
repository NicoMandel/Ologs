#!/usr/bin/env python3

"""
    Example file to run an Experiment.

    TODO: Indexing into the right higher level 
    block will require indexing like CUDA! blockDim.x * blockidx.X + ThreadIdx.x
    # TODO: resolution dependant on distance
"""
import numpy as np

def createMap(n, m):
    """
        Function to create a map of size n x m
        returns a numpy array
        row-major
    """
    return np.empty([n, m])

def retrieveVisibleFields(wp, fov=1):
    """
        Retrieves the indices of visible fields from the given [x, y] index of the UAV.
        Use the fov + in each direction. Assumes index 0,0 to be the corner between 0,0 and 1,1! 
    """
    x_min = wp[0]-fov+1
    x_max = wp[0]+fov
    y_min = wp[1]-fov+1
    y_max = wp[1]+fov
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
    for i in range(y_max-fov+1):
        for j in range(x_max-fov+1):
            next_wp = tuple([i,j])
            pattern.append(next_wp)

    # Desired output: List of the form: [0,0], [1,0], [2, 0]
    return np.asarray(pattern)



if __name__=="__main__":
    max_map_size = 64
    blockDim = 2            # TODO: Indexing in the inverse hierarchical structure, similar to CUDA
    n1 = m1 = max_map_size
    n2 = m2 = max_map_size/2
    fov = 1

    # n3 = m3 = max_map_size/(2)            Keep to 2 levels for now
    l1map = createMap(n1,m1)
    classlist = ["House", "Road", "Lawn", "Tree"]
    obs = obs_probab(classlist)
    x = getpattern(n1,m1)
    last_wp = x[-1]
    x_min, x_max, y_min, y_max = retrieveVisibleFields(last_wp)
    vis_fields = l1map[x_min:x_max, y_min:y_max]
    
    print("Test Done")