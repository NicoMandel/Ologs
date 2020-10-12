#!/usr/bin/env python3

import sys
import os.path
import numpy as np
from argparse import ArgumentParser
import h5py
import runsim as rs
import time
import pandas as pd

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
    parser.add_argument("-r", "--random", default=False, action="store_true", help="Whether object locations should be randomly generated or not. Only affects simulation case 2")
    parser.add_argument("--testconfig", default=False, action="store_true", help="Whether the convoluted case of overlapping areas should be used")
    parser.add_argument("-p", "--ptu", default=0, type=int, help="Which Ptu to take. If 0: choose predefined set. If 1: Choose Dynamic, if 2: Choose biggest difference")
    args = parser.parse_args()
    return args

def readh5(fname):
    """
        reading the h5 file, and converting the h5py format to numpy arrays
    """
    datadict = {}
    with h5py.File(fname, 'r') as f:
        for k, v in f.items():
            datadict[k] = v.value
    return datadict

def checkarguments(args, curr_cases=2):
    """
        Function to check the arguments for admissible/inadmissible stuff
    """

    fov = args.fov
    h_overlap = 1-args.overlaph
    v_overlap = 1-args.overlapv
    overlap = (h_overlap, v_overlap)
    
    #  Check on the Overlap 
    for over in overlap:
        if (over % (1.0 / (2*fov)) != 0):
            raise ValueError("Error: Overlap {} Not a multiple of: {}".format(
                over, (1.0 / (2*fov))
            ))
    # Check on the Sim Cases
    if args.simcase > curr_cases:
        raise ValueError("Sim Cases invalid. Only {} cases currently exist".format(curr_cases))
    
    # Check if random is true if it is simcase 2
    if args.random:
        if not(args.simcase == 2):
            print("Random Flag set, but not case 2 selected. Irrelevant parameter ignored")

    # Check if the -p flag is smaller than 3
    if args.ptu > 2:
        raise ValueError("Error: p-flag not a valid configuration: {}, choose a value between 0 and 2".format(args.ptu))
    
    # Check if dims is divisble by 4
    if args.dim % 4 != 0:
        raise ValueError("Dimension {} not divisble by 4".format(args.dim))

    # If everything is fine, continue:
    print("All checks passed. Continuing with case:")
    ar = vars(args)
    for k,v in ar.items():
        print("{}: {}".format(
            k,v
        ))

# Set the observation likelihood
def observation_probabilities(num_classes, maxim=0.8):
    """
        Returns an array with the observation probabilities for each class.
        The observation probabilities are calculated using maxim as p(z|q) and a uniform distribution over all other values
    """

    conf_probab = (1.0-maxim)/(num_classes-1)
    arr = np.empty([num_classes, num_classes])
    np.fill_diagonal(arr, maxim)
    off_diag = np.where(~np.eye(num_classes,dtype=bool))
    arr[off_diag] = conf_probab
    return arr

# For case 0 set the combined probabilities
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
    return df.to_numpy()

if __name__=="__main__":
    # Simulation arguments
    try:
        args = parse_args()
        checkarguments(args)
    except ValueError as e:
        sys.exit(1)
    
    # Saving directory
    parentDir = os.path.dirname(__file__)
    indir = os.path.abspath(os.path.join(parentDir, 'tmp'))

    # Fixed parameters
    noobjects = 5
    noareas = 3
    classlist = np.asarray(["house", "pavement", "grass", "tree", "vehicle"])
    arealist = np.asarray(["urban", "road", "forest"])


    if args.ptu == 0:
        # This is where ptu is predefined
        pu =  np.asarray([0.1, 0.2, 0.7])
        ptu = gethierarchprobab(arealist, classlist)
    elif args.ptu == 1:
        # This is where the file "dyn.hdf5" should be loaded
        f = "dyn.hdf5"
        fpath = os.path.abspath(os.path.join(indir,f))
        di = readh5(fpath)
        ptu = di["Ptu"]
        pu = di["Pu"]
    elif args.ptu == 2:
        # This is where the file "diff.hdf5" should be loaded
        f = "diff.hdf5"
        fpath = os.path.abspath(os.path.join(indir,f))
        di = readh5(fpath)
        ptu = di["Ptu"]
        pu = di["Pu"]

    # p(z|t)
    detection_certainty = observation_probabilities(noobjects, args.accuracy)

    # Run the final simmulation
    rs.runsimulation(args, pu, ptu, obs_prob=detection_certainty, arealist=arealist, classlist=classlist)
    
    

