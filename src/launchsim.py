#!/usr/bin/env python3

import time
import sys
import os.path
import numpy as np

import subprocess

# ARGUMENTS:
# parser = ArgumentParser(description="Path planning module Parameters")
# parser.add_argument("-d", "--dim", default=64, help="Dimensions of the (square) map in fields. Default is 64", type=int)
# parser.add_argument("-f", "--fov", default=1, help="1/2 of the FOV of the UAV. Is used forward AND backward. Default is 1", type=int)
# parser.add_argument("--overlaph", default=0.5, help="Horizontal desired overlap. Default is 0.5", type=float)
# parser.add_argument("--overlapv", default=0.5, help="Vertical desired overlap. Default is 0.5", type=float)
# parser.add_argument("-a", "--accuracy", default=0.8, help="Detection accuracy. Default is 0.8", type=float)
# parser.add_argument("-t", "--transposed", default=False, help="Whether the map should be transposed. Default is false", action="store_true")
# parser.add_argument("-s", "--simcase", default=1, help="Which simulation case to run. Default is 1", type=int)
# parser.add_argument("-r", "--random", default=False, action="store_true", help="Whether object locations should be randomly generated or not. Only affects simulation case 2")
# parser.add_argument("-c", "--testconfig", default=False, action="store_true", help="Whether the convoluted case of overlapping areas should be used")
# parser.add_argument("-p", "--ptu", default=0, type=int, help="Which Ptu to take. If 0: choose predefined set. If 1: Choose Dynamic, if 2: Choose biggest difference")
# args = parser.parse_args()

if __name__=="__main__":
    
    # The basic string
    basepath = os.path.abspath(os.path.dirname(__file__))
    fname = "setupsim.py"
    filetorun = os.path.join(basepath, fname)
    basestr = "python3 {} -d {} -f {} -a {} -s {} -p {} --overlaph {} --overlapv {}"
    
    # All the available options
    dims = [48, 64]
    simcases = [1, 2]
    fov = [1, 2, 3]
    overh = overv = [0.25, 0.5, 0.75, 1.0/3.0, 2.0/3.0]
    acc = [0.6, 0.7, 0.8, 0.9, 0.95]
    probs = [0, 1, 2]

    # The boolean options
    trans = [False, True]
    rand = [False, True]
    testconf = [False, True]

    # The setup
    noofiterations = len(dims) * len(simcases) * len(fov) * len(overh) * len(overv) * len(acc) * len(probs) *\
                    len(trans) * len(rand) * len(testconf)
    t1 = time.time()
    ctr = 0
    tenpercent = 0
    # Launch all the options, maintain the order!
    for dim in dims:
        for f in fov:
            for a in acc:
                for sim in simcases:
                    for prob in probs:
                        for oh in overh:
                            for ov in overv:
                                # Boolean options
                                for tr in trans:
                                    for ra in rand:
                                        for tc in testconf:
                                            # Creating the basic string
                                            launchstring = basestr.format(
                                                filetorun, dim, f, a, sim, prob, oh, ov
                                            )
                                            # use boolean option
                                            if tr:
                                                launchstring += " -t"
                                            if ra:
                                                launchstring += " -r"
                                            if tc:
                                                launchstring += " -c"
                                            # The actual launching of the process
                                            try:
                                                subprocess.run(launchstring)
                                            except Exception as e:
                                                print("Simulation Failed for case: {}".format(
                                                    launchstring    
                                                ))
                                                print("Exception Type: {}, File: {}, Line: {}".format(
                                                    sys.exc_info()[0], os.path.split(sys.exc_info()[2].tb_frame.f_code.co_filename)[1], sys.exc_info()[2].tb_lineno
                                                    ))

                                            # Timing and checking
                                            t2 = time.time()        
                                            diff = t2-t1
                                            ctr +=1
                                            avg_time = diff / ctr
                                            if ctr % (noofiterations / 10) == 0:
                                                tenpercent +=1
                                                print("Reached {} %, Avg time: {}, ETA: {}".format(
                                                    tenpercent * 10, avg_time, (noofiterations-ctr) * avg_time
                                                ))