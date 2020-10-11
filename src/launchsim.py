#!/usr/bin/env python3

import time
import sys
import os.path

if __name__=="__main__":
    
    launchstring = ""
    try:
        #  Run the launchstring here
        pass        
    except Exception as e:
        print("Exception Type: {}, File: {}, Line: {}".format(
            sys.exc_info()[0], os.path.split(sys.exc_info()[2].tb_frame.f_code.co_filename)[1], sys.exc_info()[2].tb_lineno
            ))
