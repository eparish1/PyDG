import numpy as np
import sys
PyDG_DIR = '../../../src_spacetime/'
sys.path.append(PyDG_DIR)
sys.path.append(PyDG_DIR + '/romFunctions/')

from make_pod_basis import makePodBases

## Input information
gridPath = '../'
solPath = '../Solution/'
tolerance = 1. - 1e-7
startingStep = 0
endingStep = 

