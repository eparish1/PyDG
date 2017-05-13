import numpy as np
import sys
sys.path.append("../../src") #link the the source directory for PyDG
from ic_functions_premade import TGVIC  #import the IC for taylor green vortex.

## Make square grid
L = 2.*np.pi                       #|  length
Nel = np.array([2**0,2**0,2**0])   #|  elements in x,y,z
order = 8                         #|  spatial order
quadpoints = order*2               #|  number of quadrature points. 2x the order is reccomended
mu = 1./400.                       #|  viscocity
x = np.linspace(0,L,Nel[0]+1)      #|  x, y, and z
y = np.linspace(0,L,Nel[1]+1)      #|
z = np.linspace(0,L,Nel[2]+1)      #|
t = 0                              #|  simulation start time
dt =  0.02                         #|  simulation time step
et = 10.                           #|  simulation end time
save_freq = 1                      #|  frequency to save output and print to screen
eqn_str = 'Navier-Stokes'          #|  equation set
schemes = ('roe','IP')             #|  inviscid and viscous flux schemes
procx = 1                          #|  processor decomposition in x
procy = 1                          #|  same in y. Note that procx*procy needs to equal total number of procs
				   #|
time_integration = 'CrankNicolson'   #| 
linear_solver_str = 'GMRes'
nonlinear_solver_str = 'Newton_MG'

#== Assign initial condition function. Note that you can alternatively define this here
#== function layout is my_ic_function(x,y,z), where x,y,z are the decomposed quadrature points
IC_function = TGVIC                #|
                                   #|
execfile('../../src/PyDG.py')      #|  call the solver
