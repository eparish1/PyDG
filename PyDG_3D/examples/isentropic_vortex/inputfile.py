import numpy as np
import sys
sys.path.append("../../src") #link the the source directory for PyDG
from ic_functions_premade import vortexICS  #import the IC for taylor green vortex.

## Make square grid
L = 10.                       #|  length
Nel = np.array([2**4,2**4,1])   #|  elements in x,y,z
order =np.array([4,4,1])                       #|  spatial order
quadpoints = order*2               #|  number of quadrature points. 2x the order is reccomended
mu = 0.01                       #|  viscocity
x = np.linspace(0,L,Nel[0]+1)      #|  x, y, and z
y = np.linspace(0,L,Nel[1]+1)      #|
z = np.linspace(0,L,Nel[2]+1)      #|
t = 0                              #|  simulation start time
dt = 0.005                        #|  simulation time step
et = 10.                           #|  simulation end time
save_freq = 100                      #|  frequency to save output and print to screen
eqn_str = 'Navier-Stokes'          #|  equation set
schemes = ('roe','Inviscid')             #|  inviscid and viscous flux schemes
procx = 2                         #|  processor decomposition in x
procy = 2                          #|  same in y. Note that procx*procy needs to equal total number of procs


right_bc = 'periodic'
left_bc = 'periodic'
top_bc = 'periodic'
bottom_bc = 'periodic'

T_wall = 1
right_bc_args = []
left_bc_args = []
top_bc_args = []
bottom_bc_args = []
BCs = [right_bc,right_bc_args,top_bc,top_bc_args,left_bc,left_bc_args,bottom_bc,bottom_bc_args]
				   #|
time_integration = 'ExplicitRK4'   #| 
linear_solver_str = 'GMRes'
nonlinear_solver_str = 'Newton'

#== Assign initial condition function. Note that you can alternatively define this here
#== function layout is my_ic_function(x,y,z), where x,y,z are the decomposed quadrature points
IC_function = vortexICS                #|
                                   #|
execfile('../../src/PyDG.py')      #|  call the solver
