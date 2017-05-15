import numpy as np
import sys
sys.path.append("../../src") #link the the source directory for PyDG
from ic_functions_premade import shocktubeIC  #import the IC for taylor green vortex.

## Make square grid
L = 1.                       #|  length
Nel = np.array([128,1,1])   #|  elements in x,y,z
order =np.array([3,1,1])                       #|  spatial order
quadpoints = order*2               #|  number of quadrature points. 2x the order is reccomended
mu = 5e-5                       #|  viscocity
x = np.linspace(0,L,Nel[0]+1)      #|  x, y, and z
y = np.linspace(0,L,Nel[1]+1)      #|
z = np.linspace(0,L,Nel[2]+1)      #|
t = 0                              #|  simulation start time
dt = 0.001                        #|  simulation time step
et = 200.                           #|  simulation end time
save_freq = 1                      #|  frequency to save output and print to screen
eqn_str = 'Navier-Stokes'          #|  equation set
schemes = ('rusanov','IP')             #|  inviscid and viscous flux schemes
procx = 1                          #|  processor decomposition in x
procy = 1                          #|  same in y. Note that procx*procy needs to equal total number of procs
				   #|
time_integration = 'SDIRK2'   #| 
linear_solver_str = 'GMRes'
nonlinear_solver_str = 'Newton'

right_bc = 'adiabatic_wall'
left_bc = 'adiabatic_wall'
top_bc = 'periodic'
bottom_bc = 'periodic'

top_bc_args = [0.,0.,0.,1.]
bottom_bc_args = [0.,0.,0.,1.]
right_bc_args = [0.,0,0,1.]
left_bc_args = [0.,0,0,1.]
BCs = [right_bc,right_bc_args,top_bc,top_bc_args,left_bc,left_bc_args,bottom_bc,bottom_bc_args]
#== Assign initial condition function. Note that you can alternatively define this here
#== function layout is my_ic_function(x,y,z), where x,y,z are the decomposed quadrature points
IC_function = shocktubeIC                #|
                                   #|
execfile('../../src/PyDG.py')      #|  call the solver
