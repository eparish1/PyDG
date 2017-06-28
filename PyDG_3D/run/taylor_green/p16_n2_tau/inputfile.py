import numpy as np
import sys
sys.path.append("../../../src_spacetime")
from ic_functions_premade import TGVIC  #import the IC for taylor green vortex.

## Make square grid
L = 2.*np.pi                       #|  length
Nel = np.array([2,2,2,1])   #|  elements in x,y,z
order = np.array([16,16,16,1])                         #|  spatial order
quadpoints = order*2               #|  number of quadrature points. 2x the order is reccomended
quadpoints[-1] = order[-1]
Minf = 0.2
mu = 1./1600.*Minf                       #|  viscocity
x = np.linspace(0,L,Nel[0]+1)      #|  x, y, and z
y = np.linspace(0,L,Nel[1]+1)      #|
z = np.linspace(0,L,Nel[2]+1)      #|
t = 0                              #|  simulation start time
dt = 0.01                        #|  simulation time step
et = 100.                           #|  simulation end time
save_freq = 50                      #|  frequency to save output and print to screen
eqn_str = 'Navier-Stokes'          #|  equation set
schemes = ('central','IP')             #|  inviscid and viscous flux schemes
enriched_eqn_str = 'Navier-Stokes'
enriched_schemes = ('central','Inviscid')   #|  inviscid and viscous flux schemes
procx = 2                          #|  processor decomposition in x
procy = 2                          #|  same in y. Note that procx*procy needs to equal total number of procs
                                   #|
enriched = 'True'
time_integration = 'ExplicitRK4'
#time_integration = 'CrankNicolson'   #| 
turb_str = 'tau-model'
linear_solver_str = 'GMRes'
nonlinear_solver_str = 'Newton_MG'

right_bc = 'periodic'
left_bc = 'periodic'
top_bc = 'periodic'
bottom_bc = 'periodic'

right_bc_args = []
left_bc_args = []
top_bc_args = []
bottom_bc_args = []
BCs = [right_bc,right_bc_args,top_bc,top_bc_args,left_bc,left_bc_args,bottom_bc,bottom_bc_args]

#== Assign initial condition function. Note that you can alternatively define this here
#== function layout is my_ic_function(x,y,z), where x,y,z are the decomposed quadrature points
IC_function = TGVIC                #|
                                   #|
execfile('../../../src_spacetime/PyDG.py')      #|  call the solver

