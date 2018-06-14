import numpy as np
import sys
PyDG_DIR = '../../src_spacetime'
sys.path.append(PyDG_DIR) #link the the source directory for PyDG
from ic_functions_premade import TGVIC  #import the IC for taylor green vortex.
def savehook(regionManager):
  pass
## Make square grid
L = 2.*np.pi                       #|  length
Nel = np.array([4,4,4,1])   #|  elements in x,y,z
order = np.array([4,4,4,1])                         #|  spatial order
quadpoints = order               #|  number of quadrature points. 2x the order is reccomended
quadpoints[-1] = order[-1]
mu = 1./1600.                       #|  viscocity
x = np.linspace(0,L,Nel[0]+1)      #|  x, y, and z
y = np.linspace(0,L,Nel[1]+1)      #|
z = np.linspace(0,L,Nel[2]+1)      #|
x,y,z = np.meshgrid(x,y,z,indexing='ij')
t = 0                              #|  simulation start time
dt = 0.025                        #|  simulation time step
et = 100.                           #|  simulation end time
save_freq = 1                      #|  frequency to save output and print to screen
eqn_str = 'Navier-Stokes'          #|  equation set
schemes = ('roe','BR1')             #|  inviscid and viscous flux schemes
procx = 4                          #|  processor decomposition in x
procy = 1                          #|  same in y. Note that procx*procy needs to equal total number of procs
				   #|
#time_integration = 'BackwardEuler'   #| 
time_integration = 'BackwardEuler'   #| 
linear_solver_str = 'GMRes'
nonlinear_solver_str = 'Newton'
orthogonal_str = 'True'           #| Tell PyDG that the grid is orthogonal to speed calculations

right_bc = 'periodic'
left_bc = 'periodic'
top_bc = 'periodic'
bottom_bc = 'periodic'
front_bc = 'periodic'
back_bc = 'periodic'

right_bc_args = [0,0,0,0]
left_bc_args = [0,-1,0,0]
top_bc_args = [0,0,0,0]
bottom_bc_args = [0,-1,0,0]
front_bc_args = [0,0,0,0]
back_bc_args = [0,-1,0,0]

BCs = [left_bc,left_bc_args,right_bc,right_bc_args,bottom_bc,bottom_bc_args,top_bc,top_bc_args,back_bc,back_bc_args,front_bc,front_bc_args]


#== Assign initial condition function. Note that you can alternatively define this here
#== function layout is my_ic_function(x,y,z), where x,y,z are the decomposed quadrature points
IC_function = TGVIC                #|
                                   #|
execfile(PyDG_DIR + '/PyDG.py')      #|  call the solver
