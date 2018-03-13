import numpy as np
import sys
sys.path.append("../../src_spacetime") #link the the source directory for PyDG
from ic_functions_premade import zeroFSIC  #import the IC for taylor green vortex.
def savehook(main):
  pass
def myICs(x,y,z,gas):
  return x*0.


### Input deck to solve the poisson equation in 2D
##  \nabla \cdot u = -1 
## 16 x 16 grid at 3rd order accuarcy with steady state BICGSTAB solver
## post process the case by going to solution folder and then using
## python ../../../src_spacetime/postProcessing.py Scalar

## Make square grid
L = 1.                       #|  length
Nel = np.array([16,16,1,1])   #|  elements in x,y,z
order =np.array([3,3,1,1])                       #|  spatial order
quadpoints = order*2               #|  number of quadrature points. 2x the order is reccomended
quadpoints[-1] = order[-1]
quadpoints[-2] = order[-2]
mu = 1                       #|  viscocity
x = np.linspace(0,L,Nel[0]+1)      #|  x, y, and z
y = np.linspace(0,L,Nel[1]+1)      #|
z = np.linspace(0,L,Nel[2]+1)      #|
x,y,z = np.meshgrid(x,y,z,indexing='ij')
t = 0                              #|  simulation start time
dt = 10.                       #|  simulation time step
et = 30.                           #|  simulation end time
save_freq = 1                      #|  frequency to save output and print to screen
eqn_str = 'Diffusion'          #|  equation set
schemes = ('central','BR1')             #|  inviscid and viscous flux schemes
procx = 1                          #|  processor decomposition in x
procy = 1                          #|  same in y. Note that procx*procy needs to equal total number of procs
				   #|
time_integration = 'SteadyState'   #| 
linear_solver_str = 'BICGSTAB'
nonlinear_solver_str = 'Newton'
right_bc = 'dirichlet'
left_bc = 'dirichlet'
top_bc = 'dirichlet'
bottom_bc = 'dirichlet'
front_bc = 'periodic'
back_bc = 'periodic'
source_mag = np.array([1])
right_bc_args = [0]
left_bc_args = [0]
top_bc_args = [0]
bottom_bc_args = [0]
front_bc_args = []
back_bc_args = []
fsource = True
BCs = [right_bc,right_bc_args,top_bc,top_bc_args,left_bc,left_bc_args,bottom_bc,bottom_bc_args,front_bc,front_bc_args,back_bc,back_bc_args]

#== Assign initial condition function. Note that you can alternatively define this here
#== function layout is my_ic_function(x,y,z), where x,y,z are the decomposed quadrature points
IC_function = myICs                #|
                                   #|
execfile('../../src_spacetime/PyDG.py')      #|  call the solver
