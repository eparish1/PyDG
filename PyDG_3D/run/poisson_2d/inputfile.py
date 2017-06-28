import numpy as np
import sys
sys.path.append("../../src_spacetime") #link the the source directory for PyDG
from ic_functions_premade import vortexICS  #import the IC for taylor green vortex.

def diffusionIC(x,y,z,gas):
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  u = np.zeros((1,nqx,nqy,nqz,Nelx,Nely,Nelz))
  u[:] = 0
  return u 

## Make square grid
L = 1.                       #|  length
Nel = np.array([2**1,2**1,1,1])   #|  elements in x,y,z
order =np.array([64,64,1,1])                       #|  spatial order
quadpoints = order*2               #|  number of quadrature points. 2x the order is reccomended
quadpoints[-1] = order[-1]
mu = 1                       #|  viscocity
x = np.linspace(0,L,Nel[0]+1)      #|  x, y, and z
y = np.linspace(0,L,Nel[1]+1)      #|
z = np.linspace(0,L,Nel[2]+1)      #|
t = 0                              #|  simulation start time
dt = 1                        #|  simulation time step
et = 1.                           #|  simulation end time
save_freq = 1                      #|  frequency to save output and print to screen
eqn_str = 'Diffusion'          #|  equation set
schemes = ('central','IP')             #|  inviscid and viscous flux schemes
procx = 2                         #|  processor decomposition in x
procy = 2                          #|  same in y. Note that procx*procy needs to equal total number of procs


right_bc = 'dirichlet'
left_bc = 'dirichlet'
top_bc = 'dirichlet'
bottom_bc = 'dirichlet'

right_bc_args = [0]
left_bc_args = [0]
top_bc_args = [0]
bottom_bc_args = [0]
source = True
source_mag = np.array([1.0])

BCs = [right_bc,right_bc_args,top_bc,top_bc_args,left_bc,left_bc_args,bottom_bc,bottom_bc_args]
				   #|
time_integration = 'SteadyState'   #| 
linear_solver_str = 'GMRes'
nonlinear_solver_str = 'Newton_MG'

#== Assign initial condition function. Note that you can alternatively define this here
#== function layout is my_ic_function(x,y,z), where x,y,z are the decomposed quadrature points
IC_function = diffusionIC                #|
                                   #|
execfile('../../src_spacetime/PyDG.py')      #|  call the solver
