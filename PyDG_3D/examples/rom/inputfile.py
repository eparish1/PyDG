import numpy as np
import sys
PyDG_DIR = '../../../src_spacetime'
sys.path.append("../../../src_spacetime") #link the the source directory for PyDG
from ic_functions_premade import shocktubeIC  #import the IC for taylor green vortex.

def savehook(regionManager):
  pass
#  print(np.linalg.norm(regionManager.a))

## Make square grid
L = 1.                       #|  length
Nel = np.array([500,1,1,1])   #|  elements in x,y,z
order =np.array([2,1,1,1])                       #|  spatial order
quadpoints = order*2               #|  number of quadrature points. 2x the order is reccomended
#quadpoints[2::] = order[2::]
x = np.linspace(0,L,Nel[0]+1)      #|  x, y, and z
y = np.linspace(0,L,Nel[1]+1)      #|
z = np.linspace(0,L,Nel[2]+1)      #|
x,y,z = np.meshgrid(x,y,z,indexing='ij')
mu = 0.
t = 0                              #|  simulation start time
dt = 0.001                       #|  simulation time step
et = 1.                          #|  simulation end time
save_freq = 1                      #|  frequency to save output and print to screen
eqn_str = 'Navier-Stokes'          #|  equation set
schemes = ('rusanov','Inviscid')             #|  inviscid and viscous flux schemes
procx = 1                         #|  processor decomposition in x
procy = 1                          #|  same in y. Note that procx*procy needs to equal total number of procs
fsource = None
source_mag = None				   #|
#time_integration = 'crankNicolson_LSPG_windowed'   #| 
time_integration = 'SSP_RK3'   #| 
#time_integration = 'CrankNicolson'
linear_solver_str = 'GMRes'
nonlinear_solver_str = 'Newton'
mol_str = None
#turb_str = 'orthogonal subscale POD'
right_bc = 'reflecting_wall'
left_bc = 'reflecting_wall'
top_bc = 'periodic'
bottom_bc = 'periodic'
front_bc = 'periodic'
back_bc = 'periodic'

right_bc_args = [0.,0,0]
left_bc_args = [0.,0,0]
top_bc_args = [0,0,0,0]
bottom_bc_args = [0,-1,0,0]
front_bc_args = [0,0,0,0]
back_bc_args = [0,-1,0,0]

#BCs = [right_bc,right_bc_args,top_bc,top_bc_args,left_bc,left_bc_args,bottom_bc,bottom_bc_args,front_bc,front_bc_args,back_bc,back_bc_args]
BCs = [left_bc,left_bc_args,right_bc,right_bc_args,bottom_bc,bottom_bc_args,top_bc,top_bc_args,back_bc,back_bc_args,front_bc,front_bc_args]

#== Assign initial condition function. Note that you can alternatively define this here
#== function layout is my_ic_function(x,y,z), where x,y,z are the decomposed quadrature points
IC_function = shocktubeIC                #|
write_output = 1                                   #|
execfile('../../../src_spacetime/PyDG.py')      #|  call the solver
