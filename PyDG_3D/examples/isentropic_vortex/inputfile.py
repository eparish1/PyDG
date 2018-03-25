import numpy as np
import sys
sys.path.append("../../src_spacetime") #link the the source directory for PyDG
from ic_functions_premade import vortexICS  #import the IC for taylor green vortex.
def savehook(main):
  pass
## Make square grid
L = 10.                       #|  length
Nel = np.array([8,8,2,1])   #|  elements in x,y,z
order =np.array([4,4,1,1])                       #|  spatial order
quadpoints = np.array([order[0],order[1],order[2],order[3] ])  #|  number of quadrature points. 2x the order is reccomended
mu = 0.#5                       #|  viscocity
#x = -np.cos( np.pi*np.linspace(0,Nel[0]+1 - 1 ,Nel[0] + 1) /(Nel[0] + 1 -1) )*L/2. + L/2.
#y = -np.cos( np.pi*np.linspace(0,Nel[1]+1 - 1 ,Nel[1] + 1) /(Nel[1] + 1 -1) )*L/2. + L/2.
#z = -np.cos( np.pi*np.linspace(0,Nel[2]+1 - 1 ,Nel[2] + 1) /(Nel[2] + 1 -1) )*L/2. + L/2.
x = np.linspace(0,L,Nel[0]+1)      #|  x, y, and z
y = np.linspace(0,L,Nel[1]+1)      #|
z = np.linspace(0,L,Nel[2]+1)      #|
x,y,z = np.meshgrid(x,y,z,indexing='ij')
t = 0                              #|  simulation start time
dt = 0.025
et = 10.                           #|  simulation end time
save_freq = 10.                      #|  frequency to save output and print to screen
eqn_str = 'Navier-Stokes'          #|  equation set
schemes = ('roe','Inviscid')             #|  inviscid and viscous flux schemes
procx = 1                         #|  processor decomposition in x
procy = 1                          #|  same in y. Note that procx*procy needs to equal total number of procs
procz = 1

right_bc = 'periodic'
left_bc = 'periodic'
top_bc = 'periodic'
bottom_bc = 'periodic'
front_bc = 'periodic'
back_bc = 'periodic'

right_bc_args = []
left_bc_args = []
top_bc_args = []
bottom_bc_args = []
front_bc_args = []
back_bc_args = []

BCs = [right_bc,right_bc_args,top_bc,top_bc_args,left_bc,left_bc_args,bottom_bc,bottom_bc_args,front_bc,front_bc_args,back_bc,back_bc_args]
source_mag= False
mol_str = False				   #|
time_integration = 'SSP_RK3'
linear_solver_str = 'GMRes'
nonlinear_solver_str = 'Newton'

#== Assign initial condition function. Note that you can alternatively define this here
#== function layout is my_ic_function(x,y,z), where x,y,z are the decomposed quadrature points
IC_function = vortexICS                #|
                                   #|
execfile('../../src_spacetime/PyDG.py')      #|  call the solver
correct = 174.914240014
if (main.mpi_rank == 0):
  sol_norm = np.linalg.norm(uG)
  if (np.abs(sol_norm - correct) >= 1e-7):
    print('Error in solution, check source code')
  else:
    print('Succesful run!')

