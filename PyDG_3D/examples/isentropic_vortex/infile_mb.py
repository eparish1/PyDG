import numpy as np
import sys
PyDG_DIR = '../../src_spacetime'
sys.path.append(PyDG_DIR) #link the the source directory for PyDG
from ic_functions_premade import vortexICS,zeroFSIC  #import the IC for taylor green vortex.
def savehook(regionManager):
  ### get a0
  region_counter = 0
  rho_int = 0.
  s_int = 0.
  for j in regionManager.mpi_regions_owned:
    main = regionManager.region[region_counter]
    region_counter += 1#
    # compute entropy
    p = (1.4 - 1.)*(main.a.u[4] - 0.5*main.a.u[1]**2/main.a.u[0] - 0.5*main.a.u[2]**2/main.a.u[0] - 0.5*main.a.u[3]**2/main.a.u[0])
    shp = np.append(6,np.shape(main.a.u[0]))
    tmp = np.zeros( shp )
    tmp[0:5] = main.a.u
    tmp[-1]  = ( ( np.log(p) - 1.4*np.log(main.a.u[0]) )*main.a.u[0]/(1.4 - 1.)*-1.)
    vol_integral =  main.basis.volIntegrate(main.weights0,main.weights1,main.weights2,main.weights3,tmp*main.Jdet[None,:,:,:,None,:,:,:,None])
    s_int += globalSum(vol_integral[-1] , main)
    rho_int += globalSum(vol_integral[0] , main)
  if (main.mpi_rank == 0):
      sys.stdout.write('Mass = ' + str(rho_int) +  '  Entropy = ' + str(s_int) +  '\n')
      sys.stdout.flush()

## Make square grid
L = 10.                       #|  length
n_blocks = 4
Nel_block1 = np.array([4,4,2,1])   #|  elements in x,y,z
Nel_block2 = np.array([4,4,2,1])   #|  elements in x,y,z
Nel_block3 = np.array([4,4,2,1])   #|  elements in x,y,z
Nel_block4 = np.array([4,4,2,1])   #|  elements in x,y,z

order =np.array([4,4,1,1])                       #|  spatial order
quadpoints = np.array([order[0],order[1],order[2],order[3] ])  #|  number of quadrature points. 2x the order is reccomended

mu = 0.01#5                       #|  viscocity
x = np.linspace(0,L/2.,Nel_block1[0]+1)      #|  x, y, and z
y = np.linspace(0,L/2.,Nel_block1[1]+1)      #|
z = np.linspace(0,L,Nel_block1[2]+1)      #|
x_block0,y_block0,z_block0 = np.meshgrid(x,y,z,indexing='ij')

x = np.linspace(L/2.,L,Nel_block4[0]+1)      #|  x, y, and z
y = np.linspace(L/2.,L,Nel_block4[1]+1)      #|
z = np.linspace(0,L,Nel_block4[2]+1)      #|
x_block1,y_block1,z_block1 = np.meshgrid(x,y,z,indexing='ij')


x = np.linspace(L/2.,L,Nel_block2[0]+1)      #|  x, y, and z
y = np.linspace(0,L/2.,Nel_block2[1]+1)      #|
z = np.linspace(0,L,Nel_block2[2]+1)      #|
x_block2,y_block2,z_block2 = np.meshgrid(x,y,z,indexing='ij')


x = np.linspace(0,L/2.,Nel_block3[0]+1)      #|  x, y, and z
y = np.linspace(L/2.,L,Nel_block3[1]+1)      #|
z = np.linspace(0,L,Nel_block3[2]+1)      #|
x_block3,y_block3,z_block3 = np.meshgrid(x,y,z,indexing='ij')



h = float(10./Nel_block1[0])
Nel_block = [Nel_block1,Nel_block2,Nel_block3,Nel_block4]
x_block = [x_block0,x_block1,x_block2,x_block3]
y_block = [y_block0,y_block1,y_block2,y_block3]
z_block = [z_block0,z_block1,z_block2,z_block3]

t = 0                              #|  simulation start time
dt = 0.025                       #|  simulation time step
et = 10.                           #|  simulation end time
save_freq = 5                      #|  frequency to save output and print to screen
eqn_str = 'Navier-Stokes'          #|  equation set
schemes = ('roe','Inviscid')             #|  inviscid and viscous flux schemes
basis_functions_str = 'TensorDot'
orthogonal_str = True

#=========== MPI information ==========
procx_block0 = 1
procy_block0 = 1
procz_block0 = 1

procx_block1 = 1
procy_block1 = 1
procz_block1 = 1

procx_block2 = 1
procy_block2 = 1
procz_block2 = 1

procx_block3 = 1
procy_block3 = 1
procz_block3 = 1

starting_rank0 = 0
starting_rank1 = 0
starting_rank2 = 0 
starting_rank3 = 0

procx = [procx_block0,procx_block1,procx_block2,procx_block3]
procy = [procy_block0,procy_block1,procy_block2,procy_block3]
procz = [procz_block0,procz_block1,procz_block2,procz_block3]
starting_rank = [starting_rank0,starting_rank1,starting_rank2,starting_rank3]
#=================
right_bc = 'patch'
left_bc = 'patch'
top_bc = 'patch'
bottom_bc = 'patch'
front_bc = 'periodic'
back_bc = 'periodic'
right_bc_args = [2,0,0,0]
left_bc_args = [2,-1,0,0]
top_bc_args = [3,0,0,0]
bottom_bc_args = [3,-1,0,0]
front_bc_args = [0,0,0,0]
back_bc_args = [0,-1,0,0]
BCs_block0 = [left_bc,left_bc_args,right_bc,right_bc_args,bottom_bc,bottom_bc_args,top_bc,top_bc_args,back_bc,back_bc_args,front_bc,front_bc_args]
#==================
#=================
right_bc = 'patch'
left_bc = 'patch'
top_bc = 'patch'
bottom_bc = 'patch'
front_bc = 'periodic'
back_bc = 'periodic'
right_bc_args = [3,0,0,0]
left_bc_args = [3,-1,0,0]
top_bc_args = [2,0,0,0]
bottom_bc_args = [2,-1,0,0]
front_bc_args = [1,0,0,0]
back_bc_args = [1,-1,0,0]
BCs_block1 = [left_bc,left_bc_args,right_bc,right_bc_args,bottom_bc,bottom_bc_args,top_bc,top_bc_args,back_bc,back_bc_args,front_bc,front_bc_args]
#==================
#=================
right_bc = 'patch'
left_bc = 'patch'
top_bc = 'patch'
bottom_bc = 'patch'
front_bc = 'periodic'
back_bc = 'periodic'
right_bc_args = [0,0,0,0]
left_bc_args = [0,-1,0,0]
top_bc_args = [1,0,0,0]
bottom_bc_args = [1,-1,0,0]
front_bc_args = [2,0,0,0]
back_bc_args = [2,-1,0,0]
BCs_block2 = [left_bc,left_bc_args,right_bc,right_bc_args,bottom_bc,bottom_bc_args,top_bc,top_bc_args,back_bc,back_bc_args,front_bc,front_bc_args]
#==================
#=================
right_bc = 'patch'
left_bc = 'patch'
top_bc = 'patch'
bottom_bc = 'patch'
front_bc = 'periodic'
back_bc = 'periodic'
right_bc_args = [1,0,0,0]
left_bc_args = [1,-1,0,0]
top_bc_args = [0,0,0,0]
bottom_bc_args = [0,-1,0,0]
front_bc_args = [3,0,0,0]
back_bc_args = [3,-1,0,0]
BCs_block3 = [left_bc,left_bc_args,right_bc,right_bc_args,bottom_bc,bottom_bc_args,top_bc,top_bc_args,back_bc,back_bc_args,front_bc,front_bc_args]
#==================

BCs = [BCs_block0,BCs_block1,BCs_block2,BCs_block3]
source_mag = False
#mol_str = False				   #|
#turb_str = 'orthogonal subscale'
#time_integration = 'SpaceTime'   #| 
#time_integration = 'CrankNicolson'   #| 
time_integration = 'SSP_RK3'
linear_solver_str = 'GMRes'
nonlinear_solver_str = 'Newton'

#== Assign initial condition function. Note that you can alternatively define this here
#== function layout is my_ic_function(x,y,z), where x,y,z are the decomposed quadrature points
IC_function = [vortexICS,vortexICS,vortexICS,vortexICS]                #|
#IC_function = zeroFSIC                                   #|
execfile('../../src_spacetime/PyDG.py')      #|  call the solver


correct = 174.914240014
if (main.mpi_rank == 0):
  sol_norm = np.linalg.norm(regionManager.uG_norm)
  if (np.abs(sol_norm - correct) >= 1e-7):
    print('Error in solution, check source code')
  else:
    print('Succesful run!')
