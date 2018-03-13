import numpy as np
import sys
sys.path.append("../../src_spacetime") #link the the source directory for PyDG
from ic_functions_premade import shocktubeIC  #import the IC for taylor green vortex.
def savehook(main):
  # compute entropy
  p = (1.4 - 1.)*(main.a.u[4] - 0.5*main.a.u[1]**2/main.a.u[0] - 0.5*main.a.u[2]**2/main.a.u[0] - 0.5*main.a.u[3]**2/main.a.u[0])
  shp = np.append(6,np.shape(main.a.u[0]))
  tmp = np.zeros( shp )
  tmp[0:5] = main.a.u
  tmp[-1]  = ( ( np.log(p) - 1.4*np.log(main.a.u[0]) )*main.a.u[0]/(1.4 - 1.)*-1.)
  vol_integral =  main.basis.volIntegrate(main.weights0,main.weights1,main.weights2,main.weights3,tmp*main.Jdet[None,:,:,:,None,:,:,:,None])
  s_int = globalSum(vol_integral[-1] , main)
  rho_int = globalSum(vol_integral[0] , main)
  if (main.mpi_rank == 0):
      sys.stdout.write('Mass = ' + str(rho_int) +  '  Entropy = ' + str(s_int) +  '\n')
      sys.stdout.flush()

## Make square grid
L = 1.                       #|  length
Nel = np.array([100,1,1,1])   #|  elements in x,y,z
order =np.array([1,1,1,1])                       #|  spatial order
quadpoints = order*2               #|  number of quadrature points. 2x the order is reccomended
quadpoints[2::] = order[2::]
x = np.linspace(0,L,Nel[0]+1)      #|  x, y, and z
y = np.linspace(0,L,Nel[1]+1)      #|
z = np.linspace(0,L,Nel[2]+1)      #|
x,y,z = np.meshgrid(x,y,z,indexing='ij')
mu = 0.
t = 0                              #|  simulation start time
dt = 0.000125                       #|  simulation time step
et = 0.2                           #|  simulation end time
save_freq = 10                      #|  frequency to save output and print to screen
eqn_str = 'Navier-Stokes'          #|  equation set
schemes = ('roe','Inviscid')             #|  inviscid and viscous flux schemes
procx = 1                          #|  processor decomposition in x
procy = 1                          #|  same in y. Note that procx*procy needs to equal total number of procs
time_integration = 'SSP_RK3'   #| 

right_bc = 'reflecting_wall'
left_bc = 'reflecting_wall'
top_bc = 'periodic'
bottom_bc = 'periodic'
front_bc = 'periodic'
back_bc = 'periodic'

top_bc_args = [0.,0.,0.]
bottom_bc_args = [0.,0.,0.]
right_bc_args = [0.,0,0]
left_bc_args = [0.,0,0]
front_bc_args = []
back_bc_args = []

BCs = [right_bc,right_bc_args,top_bc,top_bc_args,left_bc,left_bc_args,bottom_bc,bottom_bc_args,front_bc,front_bc_args,back_bc,back_bc_args]
#== Assign initial condition function. Note that you can alternatively define this here
#== function layout is my_ic_function(x,y,z), where x,y,z are the decomposed quadrature points
IC_function = shocktubeIC                #|
                                   #|
execfile('../../src_spacetime/PyDG.py')      #|  call the solver
