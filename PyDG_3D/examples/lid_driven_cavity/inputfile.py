import numpy as np
import sys
sys.path.append("../../src_spacetime") #link the the source directory for PyDG
from ic_functions_premade import zeroFSIC  #import the IC for taylor green vortex.
def savehook(main):
  pass

def customFSIC(x,y,z,main):
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  q = np.zeros((5,nqx,nqy,nqz,Nelx,Nely,Nelz))
  gamma = main.gas.gamma
  Cv = main.gas.Cv
  Cp = Cv*gamma
  R = Cp - Cv
  T = np.zeros((nqx,nqy,nqz,Nelx,Nely,Nelz))
  rho = np.zeros(np.shape(T))
  u = np.zeros(np.shape(rho))
  v = np.zeros(np.shape(u))
  w = np.zeros(np.shape(u))
  E = np.zeros(np.shape(u))
  rho[:] = 1.
  p = 1.
  E[:] = p/(1.4 - 1.)
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*w
  q[4] = rho*E
  return q


## Make square grid
L = 1.                       #|  length
Nel = np.array([20,20,1,1])   #|  elements in x,y,z
order =np.array([3,3,1,1])                       #|  spatial order
quadpoints = order*2
quadpoints[-2::] = order[-2::]               #|  number of quadrature points. 2x the order is reccomended
mu = 0.001                       #|  viscocity
x = np.linspace(0,L,Nel[0]+1)      #|  x, y, and z
y = np.linspace(0,L,Nel[1]+1)      #|
z = np.linspace(0,L,Nel[2]+1)      #|
x,y,z = np.meshgrid(x,y,z,indexing='ij')

h = float(1./Nel[0]) 
perturb_x = 0.2*h*np.sin((x + 6.)**(y + 6.)) 
perturb_y = 0.2*h*np.cos((y + 6.)**(x + 6.))
x[1:-1,1:-1,:] += perturb_x[1:-1,1:-1,:]
y[1:-1,1:-1,:] += perturb_y[1:-1,1:-1,:]

t = 0                              #|  simulation start time
dt = 0.004                        #|  simulation time step
et = 300.                           #|  simulation end time
save_freq = 1000                      #|  frequency to save output and print to screen
eqn_str = 'Navier-Stokes'          #|  equation set
schemes = ('roe','BR1')             #|  inviscid and viscous flux schemes
procx = 2                          #|  processor decomposition in x
procy = 2                          #|  same in y. Note that procx*procy needs to equal total number of procs
				   #|
time_integration = 'SSP_RK3'   #| 
linear_solver_str = 'GMRes'
nonlinear_solver_str = 'Newton'

right_bc = 'isothermal_wall'
left_bc = 'isothermal_wall'
top_bc = 'isothermal_wall'
bottom_bc = 'isothermal_wall'
front_bc = 'periodic'
back_bc = 'periodic'
#orthogonal_str = 'True'
T_wall = 1.
right_bc_args = [0,0,0,T_wall]
left_bc_args = [0,0,0,T_wall]
top_bc_args = [0.1,0,0,T_wall]
bottom_bc_args = [0,0,0,T_wall]
front_bc_args = []
back_bc_args = []
BCs = [right_bc,right_bc_args,top_bc,top_bc_args,left_bc,left_bc_args,bottom_bc,bottom_bc_args,front_bc,front_bc_args,back_bc,back_bc_args]
#== Assign initial condition function. Note that you can alternatively define this here
#== function layout is my_ic_function(x,y,z), where x,y,z are the decomposed quadrature points
IC_function = customFSIC                #|
                                   #|
execfile('../../src_spacetime/PyDG.py')      #|  call the solver
