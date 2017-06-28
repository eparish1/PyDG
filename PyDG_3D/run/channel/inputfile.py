import numpy as np
import sys
sys.path.append("../../src_spacetime") #link the the source directory for PyDG
from ic_functions_premade import zeroFSIC  #import the IC for taylor green vortex.

def zeroFSIC(x,y,z,gas):
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  q = np.zeros((5,nqx,nqy,nqz,Nelx,Nely,Nelz))
  gamma = gas.gamma
  Cv = gas.Cv
  Cp = Cv*gamma
  R = Cp - Cv
  T = np.zeros((nqx,nqy,nqz,Nelx,Nely,Nelz))
  rho = np.zeros(np.shape(T))
  #u = np.zeros(np.shape(rho))
  u = 1./mu*pbar_x*(1. - y**2)/2.
  v = np.zeros(np.shape(u))
  w = np.zeros(np.shape(u))
  E = np.zeros(np.shape(u))
  T[:] = 273.#1./gamma
  rho[:] = 1.#T**(1./(gamma - 1.))
  E[:] = Cv*T + 0.5*(u**2 + v**2)
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*w
  q[4] = rho*E
  return q




## Make square grid
Minf = 0.2
L = 2.
Lx,Ly,Lz = 4.*np.pi,2,4.*np.pi/3.                       #|  length
Nel = np.array([1,16,1,1])   #|  elements in x,y,z
order =np.array([1,4,1,1])                       #|  spatial order
quadpoints = order*2               #|  number of quadrature points. 2x the order is reccomended
quadpoints[-1] = order[-1]
mu = 0.01                       #|  viscocity
x = np.linspace(0,Lx,Nel[0]+1)      #|  x, y, and z
y = -np.cos( np.pi*np.linspace(0,Nel[1]+1 - 1 ,Nel[1] + 1) /(Nel[1] + 1 -1) )*Ly/2. + Ly/2. - 1
#y = np.linspace(-L/2.,L/2.,Nel[1]+1)      #|
z = np.linspace(0,Lz,Nel[2]+1)      #|
t = 0                              #|  simulation start time
dt = 0.00025                        #|  simulation time step
et = 200.                           #|  simulation end time
save_freq = 100                      #|  frequency to save output and print to screen
eqn_str = 'Navier-Stokes'          #|  equation set
schemes = ('roe','IP')             #|  inviscid and viscous flux schemes
procx = 1                          #|  processor decomposition in x
procy = 1                          #|  same in y. Note that procx*procy needs to equal total number of procs
                                   #|
time_integration = 'SteadyState'
#time_integration = 'CrankNicolson'   #| 
#time_integration = 'ExplicitRK4'
linear_solver_str = 'GMRes'
nonlinear_solver_str = 'Newton'

right_bc = 'periodic'
left_bc = 'periodic'
top_bc = 'isothermal_wall'
bottom_bc = 'isothermal_wall'

T_wall = 273.
right_bc_args = []
left_bc_args = []
top_bc_args = [0.,0,0,T_wall]
bottom_bc_args = [0.,0,0,T_wall]
BCs = [right_bc,right_bc_args,top_bc,top_bc_args,left_bc,left_bc_args,bottom_bc,bottom_bc_args]
source = True
Re_tau = 18
pbar_x = Re_tau**2*mu**2
source_mag = np.array([0,pbar_x,0,0,0])

#== Assign initial condition function. Note that you can alternatively define this here
#== function layout is my_ic_function(x,y,z), where x,y,z are the decomposed quadrature points
IC_function = zeroFSIC                #|
                                   #|
execfile('../../src_spacetime/PyDG.py')      #|  call the solver

