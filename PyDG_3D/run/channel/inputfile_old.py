import numpy as np
import sys
sys.path.append("../../src_spacetime") #link the the source directory for PyDG
from scipy.interpolate import RegularGridInterpolator

def TGVIC_custom(x,y,z,gas):
  Lx = 2.*np.pi
  Ly = 2.*np.pi
  Lz = 2.*np.pi
  Minf = 0.2
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  gamma = gas.gamma
  T0 = 1./gamma
  R = gas.R #1
  rho = 1.
  p0 = rho*R*T0
  a = np.sqrt(gamma*R*T0)
  V0 = Minf*a
  Cv = 5./2.*R
  u = V0*np.sin(x*2.*np.pi/Lx)*np.cos(y*2.*np.pi/Ly)*np.cos(z*2.*np.pi/Lz)
  v = -V0*np.cos(x*2.*np.pi/Lx)*np.sin(y*2.*np.pi/Ly)*np.cos(z*2.*np.pi/Lz)
  w = 0
  p = p0 + rho*V0**2/16.*(np.cos(2.*x*2.*np.pi/Lx) + np.cos(2.*y*2.*np.pi/Ly) )*(np.cos(2.*z*2.*np.pi/Lz) + 2.)
  T = p/(rho*R)
  E = Cv*T + 0.5*(u**2 + v**2 + w**2)
  q = np.zeros((5,nqx,nqy,nqz,Nelx,Nely,Nelz))
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*w
  q[4] = rho*E
  return q

def source_hook(main,f):
  f[1] = 0.01

## Make square grid
L = 2.*np.pi                       #|  length
Nel = np.array([8,8,8,1])   #|  elements in x,y,z
order =np.array([1,1,1,1])                       #|  spatial order
quadpoints = order*2               #|  number of quadrature points. 2x the order is reccomended
mu = 0.01                       #|  viscocity
x = np.linspace(0,L,Nel[0]+1)      #|  x, y, and z
y = np.linspace(-L/2.,L/2.,Nel[1]+1)      #|
z = np.linspace(0,L,Nel[2]+1)      #|
t = 0                              #|  simulation start time
dt = 0.5                        #|  simulation time step
et = 200.                           #|  simulation end time
save_freq = 100                      #|  frequency to save output and print to screen
eqn_str = 'Navier-Stokes'          #|  equation set
schemes = ('roe','IP')             #|  inviscid and viscous flux schemes
procx = 1                          #|  processor decomposition in x
procy = 1                          #|  same in y. Note that procx*procy needs to equal total number of procs
				   #|
time_integration = 'CrankNicolson'   #| 
linear_solver_str = 'GMRes'
nonlinear_solver_str = 'Newton'

right_bc = 'periodic'
left_bc = 'periodic'
top_bc = 'isothermal_wall'
bottom_bc = 'isothermal_wall'

T_wall = 1
right_bc_args = []
left_bc_args = []
top_bc_args = [0.,0,0,T_wall]
bottom_bc_args = [0.,0,0,T_wall+0.1]
BCs = [right_bc,right_bc_args,top_bc,top_bc_args,left_bc,left_bc_args,bottom_bc,bottom_bc_args]
source = True
Re_tau = 180 
pbar_x = Re_tau**2*mu**2
source_mag = np.array([0,0.01,0,0,0])

#== Assign initial condition function. Note that you can alternatively define this here
#== function layout is my_ic_function(x,y,z), where x,y,z are the decomposed quadrature points
IC_function = TGVIC_custom               #|
                                   #|
execfile('../../src_spacetime/PyDG.py')      #|  call the solver
