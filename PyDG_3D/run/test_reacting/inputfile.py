import numpy as np
import sys
sys.path.append("../../src_spacetime") #link the the source directory for PyDG
from eos_functions import *
from ic_functions_premade import vortexICS  #import the IC for taylor green vortex.

def vortexICS_custom(x,y,z,main):
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  q = np.zeros((7,nqx,nqy,nqz,Nelx,Nely,Nelz))
  gamma = 1.4
  y0 = 5.
  x0 = 5.
  Cv = main.Cv[0]
  Cp = main.Cp[0]
  R = Cp - Cv
  T = np.zeros((nqx,nqy,nqz,Nelx,Nely,Nelz))
  rho = np.zeros(np.shape(T))
  u = np.zeros(np.shape(rho))
  v = np.zeros(np.shape(u))
  w = np.zeros(np.shape(u))
  E = np.zeros(np.shape(u))
  r = ( (x - x0)**2 + (y - y0)**2 )**0.5
  beta = 0.
  pi = np.pi
  T[:] = 298.# - (gamma - 1.)*beta**2/(8.*gamma*pi**2)*np.exp(1. - r**2)
  rho[:] = 1.225#T**(1./(gamma - 1.))
  q[5,x>L/2] = rho[x>L/2]*1. #mixture fractions
  q[6] = rho*(1. - 1./rho*q[5])

  u[:] = 0. + beta/(2.*pi)*np.exp( (1. - r**2)/2.)*-(y - y0)
  v[:] = 0. +  beta/(2.*pi)*np.exp( (1. - r**2)/2.)*(x - x0)
  E[:] = computeEnergy(main,T,q[5::]/rho,u,v,w)
  print(np.linalg.norm(E))
  #Cv*T + 0.5*(u**2 + v**2)
  print('mach = ' + str(np.mean(np.sqrt(gamma*R*1000*T))))
  print('u = ' + str(np.mean(np.abs(u))))
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*w
  q[4] = rho*E
  p2,T2 = computePressure_and_Temperature(main,q)
  p3 = (gamma - 1.)*(q[4] - 0.5*q[1]**2/q[0] - 0.5*q[2]**2/q[0] - 0.5*q[3]**2/q[0])
  print('p2 = ' + str(np.mean(p2)))
  print('p3 = ' + str(np.mean(p3)))

  print('T2 = ' + str(np.mean(T2)))
  return q


## Make square grid
L = 10.                       #|  length
Nel = np.array([2**3,2**3,1,1])   #|  elements in x,y,z
order =np.array([4,4,4,1])                       #|  spatial order
quadpoints = order*2               #|  number of quadrature points. 2x the order is reccomended
quadpoints[-2] = order[-2]
quadpoints[-1] = order[-1]
mu = 0.001                       #|  viscocity
x = np.linspace(0,L,Nel[0]+1)      #|  x, y, and z
y = np.linspace(0,L,Nel[1]+1)      #|
z = np.linspace(0,L,Nel[2]+1)      #|
t = 0                              #|  simulation start time
dt = 0.00005
#dt = 2                        #|  simulation time step
et = 40.                           #|  simulation end time
save_freq = 10                      #|  frequency to save output and print to screen
eqn_str = 'Navier-Stokes Reacting 2'          #|  equation set
schemes = ('rusanov','Inviscid')             #|  inviscid and viscous flux schemes
procx = 1                         #|  processor decomposition in x
procy = 1                          #|  same in y. Note that procx*procy needs to equal total number of procs

mol_str = ['N2','O2']
right_bc = 'periodic'
left_bc = 'periodic'
top_bc = 'periodic'
bottom_bc = 'periodic'

T_wall = 1
right_bc_args = []
left_bc_args = []
top_bc_args = []
bottom_bc_args = []
BCs = [right_bc,right_bc_args,top_bc,top_bc_args,left_bc,left_bc_args,bottom_bc,bottom_bc_args]
				   #|
#time_integration = 'CrankNicolson'
time_integration = 'ExplicitRK4'
#turb_str = 'tau-modelFD'
#time_integration = 'SpaceTime'   #| 
linear_solver_str = 'GMRes'
nonlinear_solver_str = 'Newton_MG'

#== Assign initial condition function. Note that you can alternatively define this here
#== function layout is my_ic_function(x,y,z), where x,y,z are the decomposed quadrature points
IC_function = vortexICS_custom                #|
                                   #|
execfile('../../src_spacetime/PyDG.py')      #|  call the solver
