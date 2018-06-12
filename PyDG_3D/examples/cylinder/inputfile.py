import numpy as np
import sys
PyDG_DIR = '../../src_spacetime'
sys.path.append("../../src_spacetime") #link the the source directory for PyDG
from ic_functions_premade import zeroFSIC  #import the IC for taylor green vortex.
def cylinderIC(x,y,z,main):
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  q = np.zeros((5,nqx,nqy,nqz,Nelx,Nely,Nelz))
  gamma = main.gas.gamma
  Cv = main.gas.Cv
  Cp = Cv*gamma
  R = Cp - Cv
  T = np.zeros((nqx,nqy,nqz,Nelx,Nely,Nelz))
  rho = np.zeros(np.shape(T))
  u = np.zeros(np.shape(rho))
  u[:] = 0.1
  v = np.zeros(np.shape(u))
  w = np.zeros(np.shape(u))
  E = np.zeros(np.shape(u))
  T = 1./gamma
  p = 1.
  #p = (gamma - 1.)*(rhoE - 0.5*u*rhoU - 0.5*v*rhoV - 0.5*w*rhoW)
  rho[:] = p/(R*T)
  #print(R)
  #E[:] = p/(1.4 - 1.)
  E[:] = Cv*T + 0.5*(u**2 + v**2 + w**2)
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*w
  q[4] = rho*E
  return q

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
    s_int += globalSum(vol_integral[-1] , regionManager)
    rho_int += globalSum(vol_integral[0] , regionManager)
  if (main.mpi_rank == 0):
      sys.stdout.write('Mass = ' + str(rho_int) +  '  Entropy = ' + str(s_int) +  '\n')
      sys.stdout.flush()


## Make square grid
L = 1.                             #|  length
Nel = np.array([32,32,1,1])        #|  elements in x,y,z
order =np.array([3,3,1,1])         #|  spatial order
quadpoints = order               #|  number of quadrature points. 2x the order is reccomended
quadpoints[-2::] = order[-2::]

D = 2.
Re = 200.
rho0 = 1.4
Minf = 0.1
mu = (rho0*Minf*D)/Re                          #|  viscocity
z = np.linspace(0,L,Nel[2]+1)      #|
#r = np.linspace(0.1,1,Nel[1]+1)
Rg = 40.**(1./Nel[1])
r = np.zeros(Nel[1]+1)
r[0] = 1.
for i in range(0,Nel[1]):
  r[i+1] = r[i] + r[i]*(Rg - 1.) 

theta = np.linspace(np.pi,-np.pi,Nel[0] + 1)
theta,r,z = np.meshgrid(theta,r,z,indexing='ij')
x = r*np.cos(theta)
y = r*np.sin(theta)
#x,y,z = np.meshgrid(x,y,z,indexing='ij')

t = 0                              #|  simulation start time
dt = 0.02                       #|  simulation time step
et = 1000.                           #|  simulation end time
save_freq = 100                      #|  frequency to save output and print to screen
eqn_str = 'Navier-Stokes'          #|  equation set
schemes = ('roe','Inviscid')             #|  inviscid and viscous flux schemes
procx = 4                          #|  processor decomposition in x
procy = 1                          #|  same in y. Note that procx*procy needs to equal total number of procs
procz = 1				   #|
time_integration = 'SSP_RK3'   #| 
linear_solver_str = 'GMRes'
nonlinear_solver_str = 'Newton'

right_bc = 'periodic'
left_bc = 'periodic'
#top_bc = 'isothermal_wall'
top_bc = 'dirichlet'
bottom_bc = 'inviscid_wall'
front_bc = 'periodic'
back_bc = 'periodic'
right_bc_args = [0,0,0,0]
left_bc_args = [0,-1,0,0]
T_inlet = (2.5*1./1.4 - 1./2.*0.1**2)/(2.5)
#top_bc_args = [0.1,0.,0,T_inlet]
rho0 = 1.4
rhoU0 = rho0*Minf
rhoE0 = 2.5
top_bc_args = [rho0,rhoU0,0.,0.,rhoE0]
bottom_bc_args = [0.,0.,0.,1./1.4]
front_bc_args = [0,0,0,0]
back_bc_args = [0,-1,0,0]
BCs = [left_bc,left_bc_args,right_bc,right_bc_args,bottom_bc,bottom_bc_args,top_bc,top_bc_args,back_bc,back_bc_args,front_bc,front_bc_args]
mol_str = False
fsource = False 
source_mag = False

#== Assign initial condition function. Note that you can alternatively define this here
#== function layout is my_ic_function(x,y,z,main), where x,y,z are the decomposed quadrature points
IC_function = cylinderIC               #|
                                   #|
execfile('../../src_spacetime/PyDG.py')      #|  call the solver
