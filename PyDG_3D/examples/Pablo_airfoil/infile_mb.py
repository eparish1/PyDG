import numpy as np
import sys
sys.path.append("../../src_spacetime") #link the the source directory for PyDG
from ic_functions_premade import vortexICS,zeroFSIC  #import the IC for taylor green vortex.

#==========================================================================

alfa      = 15.*np.pi/180.
uinf      = 0.3*np.cos(alfa)
vinf      = 0.3*np.sin(alfa)

#==========================================================================

def savehook(main):
  
  if (main.mpi_rank == 0):
      print('rho norm  ' + str(np.linalg.norm(main.a.u[0])))

#==========================================================================

def airfoilIC(x,y,z,main):

  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  q       = np.zeros((5,nqx,nqy,nqz,Nelx,Nely,Nelz))
  gamma   = main.gas.gamma
  Cv      = main.gas.Cv
  Cp      = Cv*gamma
  R       = Cp - Cv
  T       = np.zeros((nqx,nqy,nqz,Nelx,Nely,Nelz))
  rho     = np.zeros(np.shape(T))
  u       = np.zeros(np.shape(T))
  v       = np.zeros(np.shape(T))

  global uinf
  global vinf

  u[:]    = uinf
  v[:]    = vinf
  w       = np.zeros(np.shape(u))
  E       = np.zeros(np.shape(u))
  T       = 1./gamma
  p       = 1.
  rho[:]  = p/(R*T)
  q[0]    = rho
  q[1]    = rho*u
  q[2]    = rho*v
  q[3]    = rho*w
  q[4]    = p/(1.4 - 1.) + 0.5*rho*(u**2 + v**2 + w**2)

  return q

#==========================================================================
## Make square grid

n_blocks = 6

Nel_block0 = np.array([10,24,10,1])   #|  elements in x,y,z
Nel_block1 = np.array([30,89,10,1])   #|  elements in x,y,z
Nel_block2 = np.array([30,54,10,1])   #|  elements in x,y,z
Nel_block3 = np.array([30,24,10,1])   #|  elements in x,y,z
Nel_block4 = np.array([30,54,10,1])   #|  elements in x,y,z
Nel_block5 = np.array([30,89,10,1])   #|  elements in x,y,z

order =np.array([1,1,1,1])            #|  spatial order

quadpoints = np.array([order[0],order[1],order[2],order[3]])               #|  number of quadrature points. 2x the order is reccomended

mu = 0.00001875
gamma = 1.4

#=============== Importing blocks and constructing mesh ===================

blk0 = np.genfromtxt('Meshes/3Dblk0.dat')
blk1 = np.genfromtxt('Meshes/3Dblk1.dat')
blk2 = np.genfromtxt('Meshes/3Dblk2.dat')
blk3 = np.genfromtxt('Meshes/3Dblk3.dat')
blk4 = np.genfromtxt('Meshes/3Dblk4.dat')
blk5 = np.genfromtxt('Meshes/3Dblk5.dat')

x_temp_0 = np.reshape(blk0[:,0],((Nel_block0[0]+1,Nel_block0[1]+1,Nel_block0[2]+1)))
y_temp_0 = np.reshape(blk0[:,1],((Nel_block0[0]+1,Nel_block0[1]+1,Nel_block0[2]+1)))
z_temp_0 = np.reshape(blk0[:,2],((Nel_block0[0]+1,Nel_block0[1]+1,Nel_block0[2]+1)))


x_temp_1 = np.reshape(blk1[:,0],((Nel_block1[0]+1,Nel_block1[1]+1,Nel_block1[2]+1)))
y_temp_1 = np.reshape(blk1[:,1],((Nel_block1[0]+1,Nel_block1[1]+1,Nel_block1[2]+1)))
z_temp_1 = np.reshape(blk1[:,2],((Nel_block1[0]+1,Nel_block1[1]+1,Nel_block1[2]+1)))


x_temp_2 = np.reshape(blk2[:,0],((Nel_block2[0]+1,Nel_block2[1]+1,Nel_block2[2]+1)))
y_temp_2 = np.reshape(blk2[:,1],((Nel_block2[0]+1,Nel_block2[1]+1,Nel_block2[2]+1)))
z_temp_2 = np.reshape(blk2[:,2],((Nel_block2[0]+1,Nel_block2[1]+1,Nel_block2[2]+1)))


x_temp_3 = np.reshape(blk3[:,0],((Nel_block3[0]+1,Nel_block3[1]+1,Nel_block3[2]+1)))
y_temp_3 = np.reshape(blk3[:,1],((Nel_block3[0]+1,Nel_block3[1]+1,Nel_block3[2]+1)))
z_temp_3 = np.reshape(blk3[:,2],((Nel_block3[0]+1,Nel_block3[1]+1,Nel_block3[2]+1)))


x_temp_4 = np.reshape(blk4[:,0],((Nel_block4[0]+1,Nel_block4[1]+1,Nel_block4[2]+1)))
y_temp_4 = np.reshape(blk4[:,1],((Nel_block4[0]+1,Nel_block4[1]+1,Nel_block4[2]+1)))
z_temp_4 = np.reshape(blk4[:,2],((Nel_block4[0]+1,Nel_block4[1]+1,Nel_block4[2]+1)))


x_temp_5 = np.reshape(blk5[:,0],((Nel_block5[0]+1,Nel_block5[1]+1,Nel_block5[2]+1)))
y_temp_5 = np.reshape(blk5[:,1],((Nel_block5[0]+1,Nel_block5[1]+1,Nel_block5[2]+1)))
z_temp_5 = np.reshape(blk5[:,2],((Nel_block5[0]+1,Nel_block5[1]+1,Nel_block5[2]+1)))

#==========================================================================

Nel_block = [Nel_block0,Nel_block1,Nel_block2,Nel_block3,Nel_block4,Nel_block5]
x_block   = [ x_temp_0,  x_temp_1,  x_temp_2,  x_temp_3,  x_temp_4,  x_temp_5 ]
y_block   = [ y_temp_0,  y_temp_1,  y_temp_2,  y_temp_3,  y_temp_4,  y_temp_5 ]
z_block   = [ z_temp_0,  z_temp_1,  z_temp_2,  z_temp_3,  z_temp_4,  z_temp_5 ]

t                   = 0                     #|  simulation start time
dt                  = 0.000125              #|  simulation time step
et                  = 10.                   #|  simulation end time
save_freq           = 5                     #|  frequency to save output and print to screen
eqn_str             = 'Navier-Stokes'       #|  equation set
schemes             = ('roe','BR1')         #|  inviscid and viscous flux schemes
basis_functions_str = 'TensorDot'
#orthogonal_str      = True

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

procx_block4 = 1
procy_block4 = 1
procz_block4 = 1

procx_block5 = 1
procy_block5 = 1
procz_block5 = 1

starting_rank0 = 0
starting_rank1 = starting_rank0 + procx_block0*procy_block0*procz_block0
starting_rank2 = starting_rank1 + procx_block1*procy_block1*procz_block1 
starting_rank3 = starting_rank2 + procx_block2*procy_block2*procz_block2
starting_rank4 = starting_rank3 + procx_block3*procy_block3*procz_block3 
starting_rank5 = starting_rank4 + procx_block4*procy_block4*procz_block4

#=================

procx = [procx_block0,procx_block1,procx_block2,procx_block3,procx_block4,procx_block5]
procy = [procy_block0,procy_block1,procy_block2,procy_block3,procy_block4,procy_block5]
procz = [procz_block0,procz_block1,procz_block2,procz_block3,procz_block4,procy_block5]

starting_rank = [starting_rank0,starting_rank1,starting_rank2,starting_rank3,starting_rank4,starting_rank5]

#======================================

l_bc0 = 'isothermal_wall'
r_bc0 = 'patch'
d_bc0 = 'isothermal_wall'
u_bc0 = 'isothermal_wall'
b_bc0 = 'periodic'
f_bc0 = 'periodic'

l_bc1 = [uinf,vinf,0.,1./gamma]
r_bc1 = [3,0,0,0,0]
d_bc1 = [uinf,vinf,0.,1./gamma]
u_bc1 = [uinf,vinf,0.,1./gamma]
b_bc1 = [0,0,0,0,0]
f_bc1 = [0,0,0,0,0]

BCs_block0 = [l_bc0, l_bc1, r_bc0, r_bc1, d_bc0, d_bc1, u_bc0, u_bc1, b_bc0, b_bc1, f_bc0, f_bc1]

#=================
#=================

l_bc0 = 'isothermal_wall'
r_bc0 = 'patch'
d_bc0 = 'patch'
u_bc0 = 'isothermal_wall'
b_bc0 = 'periodic'
f_bc0 = 'periodic'

l_bc1 = [uinf,vinf,0.,1./gamma]
r_bc1 = [5,-1,0,1,0]
d_bc1 = [2,-1,0,0,0]
u_bc1 = [uinf,vinf,0.,1./gamma]
b_bc1 = [0,0,0,0,0]
f_bc1 = [0,0,0,0,0]

BCs_block1 = [l_bc0, l_bc1, r_bc0, r_bc1, d_bc0, d_bc1, u_bc0, u_bc1, b_bc0, b_bc1, f_bc0, f_bc1]

#=================
#=================

l_bc0 = 'isothermal_wall'
r_bc0 = 'isothermal_wall'
d_bc0 = 'patch'
u_bc0 = 'patch'
b_bc0 = 'periodic'
f_bc0 = 'periodic'

l_bc1 = [uinf,vinf,0.,1./gamma]
r_bc1 = [0.,0.,0.,1./gamma]
d_bc1 = [3,-1,0,0,0]
u_bc1 = [1, 0,0,0,0]
b_bc1 = [0,0,0,0,0]
f_bc1 = [0,0,0,0,0]

BCs_block2 = [l_bc0, l_bc1, r_bc0, r_bc1, d_bc0, d_bc1, u_bc0, u_bc1, b_bc0, b_bc1, f_bc0, f_bc1]

#=================
#=================

l_bc0 = 'patch'
r_bc0 = 'isothermal_wall'
d_bc0 = 'patch'
u_bc0 = 'patch'
b_bc0 = 'periodic'
f_bc0 = 'periodic'

l_bc1 = [0,-1,0,0,0]
r_bc1 = [0.,0.,0.,1./gamma]
d_bc1 = [4,-1,0,0,0]
u_bc1 = [2, 0,0,0,0]
b_bc1 = [0,0,0,0,0]
f_bc1 = [0,0,0,0,0]

BCs_block3 = [l_bc0, l_bc1, r_bc0, r_bc1, d_bc0, d_bc1, u_bc0, u_bc1, b_bc0, b_bc1, f_bc0, f_bc1]

#=================
#=================

l_bc0 = 'isothermal_wall'
r_bc0 = 'isothermal_wall'
d_bc0 = 'patch'
u_bc0 = 'patch'
b_bc0 = 'periodic'
f_bc0 = 'periodic'

l_bc1 = [uinf,vinf,0.,1./gamma]
r_bc1 = [0.,0.,0.,1./gamma]
d_bc1 = [5,-1,0,0,0]
u_bc1 = [3, 0,0,0,0]
b_bc1 = [0,0,0,0,0]
f_bc1 = [0,0,0,0,0]

BCs_block4 = [l_bc0, l_bc1, r_bc0, r_bc1, d_bc0, d_bc1, u_bc0, u_bc1, b_bc0, b_bc1, f_bc0, f_bc1]

#=================
#=================

l_bc0 = 'isothermal_wall'
r_bc0 = 'patch'
d_bc0 = 'isothermal_wall'
u_bc0 = 'patch'
b_bc0 = 'periodic'
f_bc0 = 'periodic'

l_bc1 = [uinf,vinf,0.,1./gamma]
r_bc1 = [1,-1,0,1,0]
d_bc1 = [0.,0.,0.,1./gamma]
u_bc1 = [4, 0,0,0,0]
b_bc1 = [0,0,0,0,0]
f_bc1 = [0,0,0,0,0]

BCs_block5 = [l_bc0, l_bc1, r_bc0, r_bc1, d_bc0, d_bc1, u_bc0, u_bc1, b_bc0, b_bc1, f_bc0, f_bc1]

#==================

BCs = [BCs_block0,BCs_block1,BCs_block2,BCs_block3,BCs_block4,BCs_block5]

#===============================================

source_mag = False
mol_str = False				   #|

#turb_str = 'orthogonal subscale'
#time_integration = 'SpaceTime'   #| 
#time_integration = 'CrankNicolson'   #| 

time_integration = 'SSP_RK3'
linear_solver_str = 'GMRes'
nonlinear_solver_str = 'Newton'

#== Assign initial condition function. Note that you can alternatively define this here
#== function layout is my_ic_function(x,y,z), where x,y,z are the decomposed quadrature points
IC_function = [airfoilIC,airfoilIC,airfoilIC,airfoilIC,airfoilIC,airfoilIC]                #|

execfile('../../src_spacetime/PyDG.py')      #|  call the solver
