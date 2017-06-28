import numpy as np
import sys
from scipy.interpolate import RegularGridInterpolator
sys.path.append("../../src/") #link the the source directory for PyDG
### INTERPOLATING
#gridfile = np.load('grid.npz')
dx = 2*np.pi/(256.)
dy = 2*np.pi/(256.)
dz = 2*np.pi/(256.)
xg = np.linspace(0,2.*np.pi-dx,256)
yg = np.linspace(0,2.*np.pi-dy,256)
zg = np.linspace(0,2.*np.pi-dz,256)

solfile = np.load('restartIC_withpressure.npz')
xd = xg[:]
xd = np.append(xd,2.*np.pi)
yd = yg[:]
yd = np.append(yd,2.*np.pi)
zd = zg[:]
zd = np.append(zd,2.*np.pi)

ud = solfile['u'][:,:,:]
vd = solfile['v'][:,:,:]
wd = solfile['w'][:,:,:]
pd = solfile['p'][:,:,:]
## extend solution by 1 in each direction (with periodic bcs) for interpolating
nsx,nsy,nsz = np.shape(ud)
ud2 = np.zeros((nsx+1,nsy+1,nsz+1))
ud2[0:-1,0:-1,0:-1] = ud[:,:,:]
ud2[-1,0:-1,0:-1] = ud[0,:,:]
ud2[0:-1,-1,0:-1] = ud[:,0,:]
ud2[0:-1,0:-1,-1] = ud[:,:,0]
ud2[-1,-1,:] = ud2[0,0,:]
ud2[-1,:,-1] = ud2[0,:,0]
ud2[:,-1,-1] = ud2[:,0,0]

vd2 = np.zeros((nsx+1,nsy+1,nsz+1))
vd2[0:-1,0:-1,0:-1] = vd[:,:,:]
vd2[-1,0:-1,0:-1] = vd[0,:,:]
vd2[0:-1,-1,0:-1] = vd[:,0,:]
vd2[0:-1,0:-1,-1] = vd[:,:,0]
vd2[-1,-1,:] = vd2[0,0,:]
vd2[-1,:,-1] = vd2[0,:,0]
vd2[:,-1,-1] = vd2[:,0,0]

wd2 = np.zeros((nsx+1,nsy+1,nsz+1))
wd2[0:-1,0:-1,0:-1] = wd[:,:,:]
wd2[-1,0:-1,0:-1] = wd[0,:,:]
wd2[0:-1,-1,0:-1] = wd[:,0,:]
wd2[0:-1,0:-1,-1] = wd[:,:,0]
wd2[-1,-1,:] = wd2[0,0,:]
wd2[-1,:,-1] = wd2[0,:,0]
wd2[:,-1,-1] = wd2[:,0,0]
 
pd2 = np.zeros((nsx+1,nsy+1,nsz+1))
pd2[0:-1,0:-1,0:-1] = pd[:,:,:]
pd2[-1,0:-1,0:-1] = pd[0,:,:]
pd2[0:-1,-1,0:-1] = pd[:,0,:]
pd2[0:-1,0:-1,-1] = pd[:,:,0]
pd2[-1,-1,:] = pd2[0,0,:]
pd2[-1,:,-1] = pd2[0,:,0]
pd2[:,-1,-1] = pd2[:,0,0]

my_interpolating_function_u = RegularGridInterpolator( (xd,yd,zd),ud2)
my_interpolating_function_v = RegularGridInterpolator( (xd,yd,zd),vd2)
my_interpolating_function_w = RegularGridInterpolator( (xd,yd,zd),wd2)
my_interpolating_function_p = RegularGridInterpolator( (xd,yd,zd),pd2)


def hitICS(x,y,z,gas):
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  gamma = gas.gamma
  Minf = 0.2

  R = gas.R #1
  T0 = 1./gamma
  rho = 1.
  a = np.sqrt(gamma*R*T0)
  V0 = Minf*a
  Cv = 5./2.*R
  p0 = rho*R*T0

  q = np.zeros((5,nqx,nqy,nqz,Nelx,Nely,Nelz))
  T = np.zeros((nqz,nqy,nqz,Nelx,Nely,Nelz))
  points = np.zeros((nqx*nqy*nqz*Nelx*Nely*Nelz,3))
  u = np.zeros((nqx,nqy,nqz,Nelx,Nely,Nelz))
  v = np.zeros((nqx,nqy,nqz,Nelx,Nely,Nelz))
  w = np.zeros((nqx,nqy,nqz,Nelx,Nely,Nelz))
  p = np.zeros((nqx,nqy,nqz,Nelx,Nely,Nelz))
  rho = np.ones((nqx,nqy,nqz,Nelx,Nely,Nelz))*rho
  points[:,0] = x.flatten()
  points[:,1] = y.flatten()
  points[:,2] = z.flatten()
  uInterp = my_interpolating_function_u(points)
  vInterp = my_interpolating_function_v(points)
  wInterp = my_interpolating_function_w(points)
  pInterp = my_interpolating_function_p(points)
  u[:,:,:] = V0*np.reshape(uInterp,(nqx,nqy,nqz,Nelx,Nely,Nelz))
  v[:,:,:] = V0*np.reshape(vInterp,(nqx,nqy,nqz,Nelx,Nely,Nelz))
  w[:,:,:] = V0*np.reshape(wInterp,(nqx,nqy,nqz,Nelx,Nely,Nelz))
  p[:,::,] = V0**2*np.reshape(pInterp,(nqx,nqy,nqz,Nelx,Nely,Nelz))
  p += p0
  T = p/(rho*R)
  #print(np.amax(abs(u)/(np.sqrt(gamma*R*T))))
  E = Cv*T + 0.5*(u**2 + v**2 + w**2)
  q[0] = rho[:]
  q[1] = rho[:]*u[:]
  q[2] = rho[:]*v[:]
  q[3] = rho[:]*w[:]
  q[4] = rho[:]*E[:]
  return q



## Make square grid
Minf = 0.2
L = 2.*np.pi                       #|  length
Nel = np.array([2**0,2**0,2**0])   #|  elements in x,y,z
order = np.array([64,64,64])                         #|  spatial order
quadpoints = order*3               #|  number of quadrature points. 2x the order is reccomended
mu = 0.0005 * Minf                       #|  viscocity
x = np.linspace(0,L,Nel[0]+1)      #|  x, y, and z
y = np.linspace(0,L,Nel[1]+1)      #|
z = np.linspace(0,L,Nel[2]+1)      #|
t = 0                              #|  simulation start time
dt = 0.00025                        #|  simulation time step
et = 20.                           #|  simulation end time
save_freq = 100                      #|  frequency to save output and print to screen
eqn_str = 'Navier-Stokes'          #|  equation set
schemes = ('roe','IP')             #|  inviscid and viscous flux schemes
procx = 1                          #|  processor decomposition in x
procy = 1                          #|  same in y. Note that procx*procy needs to equal total number of procs
				   #|
time_integration = 'ExplicitRK4'   #| 
linear_solver_str = 'GMRes'
nonlinear_solver_str = 'Newton_MG'

right_bc = 'periodic'
left_bc = 'periodic'
top_bc = 'periodic'
bottom_bc = 'periodic'

right_bc_args = []
left_bc_args = []
top_bc_args = []
bottom_bc_args = []
BCs = [right_bc,right_bc_args,top_bc,top_bc_args,left_bc,left_bc_args,bottom_bc,bottom_bc_args]

#== Assign initial condition function. Note that you can alternatively define this here
#== function layout is my_ic_function(x,y,z), where x,y,z are the decomposed quadrature points
IC_function = hitICS                #|
                                   #|
execfile('../../src/PyDG.py')      #|  call the solver
