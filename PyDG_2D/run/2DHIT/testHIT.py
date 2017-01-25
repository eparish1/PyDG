import numpy as np
import sys
sys.path.append("../../src")

from init_Classes import *
from DG_functions import reconstructU
from MPI_functions import gatherSol
import matplotlib.pyplot as plt
from timeSchemes import advanceSol
import time
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
def getGlobGrid(x,y,zeta):
  dx = x[1] - x[0]
  dy = y[1] - y[0]
  Nelx,Nely = np.size(x),np.size(y)
  order = np.size(zeta)
  xG = np.zeros((np.size(x)*np.size(zeta)))
  yG = np.zeros((np.size(y)*np.size(zeta)))
  for i in range(0,Nelx):
     xG[i*order:(i+1)*order] = (2.*x[i]  + dx)/2. + zeta/2.*(dx)
  for i in range(0,Nely):
     yG[i*order:(i+1)*order] = (2.*y[i]  + dy)/2. + zeta/2.*(dy)
  return xG,yG


def getGlobU(u):
  nvars,order,order,Nelx,Nely = np.shape(u)
  uG = np.zeros((nvars,order*Nelx,order*Nely))
  for i in range(0,Nelx):
    for j in range(0,Nely):
      for m in range(0,nvars):
        uG[m,i*order:(i+1)*order,j*order:(j+1)*order] = u[m,:,:,i,j]
  return uG

### INTERPOLATING
gridfile = np.load('grid.npz')
solfile = np.load('npsol20.npz')
xd = gridfile['x'][:,0,0]
xd = np.append(xd,2.*np.pi)
yd = gridfile['y'][0,:,0]
yd = np.append(yd,2.*np.pi)
zd = gridfile['z'][:,:,0]
xVec = gridfile['x'].flatten()
yVec = gridfile['y'].flatten()

ud = solfile['u'][:,:,0]
vd = solfile['v'][:,:,0]
pd = solfile['p'][:,:,0]


## extend solution by 1 in each direction (with periodic bcs) for interpolating
nsx,nsy = np.shape(ud)
ud2 = np.zeros((nsx+1,nsy+1))
ud2[0:-1,0:-1] = ud[:,:]
ud2[-1,0:-1] = ud[0,:]
ud2[0:-1,-1] = ud[:,0]
ud2[-1,-1] = ud[0,0]
vd2 = np.zeros((nsx+1,nsy+1))
vd2[0:-1,0:-1] = vd[:,:]
vd2[-1,0:-1] = vd[0,:]
vd2[0:-1,-1] = vd[:,0]
vd2[-1,-1] = vd[0,0]
pd2 = np.zeros((nsx+1,nsy+1))
pd2[0:-1,0:-1] = pd[:,:]
pd2[-1,0:-1] = pd[0,:]
pd2[0:-1,-1] = pd[:,0]
pd2[-1,-1] = pd[0,0]

my_interpolating_function_u = RegularGridInterpolator( (xd,yd),ud2)
my_interpolating_function_v = RegularGridInterpolator( (xd,yd),vd2)
my_interpolating_function_p = RegularGridInterpolator( (xd,yd),pd2)

def hitICS(x,y,qnum):
  nx,ny = np.shape(x)
  gamma = 1.4
  y0 = 5.
  x0 = 5.
  Cv = 5/2.
  Cp = Cv*gamma
  R = Cp - Cv
  nx,ny = np.shape(x)
  q = np.zeros((4,nx,ny))
  T = np.zeros((nx,ny))
  points = np.zeros((nx*ny,2))
  u = np.zeros((nx,ny))
  v = np.zeros((nx,ny))
  w = np.zeros((nx,ny))
  p = np.zeros((nx,ny))
  p0 = 1000.
  rho = np.ones((nx,ny))
  points[:,0] = x.flatten()
  points[:,1] = y.flatten()
  uInterp = my_interpolating_function_u(points)
  vInterp = my_interpolating_function_v(points)
  pInterp = my_interpolating_function_p(points)
  u[:,:] = np.reshape(uInterp,(nx,ny))
  v[:,:] = np.reshape(vInterp,(nx,ny))
  p[:,:] = np.reshape(pInterp,(nx,ny))
  p += p0
  T = p/(rho*R)
  #print(np.amax(abs(u)/(np.sqrt(gamma*R*T))))
  E = Cv*T + 0.5*(u**2 + v**2)
  q = np.zeros((4,nx,ny))
  q[0] = rho[:]
  q[1] = rho[:]*u[:] 
  q[2] = rho[:]*v[:]
  q[3] = rho[:]*E[:]
  return q[qnum]


def getIC(main,f,qnum):
  ## First perform integration in x
  for i in range(0,main.Npx):
    for j in range(0,main.Npy):
      xloc = (main.x[i] + main.x[i] + main.dx)/2. + main.zeta/2.*(main.dx)
      yloc = (main.y[j] + main.y[j] + main.dy)/2. + main.zeta/2.*(main.dy)
      for p in range(0,main.order):
        for q in range(0,main.order):
          tg = np.ones((main.order,main.order))
          main.a.a[qnum,p,q,i,j] = (2.*p+1)/2.*(2.*q+1)/2.*np.sum(main.weights[:,None]*main.weights[None,:]*main.w[p,:,None]*main.w[q,None,:]*f(xloc[:,None]*tg,yloc[None,:]*tg,qnum))


L = 2.*np.pi
L = 2.*np.pi
Nel = np.array([2**6,2**6])
order = 3
nu = 0.1
x = np.linspace(0,L,Nel[0]+1)
y = np.linspace(0,L,Nel[1]+1)
t = 0
dt = 2.e-4
et = 10.
iteration = 0
save_freq = 10
eqns = equations('Navier-Stokes')
main = variables(Nel,order,eqns,nu,x,y,t,et,dt,iteration,save_freq)
schemes = fschemes('rusanov','central')
xG,yG = getGlobGrid(x,y,main.zeta)
for qnum in range(0,eqns.nvars):
  getIC(main,hitICS,qnum)
  print(qnum)
reconstructU(main,main.a)


#print(np.linalg.norm(main.u[0]))
t0 = time.time()
while (main.t <= main.et - main.dt/2):
  if (main.iteration%main.save_freq == 0):
    reconstructU(main,main.a)
    uG = gatherSol(main,eqns,main.a)
    if (main.mpi_rank == 0):
      uGF = getGlobU(uG)
      print('t = ' + str(main.t),'rho sum = ' + str(np.sum(uG[0])) )
      plt.clf()
      plt.contourf(uGF[1,:,:],100)
      plt.pause(0.0001)
  advanceSol(main,eqns,schemes)
print('Final Time = ' + str(time.time() - t0))
