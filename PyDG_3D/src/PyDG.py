import numpy as np
import os
import sys
from mpi4py import MPI
from init_Classes import *
from solver_classes import *
from DG_functions import reconstructU,volIntegrateGlob
from MPI_functions import gatherSolSlab,gatherSolSpectral
from timeSchemes import *#advanceSol,advanceSolImplicitMG,advanceSolImplicit,advanceSolImplicitPC
import time
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator

def getGlobGrid2(x,y,z,zeta):
  dx = x[1] - x[0]
  dy = y[1] - y[0]
  dz = z[1] - z[0]
  Npx,Npy,Npz = np.size(x),np.size(y),np.size(z)

  nq = np.size(zeta)
  xG = np.zeros((nq,nq,nq,Npx-1,Npy-1,Npz-1))
  yG = np.zeros((nq,nq,nq,Npx-1,Npy-1,Npz-1))
  zG = np.zeros((nq,nq,nq,Npx-1,Npy-1,Npz-1))
  for i in range(0,Npx-1):
     xG[:,:,:,i,:,:] = ( (2.*x[i]  + dx)/2. + zeta/2.*(dx) )[:,None,None,None,None]
  for i in range(0,Npy-1):
     yG[:,:,:,:,i,:] = ( (2.*y[i]  + dy)/2. + zeta/2.*(dy) )[None,:,None,None,None]
  for i in range(0,Npz-1):
     zG[:,:,:,:,:,i] = ( (2.*z[i]  + dz)/2. + zeta/2.*(dz) )[None,None,:,None,None]
  return xG,yG,zG


def getGlobGrid(x,y,z,zeta):
  dx = x[1] - x[0]
  dy = y[1] - y[0]
  dz = z[1] - z[0]
  Nelx,Nely,Nelz = np.size(x),np.size(y),np.size(z)
  order = np.size(zeta)
  xG = np.zeros(((np.size(x)-1)*np.size(zeta)))
  yG = np.zeros(((np.size(y)-1)*np.size(zeta)))
  zG = np.zeros(((np.size(z)-1)*np.size(zeta)))
  for i in range(0,Nelx-1):
     xG[i*quadpoints:(i+1)*quadpoints] = (2.*x[i]  + dx)/2. + zeta/2.*(dx)
  for i in range(0,Nely-1):
     yG[i*quadpoints:(i+1)*quadpoints] = (2.*y[i]  + dy)/2. + zeta/2.*(dy)
  for i in range(0,Nelz-1):
     zG[i*quadpoints:(i+1)*quadpoints] = (2.*z[i]  + dz)/2. + zeta/2.*(dz)

  return xG,yG,zG


def getGlobU(u):
  nvars,quadpoints,quadpoints,quadpoints,Nelx,Nely,Nelz = np.shape(u)
  uG = np.zeros((nvars,quadpoints*Nelx,quadpoints*Nely,quadpoints*Nelz))
  for i in range(0,Nelx):
    for j in range(0,Nely):
      for k in range(0,Nelz):
        for m in range(0,nvars):
          uG[m,i*quadpoints:(i+1)*quadpoints,j*quadpoints:(j+1)*quadpoints,k*quadpoints:(k+1)*quadpoints] = u[m,:,:,:,i,j,k]
  return uG


def getIC(main,f,x,y,z):
  ## First perform integration in x
  ord_arr= np.linspace(0,order-1,order)
  scale =  (2.*ord_arr[:,None,None] + 1.)*(2.*ord_arr[None,:,None] + 1.)*(2.*ord_arr[None,None,:]+1.)/8.
  U = f(x,y,z)
  main.a.a[:] = volIntegrateGlob(main,U,main.w,main.w,main.w)*scale[None,:,:,:,None,None,None]



comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
mpi_rank = comm.Get_rank()
if (mpi_rank == 0):
  print('Running on ' + str(num_processes) + ' Procs')
dx =  L/Nel[0]
t = 0
if (mpi_rank == 0):
  print('CFL = ' + str(10.*dt/(dx/order)))
iteration = 0
eqns = equations(eqn_str,schemes)
main = variables(Nel,order,quadpoints,eqns,mu,x,y,z,t,et,dt,iteration,save_freq,'DNS',procx,procy)

xG,yG,zG = getGlobGrid(x,y,z,main.zeta)
xG2,yG2,zG2 = getGlobGrid2(x,y,z,main.zeta)

getIC(main,IC_function,xG2[:,:,:,main.sx,main.sy,:],yG2[:,:,:,main.sx,main.sy,:],zG2[:,:,:,main.sx,main.sy,:])
reconstructU(main,main.a)

timescheme = timeschemes(time_integration)

np.savez('DGgrid',x=xG,y=yG,z=zG)
if (main.mpi_rank == 0):
  if not os.path.exists('Solution'):
     os.makedirs('Solution')


t0 = time.time()
while (main.t <= main.et - main.dt/2):
  if (main.iteration%main.save_freq == 0):
    reconstructU(main,main.a)
    uG = gatherSolSlab(main,eqns,main.a)
    aG = gatherSolSpectral(main.a.a,main)
    if (main.mpi_rank == 0):
      UG = getGlobU(uG)
      #uGF = getGlobU(uG)
      sys.stdout.write('======================================' + '\n')
      sys.stdout.write('wall time = ' + str(time.time() - t0) + '\n' )
      sys.stdout.write('t = ' + str(main.t) +  '   rho sum = ' + str(np.sum(uG[0])) + '\n')
      np.savez('Solution/npsol' + str(main.iteration),U=UG,a=aG,t=main.t,iteration=main.iteration,order=order)
      sys.stdout.flush()

  timescheme.advanceSol(main,main,eqns,timescheme.args)
  #advanceSolImplicit_MYNK(main,main,eqns)
reconstructU(main,main.a)
uG = gatherSolSlab(main,eqns,main.a)
if (main.mpi_rank == 0):
  print('Final Time = ' + str(time.time() - t0),'Sol Norm = ' + str(np.linalg.norm(uG)) ),
