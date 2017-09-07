import numpy as np
import os
import sys
from mpi4py import MPI
from init_Classes import *
from solver_classes import *
from DG_functions import reconstructU_tensordot as reconstructU
from DG_functions import volIntegrateGlob_tensordot as volIntegrateGlob
from MPI_functions import gatherSolSlab,gatherSolSpectral,gatherSolScalar,globalSum
from init_reacting_additions import *
from timeSchemes import *#advanceSol,advanceSolImplicitMG,advanceSolImplicit,advanceSolImplicitPC
import time
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from basis_class import *
def getGlobGrid2(x,y,z,zeta0,zeta1,zeta2):
#  dx = x[1] - x[0]
#  dy = y[1] - y[0]
#  dz = z[1] - z[0]

  Npx,Npy,Npz = int(np.size(x)),int(np.size(y)),int(np.size(z))

  nqx = np.size(zeta0)
  nqy = np.size(zeta1)
  nqz = np.size(zeta2)

  xG = np.zeros((nqx,nqy,nqz,Npx-1,Npy-1,Npz-1))
  yG = np.zeros((nqx,nqy,nqz,Npx-1,Npy-1,Npz-1))
  zG = np.zeros((nqx,nqy,nqz,Npx-1,Npy-1,Npz-1))
  for i in range(0,Npx-1):
     dx = x[i+1] - x[i]
     xG[:,:,:,i,:,:] = ( (2.*x[i]  + dx)/2. + zeta0/2.*(dx) )[:,None,None,None,None]
  for i in range(0,Npy-1):
     dy = y[i+1] - y[i]
     yG[:,:,:,:,i,:] = ( (2.*y[i]  + dy)/2. + zeta1/2.*(dy) )[None,:,None,None,None]
  for i in range(0,Npz-1):
     dz = z[i+1] - z[i]
     zG[:,:,:,:,:,i] = ( (2.*z[i]  + dz)/2. + zeta2/2.*(dz) )[None,None,:,None,None]
  return xG,yG,zG


def getGlobGrid(main,x,y,z,zeta0,zeta1,zeta2):
#  dx = x[1] - x[0]
#  dy = y[1] - y[0]
#  dz = z[1] - z[0]
  Npx,Npy,Npz = np.shape(x)
  nqx = np.size(zeta0)
  nqy = np.size(zeta1)
  nqz = np.size(zeta2)
  xG = np.zeros((nqx,nqy,nqz,Npx-1,Npy-1,Npz-1))
  yG = np.zeros((nqx,nqy,nqz,Npx-1,Npy-1,Npz-1))
  zG = np.zeros((nqx,nqy,nqz,Npx-1,Npy-1,Npz-1))
  zeta,eta,mu = np.meshgrid(zeta0,zeta1,zeta2,indexing='ij')
  zeta = zeta[:,:,:,None,None,None]
  eta = eta[:,:,:,None,None,None]
  mu = mu[:,:,:,None,None,None]
  x0 = x[None,None,None,0:-1,0:-1,0:-1]
  x1 = x[None,None,None,1:: ,0:-1,0:-1]
  x2 = x[None,None,None,0:-1,1:: ,0:-1]
  x3 = x[None,None,None,1:: ,1:: ,0:-1]
  x4 = x[None,None,None,0:-1,0:-1,1:: ]
  x5 = x[None,None,None,1:: ,0:-1,1:: ]
  x6 = x[None,None,None,0:-1,1:: ,1:: ]
  x7 = x[None,None,None,1:: ,1:: ,1:: ]
  xG = (x1*(eta - 1)*(mu - 1)*(zeta + 1))/8 - (x0*(eta - 1)*(mu - 1)*(zeta - 1))/8 + (x2*(eta + 1)*(mu - 1)*(zeta - 1))/8 - (x3*(eta + 1)*(mu - 1)*(zeta + 1))/8 + (x4*(eta - 1)*(mu + 1)*(zeta - 1))/8 - (x5*(eta - 1)*(mu + 1)*(zeta + 1))/8 - (x6*(eta + 1)*(mu + 1)*(zeta - 1))/8 + (x7*(eta + 1)*(mu + 1)*(zeta + 1))/8
  x0 = y[None,None,None,0:-1,0:-1,0:-1]
  x1 = y[None,None,None,1:: ,0:-1,0:-1]
  x2 = y[None,None,None,0:-1,1:: ,0:-1]
  x3 = y[None,None,None,1:: ,1:: ,0:-1]
  x4 = y[None,None,None,0:-1,0:-1,1:: ]
  x5 = y[None,None,None,1:: ,0:-1,1:: ]
  x6 = y[None,None,None,0:-1,1:: ,1:: ]
  x7 = y[None,None,None,1:: ,1:: ,1:: ]

  yG = (x1*(eta - 1)*(mu - 1)*(zeta + 1))/8 - (x0*(eta - 1)*(mu - 1)*(zeta - 1))/8 + (x2*(eta + 1)*(mu - 1)*(zeta - 1))/8 - (x3*(eta + 1)*(mu - 1)*(zeta + 1))/8 + (x4*(eta - 1)*(mu + 1)*(zeta - 1))/8 - (x5*(eta - 1)*(mu + 1)*(zeta + 1))/8 - (x6*(eta + 1)*(mu + 1)*(zeta - 1))/8 + (x7*(eta + 1)*(mu + 1)*(zeta + 1))/8

  x0 = z[None,None,None,0:-1,0:-1,0:-1]
  x1 = z[None,None,None,1:: ,0:-1,0:-1]
  x2 = z[None,None,None,0:-1,1:: ,0:-1]
  x3 = z[None,None,None,1:: ,1:: ,0:-1]
  x4 = z[None,None,None,0:-1,0:-1,1:: ]
  x5 = z[None,None,None,1:: ,0:-1,1:: ]
  x6 = z[None,None,None,0:-1,1:: ,1:: ]
  x7 = z[None,None,None,1:: ,1:: ,1:: ]

  ZG = (x1*(eta - 1)*(mu - 1)*(zeta + 1))/8 - (x0*(eta - 1)*(mu - 1)*(zeta - 1))/8 + (x2*(eta + 1)*(mu - 1)*(zeta - 1))/8 - (x3*(eta + 1)*(mu - 1)*(zeta + 1))/8 + (x4*(eta - 1)*(mu + 1)*(zeta - 1))/8 - (x5*(eta - 1)*(mu + 1)*(zeta + 1))/8 - (x6*(eta + 1)*(mu + 1)*(zeta - 1))/8 + (x7*(eta + 1)*(mu + 1)*(zeta + 1))/8

  return xG,yG,zG


#def getGlobGrid(x,y,z,zeta0,zeta1,zeta2):
##  dx = x[1] - x[0]
##  dy = y[1] - y[0]
##  dz = z[1] - z[0]
#  Nelx,Nely,Nelz = np.shape(x)
#  order0 = np.size(zeta0)
#  order1 = np.size(zeta1)
#  order2 = np.size(zeta2)
#  quadpoints = np.zeros(3,dtype='int')
#  quadpoints[0] =int( np.size(zeta0) )
#  quadpoints[1] =int( np.size(zeta1) )
#  quadpoints[2] =int( np.size(zeta2) )
#
#  xG = np.zeros(((np.size(x)-1)*np.size(zeta0)))
#  yG = np.zeros(((np.size(y)-1)*np.size(zeta1)))
#  zG = np.zeros(((np.size(z)-1)*np.size(zeta2)))
#
#  for i in range(0,Nelx-1):
#     dx = x[i+1] - x[i]
#     xG[i*quadpoints[0]:(i+1)*quadpoints[0]] = (2.*x[i]  + dx)/2. + zeta0/2.*(dx)
#  for i in range(0,Nely-1):
#     dy = y[i+1] - y[i]
#     yG[i*quadpoints[1]:(i+1)*quadpoints[1]] = (2.*y[i]  + dy)/2. + zeta1/2.*(dy)
#  for i in range(0,Nelz-1):
#     dz = z[i+1] - z[i]
#     zG[i*quadpoints[2]:(i+1)*quadpoints[2]] = (2.*z[i]  + dz)/2. + zeta2/2.*(dz)
#
#  return xG,yG,zG


def getGlobU_scalar(u):
  quadpoints0,quadpoints1,quadpoints2,quadpoints3,Nelx,Nely,Nelz,Nelt = np.shape(u)
  uG = np.zeros((quadpoints0*Nelx,quadpoints1*Nely,quadpoints2*Nelz))
  for i in range(0,Nelx):
    for j in range(0,Nely):
      for k in range(0,Nelz):
          uG[i*quadpoints0:(i+1)*quadpoints0,j*quadpoints1:(j+1)*quadpoints1,k*quadpoints2:(k+1)*quadpoints2] = u[:,:,:,-1,i,j,k,-1]
  return uG


def getGlobU(u):
  nvars,quadpoints0,quadpoints1,quadpoints2,quadpoints3,Nelx,Nely,Nelz,Nelt = np.shape(u)
  uG = np.zeros((nvars,quadpoints0*Nelx,quadpoints1*Nely,quadpoints2*Nelz))
  for i in range(0,Nelx):
    for j in range(0,Nely):
      for k in range(0,Nelz):
        for m in range(0,nvars):
          uG[m,i*quadpoints0:(i+1)*quadpoints0,j*quadpoints1:(j+1)*quadpoints1,k*quadpoints2:(k+1)*quadpoints2] = u[m,:,:,:,-1,i,j,k,-1]
  return uG


def getIC(main,f,x,y,z,zeta3,Npt):
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  Nvars = np.shape(main.a.u)[0]
  ## First perform integration in x
  nt = np.size(zeta3)
  ord_arrx= np.linspace(0,order[0]-1,order[0])
  ord_arry= np.linspace(0,order[1]-1,order[1])
  ord_arrz= np.linspace(0,order[2]-1,order[2])
  ord_arrt= np.linspace(0,order[3]-1,order[3])
  scale =  (2.*ord_arrx[:,None,None,None] + 1.)*(2.*ord_arry[None,:,None,None] + 1.)*(2.*ord_arrz[None,None,:,None] + 1.)*(2.*ord_arrt[None,None,None,:] + 1.)/16.

  U = np.zeros(np.shape(main.a.u))
  U[:,:,:,:,0,:,:,:,0] = f(x,y,z,main)
  for i in range(0,nt):
    for j in range(0,Npt):
      U[:,:,:,:,i,:,:,:,j] =  U[:,:,:,:,0,:,:,:,0]  
      main.a.uFuture[:,:,:,:,:,:,:,j] = U[:,:,:,:,0,:,:,:,0] 
  main.a.a[:] = volIntegrateGlob_tensordot(main,U,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]



def getIC_collocate(main,f,x,y,z,zeta3,Npt):
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  Nvars = np.shape(main.a.u)[0]
  ## First perform integration in x
  nt = np.size(zeta3)
  ord_arrx= np.linspace(0,order[0]-1,order[0])
  ord_arry= np.linspace(0,order[1]-1,order[1])
  ord_arrz= np.linspace(0,order[2]-1,order[2])
  ord_arrt= np.linspace(0,order[3]-1,order[3])
  scale =  (2.*ord_arrx[:,None,None,None] + 1.)*(2.*ord_arry[None,:,None,None] + 1.)*(2.*ord_arrz[None,None,:,None] + 1.)*(2.*ord_arrt[None,None,None,:] + 1.)/16.
  print(np.shape(y))
  U = np.zeros((Nvars,nqx,nqy,nqz,1,Nelx,Nely,Nelz,1))
  U[:,:,:,:,0,:,:,:,0] = f(x,y,z,main)
  for i in range(0,nt):
    for j in range(0,Npt):
      U[:,:,:,:,i,:,:,:,j] =  U[:,:,:,:,0,:,:,:,0]  
      #main.a.uFuture[:,:,:,:,:,:,:,j] = U[:,:,:,:,0,:,:,:,0] 
  main.a.a[:] = volIntegrateGlob_tensordot_collocate(main,U,main.w0_c,main.w1_c,main.w2_c,main.w3_c)*scale[None,:,:,:,:,None,None,None,None]
  main.a.a[:,1::] = 0.


comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()

if 'fsource' in globals():
  pass
else:
  fsource = False
  fsource_mag = []

if 'enriched_ratio' in globals():
  pass
else:
#  enriched_ratio = np.array([2,2,2,1])
  enriched_ratio = np.array([(order[0]+1.)/order[0],(order[1]+1.)/order[1],(order[2]+1.)/order[2],1])
if 'enriched' in globals():
  pass
else:
  enriched = False
  enriched_ratio = 1

if 'turb_str' in globals():
  pass
else:
  turb_str = 'DNS'

if 'shock_capturing' in globals():
  if (mpi_rank == 0): print('shock_capturing set to ' + str(shock_capturing) )
else:
  if (mpi_rank == 0): print('shock_capturing not set, turned off by default' )
  shock_capturing = False

if 'basis_functions_str' in globals():
  pass
else:
  basis_functions_str = 'TensorDot'
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
mpi_rank = comm.Get_rank()
if (mpi_rank == 0):
  print('Running on ' + str(num_processes) + ' Procs')
dx =  (x[-1] - x[0])/Nel[0]
t = 0
if (mpi_rank == 0):
  print('dt*p/(Nt*dx) = ' + str(1.*dt*order[0]/dx[0,0]/order[-1]))
  print('dt*(p/dx)**2*mu = ' + str(dt*order[0]**2/dx[0,0]**2*mu) )
iteration = 0

eqns = equations(eqn_str,schemes,turb_str)
main = variables(Nel,order,quadpoints,eqns,mu,x,y,z,t,et,dt,iteration,save_freq,turb_str,procx,procy,BCs,fsource,source_mag,shock_capturing,mol_str)


if (enriched):
  eqnsEnriched = equations(enriched_eqn_str,enriched_schemes,turb_str)
  mainEnriched = variables(Nel,np.int64(order*enriched_ratio),quadpoints,eqnsEnriched,mu,x,y,z,t,et,dt,iteration,save_freq,turb_str,procx,procy,BCs,source,source_mag,shock_capturing,mol_str)
else:
  mainEnriched = main


#xG,yG,zG = getGlobGrid(main,x[main.sx,main.sy,:],y[main.sx,main.sy,:],z[main.sx,main.sy,:],main.zeta0,main.zeta1,main.zeta2)
#xG,yG,zG = getGlobGrid(main,x,y,z,main.zeta0,main.zeta1,main.zeta2)

#xG2,yG2,zG2 = getGlobGrid2(x,y,z,main.zeta0,main.zeta1,main.zeta2)
#xGc,yGc,zGc = getGlobGrid2(x,y,z,main.zeta0_c,main.zeta1_c,main.zeta2_c)
#xGc2,yGc2,zGc2 = getGlobGrid(x,y,z,main.zeta0_c,main.zeta1_c,main.zeta2_c)

main.basis = basis_class('Legendre',[basis_functions_str])
mainEnriched.basis = main.basis


#getIC_collocate(main,IC_function,xGc[:,:,:,main.sx,main.sy,:],yGc[:,:,:,main.sx,main.sy,:],zGc[:,:,:,main.sx,main.sy,:],main.zeta3,main.Npt)
getIC(main,IC_function,main.xG,main.yG,main.zG,main.zeta3,main.Npt)

reconstructU(main,main.a)

timescheme = timeschemes(time_integration,linear_solver_str,nonlinear_solver_str)
#main.source_hook = source_hook
xG_global = gatherSolScalar(main,main.xG[:,:,:,None,:,:,:,None])
yG_global = gatherSolScalar(main,main.yG[:,:,:,None,:,:,:,None])
zG_global = gatherSolScalar(main,main.zG[:,:,:,None,:,:,:,None])
if (main.mpi_rank == 0):
  xG_global = np.reshape( np.rollaxis(np.rollaxis(np.rollaxis(xG_global[:,:,:,0,:,:,:,0],3,0),4,2),5,4), (Nel[0]*quadpoints[0],Nel[1]*quadpoints[1],Nel[2]*quadpoints[2]) )
  yG_global = np.reshape( np.rollaxis(np.rollaxis(np.rollaxis(yG_global[:,:,:,0,:,:,:,0],3,0),4,2),5,4), (Nel[0]*quadpoints[0],Nel[1]*quadpoints[1],Nel[2]*quadpoints[2]) )
  zG_global = np.reshape( np.rollaxis(np.rollaxis(np.rollaxis(zG_global[:,:,:,0,:,:,:,0],3,0),4,2),5,4), (Nel[0]*quadpoints[0],Nel[1]*quadpoints[1],Nel[2]*quadpoints[2]) )
  if not os.path.exists('Solution'):
     os.makedirs('Solution')
  np.savez('DGgrid',x=xG_global,y=yG_global,z=zG_global)


t0 = time.time()

ord_arrx= np.linspace(0,order[0]-1,order[0])
ord_arry= np.linspace(0,order[1]-1,order[1])
ord_arrz= np.linspace(0,order[2]-1,order[2])
ord_arrt= np.linspace(0,order[3]-1,order[3])
scale =  (2.*ord_arrx[:,None,None,None] + 1.)*(2.*ord_arry[None,:,None,None] + 1.)*(2.*ord_arrz[None,None,:,None] + 1.)*(2.*ord_arrt[None,None,None,:] + 1.)/16.

def entropy_to_conservative(V):
  gamma = 1.4
  U = np.zeros(np.shape(V))
  gamma1 = gamma - 1.
  igamma1 = 1./gamma1
  gmogm1 = gamma*igamma1
  iu4 = 1./V[4]  #- p / rho
  u = -iu4*V[1]
  v = -iu4*V[2]
  w = -iu4*V[3]
  t0 = -0.5*iu4*(V[1]**2 + V[2]**2 + V[3]**2)
  t1 = V[0] - gmogm1 + t0
  t2 =np.exp(-igamma1*np.log(-V[4]) )
  t3 = np.exp(t1)
  U[0] = t2*t3
  H = -iu4*(gmogm1 + t0)
  E = (H + iu4)
  U[1] = U[0]*u
  U[2] = U[0]*v
  U[3] = U[0]*w
  U[4] = U[0]*E
  return U



while (main.t <= main.et + main.dt/2):
  if (main.iteration%main.save_freq == 0):
    reconstructU(main,main.a)
    uG = gatherSolSlab(main,eqns,main.a)
    aG = gatherSolSpectral(main.a.a,main)
    savehook(main)
    if (main.mpi_rank == 0):
      UG = getGlobU(uG)
      #uGF = getGlobU(uG)
      sys.stdout.write('======================================' + '\n')
      sys.stdout.write('wall time = ' + str(time.time() - t0) + '\n' )
      sys.stdout.write('t = ' + str(main.t) +  '\n')
      np.savez('Solution/npsol' + str(main.iteration),U=(UG),a=aG,t=main.t,iteration=main.iteration,order=order)
      sys.stdout.flush()

  timescheme.advanceSol(main,mainEnriched,eqns,timescheme.args)
  #advanceSolImplicit_MG(main,main,eqns)
reconstructU(main,main.a)
uG = gatherSolSlab(main,eqns,main.a)
if (main.mpi_rank == 0):
  print('Final Time = ' + str(time.time() - t0),'Sol Norm = ' + str(np.linalg.norm(uG)) ),
