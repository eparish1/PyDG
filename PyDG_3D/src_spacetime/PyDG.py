import numpy as np
import os
import sys
from mpi4py import MPI
from init_Classes import *
from solver_classes import *
from DG_functions import reconstructU_tensordot as reconstructU
from DG_functions import volIntegrateGlob_tensordot as volIntegrateGlob
from MPI_functions import gatherSolSlab,gatherSolSpectral,gatherSolScalar,globalSum
#from init_reacting_additions import *
from timeSchemes import *#advanceSol,advanceSolImplicitMG,advanceSolImplicit,advanceSolImplicitPC
import time
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from basis_class import *

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

##============ Initialization stuff. Everything here basically just checks the input deck =========
if (np.size(Nel) != 4):
  if (mpi_rank == 0): 
    print('Error, Nel array should be four dimensional (Nelx,Nely,Nelz,Nelt). Fix Plz. PyDG quitting')
  sys.exit()
if (np.size(order) != 4):
  if (mpi_rank == 0): 
    print('Error, order array should be four dimensional (orderx,ordery,orderz,ordert). Fix Plz. PyDG quitting')
  sys.exit()

if (len(BCs) != 12):
  if (mpi_rank == 0): 
    print('Error, BCs array should have 12 components (leftbc, lefbc_args,rightbc,...). Fix Plz. PyDG quitting')
  sys.exit()

if 'linear_solver_str ' in globals():
  pass
else:
  linear_solver_str = 'GMRes'
  if (mpi_rank == 0):
    print('Setting linear solver to GMRes by default. Ignore if you are using an explicit time scheme')

if 'nonlinear_solver_str ' in globals():
  pass
else:
  nonlinear_solver_str = 'Newton'
  if (mpi_rank == 0):
    print('Setting nonlinear solver to Newton by default. Ignore if you are using an explicit time scheme')

if 'fsource' in globals():
  pass
else:
  fsource = False
  fsource_mag = []

if 'tau' in globals():
  if (mpi_rank == 0):
    print('Setting tau =' + str(tau))
else:
  tau = 0.25
  if (mpi_rank == 0):
    print('Setting tau = 0.25')

if 'enriched_ratio' in globals():
  pass
else:
  #enriched_ratio = np.array([2,2,2,1])
  enriched_add = np.array([4,0,0,0])
  quadpoints_ratio = np.array([2,1,1,1])
#  enriched_ratio = np.array([(order[0]+1.)/order[0],(order[1]+1.)/order[1],(order[2]+1.)/order[2],1])
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
if 'orthogonal_str' in globals():
  pass
else:
  orthogonal_str = False
if 'mol_str' in globals():
  pass
else:
  mol_str = False
if 'source_mag' in globals():
  pass
else:
  source_mag = False

##======================================================

basis_args = [basis_functions_str,orthogonal_str]

comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
mpi_rank = comm.Get_rank()
if (mpi_rank == 0):
  print('Running on ' + str(num_processes) + ' Procs')
t = 0
iteration = 0

eqns = equations(eqn_str,schemes,turb_str)
main = variables(Nel,order,quadpoints,eqns,mu,x,y,z,t,et,dt,iteration,save_freq,turb_str,procx,procy,BCs,fsource,source_mag,shock_capturing,mol_str,basis_args)
main.tau = tau
main.x,main.y,main.z = x,y,z
vol_min = (np.amin(main.Jdet))**(1./3.)
CFL = ( np.amin(main.J_edge_det[0])*4. + np.amin(main.J_edge_det[1])*4 + np.amin(main.J_edge_det[2])*4. ) / (np.amin(main.Jdet)*8 )
if (mpi_rank == 0):
  print('dt*p/(Nt*dx) = ' + str(1.*dt*order[0]*CFL/order[-1] ))
  print('dt*(p/dx)**2*mu = ' + str(dt*order[0]**2*CFL**2*mu/order[-1] ))

if (enriched):
  eqnsEnriched = eqns#equations(enriched_eqn_str,enriched_schemes,turb_str)
  mainEnriched = variables(Nel,np.int64(order + enriched_add),quadpoints*quadpoints_ratio,eqnsEnriched,mu,x,y,z,t,et,dt,iteration,save_freq,turb_str,procx,procy,BCs,fsource,source_mag,shock_capturing,mol_str,basis_args)
else:
  mainEnriched = main



#main.basis = basis_class('Legendre',[basis_functions_str,orthogonal_str])
#mainEnriched.basis = main.basis


#getIC_collocate(main,IC_function,xGc[:,:,:,main.sx,main.sy,:],yGc[:,:,:,main.sx,main.sy,:],zGc[:,:,:,main.sx,main.sy,:],main.zeta3,main.Npt)
getIC(main,IC_function,main.xG,main.yG,main.zG,main.zeta3,main.Npt)

reconstructU(main,main.a)

timescheme = timeschemes(main,time_integration,linear_solver_str,nonlinear_solver_str)
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


#mg_levels =  2#int( np.log(np.amax(main.order))/np.log(2))  
##coarsen = np.int32(2**np.linspace(0,mg_levels-1,mg_levels))
#coarsen = np.linspace(0,mg_levels-1,mg_levels)
#main.mg_classes = []
#main.mg_Rn = []
#main.mg_an = []
#main.mg_b = []
#main.mg_e = []
#mg_iterations = np.array([5,10,15,20])
#mg_omega = np.array([1.,1.,1.,0.8])
#main.mg_args = [mg_levels,mg_iterations,mg_omega]
#for i in range(0,mg_levels):
##  order_coarsen = np.int32(np.fmax(main.order/coarsen[i],1))
#  order_coarsen = np.int32(np.fmax(main.order-coarsen[i],1))
##  quadpoints_coarsen = np.int32(np.fmax(main.quadpoints/(coarsen[i]),1))
#  quadpoints_coarsen = np.int32(np.fmax(main.quadpoints-(coarsen[i]),1))
#
#  main.mg_classes.append( variables(main.Nel,order_coarsen,quadpoints_coarsen,eqns,main.mus,main.x,main.y,main.z,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy,main.BCs,main.fsource,main.source_mag,main.shock_capturing,main.mol_str,main.basis_args) )
#  main.mg_classes[i].basis = main.basis
#  main.mg_Rn.append( np.zeros(np.shape(main.mg_classes[i].RHS)) )
#  main.mg_an.append( np.zeros(np.shape(main.mg_classes[i].a.a) ) )
#  main.mg_b.append( np.zeros(np.shape(main.mg_classes[i].RHS)) )
#  main.mg_e.append(  np.zeros(np.size(main.mg_classes[i].RHS)) )




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
      np.savez('Solution/npsol' + str(main.iteration),U=(UG),a=aG,t=main.t,iteration=main.iteration,order=order,tau=main.tau)
      sys.stdout.flush()

  timescheme.advanceSol(main,mainEnriched,eqns,timescheme.args)
  #advanceSolImplicit_MG(main,main,eqns)
reconstructU(main,main.a)
uG = gatherSolSlab(main,eqns,main.a)
if (main.mpi_rank == 0):
  print('Final Time = ' + str(time.time() - t0),'Sol Norm = ' + str(np.linalg.norm(uG)) ),
