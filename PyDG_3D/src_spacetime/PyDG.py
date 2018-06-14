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
from block_classes import *
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
num_processes = comm.Get_size()

execfile(PyDG_DIR + '/check_inputdeck.py')

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
  #enriched_ratio = np.array([2,2,2,1])
  enriched_add = np.array([1,1,1,0])
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
basis_args = [basis_functions_str,orthogonal_str]

comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
mpi_rank = comm.Get_rank()
if (mpi_rank == 0):
  print('Running on ' + str(num_processes) + ' Procs')
t = 0
iteration = 0

eqns = equations(eqn_str,schemes,turb_str)
#main = variables(Nel,order,quadpoints,eqns,mu,x,y,z,t,et,dt,iteration,save_freq,turb_str,procx,procy,BCs,fsource,source_mag,shock_capturing,mol_str,basis_args)

regionManager = blockClass(n_blocks,starting_rank,procx,procy,procz,et,dt,save_freq,turb_str,Nel_block,order,eqns)
region_counter = 0
for i in regionManager.mpi_regions_owned:
  regionManager.region.append( variables(regionManager,region_counter,i,Nel_block[i],order,quadpoints,eqns,mu,x_block[i],y_block[i],z_block[i],turb_str,procx[i],procy[i],procz[i],starting_rank[i],BCs[i],fsource,source_mag,shock_capturing,mol_str,basis_args) )
  region_counter += 1
regionConnector(regionManager)
#print('==============')
#print('MPI INFO',regionManager.region[0].mpi_rank,regionManager.region[0].rank_connect[3])
#print('==============')
regionManager.tau = tau
#for i in range(0,regionManager.nblocks):
region_counter = 0
for i in regionManager.mpi_regions_owned:
  region = regionManager.region[region_counter]
  region_counter += 1
  region.x,region.y,region.z = x_block[i],y_block[i],z_block[i]
  vol_min = (np.amin(region.Jdet))**(1./3.)
  CFL = ( np.amin(region.J_edge_det[0])*4. + np.amin(region.J_edge_det[1])*4 + np.amin(region.J_edge_det[2])*4. ) / (np.amin(region.Jdet)*8 )
#  if (mpi_rank == starting_rank[i]):
#    print('dt*p/(Nt*dx) = ' + str(1.*dt*order[0]*CFL/order[-1] ))
#    print('dt*(p/dx)**2*mu = ' + str(dt*order[0]**2*CFL**2*mu/order[-1] ))
  if (enriched):
    eqnsEnriched = eqns#equations(enriched_eqn_str,enriched_schemes,turb_str)
    mainEnriched = variables(Nel,np.int64(order + enriched_add),quadpoints,eqnsEnriched,mu,x,y,z,turb_str,procx,procy,procz,BCs,fsource,source_mag,shock_capturing,mol_str,basis_args)
  else:
    mainEnriched = region 
  
  
  
  getIC(region,IC_function[i],region.xG,region.yG,region.zG,region.zeta3,region.Npt)
  
  reconstructU(region,region.a)
  
  timescheme = timeschemes(regionManager,time_integration,linear_solver_str,nonlinear_solver_str)
  #main.source_hook = source_hook

  xG_global = gatherSolScalar(region,region.xG[:,:,:,None,:,:,:,None])
  yG_global = gatherSolScalar(region,region.yG[:,:,:,None,:,:,:,None])
  zG_global = gatherSolScalar(region,region.zG[:,:,:,None,:,:,:,None])

  Nel = Nel_block[i]
  if (region.mpi_rank == starting_rank[i]):
    xG_global = np.reshape( np.rollaxis(np.rollaxis(np.rollaxis(xG_global[:,:,:,0,:,:,:,0],3,0),4,2),5,4), (Nel[0]*quadpoints[0],Nel[1]*quadpoints[1],Nel[2]*quadpoints[2]) )
    yG_global = np.reshape( np.rollaxis(np.rollaxis(np.rollaxis(yG_global[:,:,:,0,:,:,:,0],3,0),4,2),5,4), (Nel[0]*quadpoints[0],Nel[1]*quadpoints[1],Nel[2]*quadpoints[2]) )
    zG_global = np.reshape( np.rollaxis(np.rollaxis(np.rollaxis(zG_global[:,:,:,0,:,:,:,0],3,0),4,2),5,4), (Nel[0]*quadpoints[0],Nel[1]*quadpoints[1],Nel[2]*quadpoints[2]) )
    if not os.path.exists('Solution'):
       os.makedirs('Solution')
    np.savez('DGgrid_block' + str(i),x=xG_global,y=yG_global,z=zG_global)
  
  
  t0 = time.time()
 
  ord_arrx= np.linspace(0,order[0]-1,order[0])
  ord_arry= np.linspace(0,order[1]-1,order[1])
  ord_arrz= np.linspace(0,order[2]-1,order[2])
  ord_arrt= np.linspace(0,order[3]-1,order[3])
  scale =  (2.*ord_arrx[:,None,None,None] + 1.)*(2.*ord_arry[None,:,None,None] + 1.)*(2.*ord_arrz[None,None,:,None] + 1.)*(2.*ord_arrt[None,None,None,:] + 1.)/16.

while (regionManager.t <= regionManager.et + regionManager.dt/2):
  if (regionManager.iteration%regionManager.save_freq == 0):
    #for z in range(0,regionManager.nblocks):
    region_counter = 0
    savehook(regionManager)
    regionManager.a_norm = globalNorm(regionManager.a,regionManager)

    for region in regionManager.region:
      reconstructU(region,region.a)
      uG = gatherSolSlab(region,eqns,region.a)
      aG = gatherSolSpectral(region.a.a,region)
      if (regionManager.mpi_rank - region.starting_rank == 0):
        UG = getGlobU(uG)
        np.savez('Solution/npsol_block' + str(region.region_number) + '_' + str(regionManager.iteration),U=(UG),a=aG,t=regionManager.t,iteration=regionManager.iteration,order=order)
        sys.stdout.flush()
    if (regionManager.mpi_rank == 0):
      sys.stdout.write('======================================' + '\n')
      sys.stdout.write('wall time = ' + str(time.time() - t0) + '\n' )
      sys.stdout.write('t = ' + str(regionManager.t) +  '\n')
      sys.stdout.flush()
  timescheme.advanceSol(regionManager,eqns,timescheme.args)


regionManager.a_norm = globalNorm(regionManager.a,regionManager)
if (regionManager.mpi_rank == 0):
  print('Final Time = ' + str(time.time() - t0),'Sol Norm = ' + str(regionManager.a_norm))

