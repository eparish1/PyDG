import numpy as np
import numpy.linalg
from init_Classes import *
import logging
from mpi4py import MPI
from MPI_functions import globalDot
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
try:
  from adolc import *
except:
  if (MPI.COMM_WORLD.Get_rank() == 0):
    logger.warning("adolc not found, can't use adolc automatic differentiation")
try: 
  import adolc
except:
  if (MPI.COMM_WORLD.Get_rank() == 0):
    logger.warning("adolc not found, can't use adolc automatic differentiation")

import scipy.sparse as sp
import time

### FINITE DIFFERENCE SCHEME TO COMPUTE JACOBIANS
# -- Outputs the complete Jacobian for all states over region
# -- Eric, June 13 2018
def computeJacobian_full(regionManager,eqns):
  sz = np.size(regionManager.a)
  J = np.zeros((sz,sz))
  Rstar0 = np.zeros(sz)
  regionManager.getRHS_REGION_OUTER(regionManager,eqns)
  Rstar0[:] = regionManager.RHS[:]
  regionManager.a0[:] = regionManager.a[:]
  eps = 1e-4
  count = 0
  for i in range(0,sz):
    regionManager.a[i] += eps
    regionManager.getRHS_REGION_OUTER(regionManager,eqns)
    J[:,i] = (regionManager.RHS[:] - Rstar0[:])/eps
    regionManager.a[i] -= eps
  return J

### FINITE DIFFERENCE SCHEME TO COMPUTE JACOBIANS FOR POD
def computeJacobian_full_pod(regionManager,eqns):
  sz = np.shape(regionManager.V)[1]
  J = np.zeros((sz,sz))

  Rstar0 = np.zeros(sz)
  regionManager.getRHS_REGION_INNER(regionManager,eqns)
  Rstar0[:] = globalDot(regionManager.V.transpose(),regionManager.RHS[:],regionManager)

  regionManager.a0[:] = regionManager.a[:]
  eps = 1e-4
  count = 0

  eps = 1.e-4
  a0_pod = globalDot(regionManager.V.transpose(),regionManager.a0.flatten(),regionManager)
  a_pod = a0_pod*1.
  for i in range(0,sz):
    a_pod[:] = a0_pod[:]
    a_pod[i] = a0_pod[i] + eps
    regionManager.a[:] = np.dot(regionManager.V,a_pod)
    #regionManager.region[0].a.a[3] = a0[3]
    regionManager.getRHS_REGION_INNER(regionManager,eqns)
    J[:,i] = ( globalDot(regionManager.V.transpose(),regionManager.RHS,regionManager)-Rstar0)/eps
  return J


## Compute brute force the Jacobian for all states over the region
## This is a test function to make sure that computeJacobian_element is right
def computeJacobian_full_element(regionManager,nl_function):
  sz = np.size(regionManager.a)
  J = np.zeros((sz,sz))
  Rstar0 = np.zeros(sz)
  Rstar0[:] = nl_function(regionManager,regionManager.a)
  eps = 1e-4
  count = 0
  for i in range(0,sz):
    regionManager.a[i] += eps
    Rstar = nl_function(regionManager,regionManager.a)
    J[:,i] = (Rstar - Rstar0[:])/eps
    regionManager.a[i] -= eps
  return J


def computeJacobian_full_elementb(regionManager,eqns):
  sz = np.size(regionManager.a)
  J = np.zeros((sz,sz))
  Rstar0 = np.zeros(sz)
  regionManager.getRHS_REGION_INNER_ELEMENT(regionManager,eqns)
  Rstar0[:] = regionManager.RHS[:]
  regionManager.a0[:] = regionManager.a[:]
  eps = 1e-4
  count = 0
  for i in range(0,sz):
    regionManager.a[i] += eps
    regionManager.getRHS_REGION_INNER_ELEMENT(regionManager,eqns)
    J[:,i] = (regionManager.RHS[:] - Rstar0[:])/eps
    regionManager.a[i] -= eps
  return J


def computeJacobian_element(regionManager,nl_function):
  regionManager.a0[:] = regionManager.a[:]
  for region in regionManager.region:
    block_size = region.nvars*np.prod(region.order)
    start_indx = regionManager.solution_start_indx[region.region_counter]
    end_indx = regionManager.solution_end_indx[region.region_counter]

    regionManager.a[:] = regionManager.a0[:]
    Rstar0 = nl_function(regionManager,regionManager.a)
    Rstar0 = np.reshape(Rstar0[start_indx:end_indx],np.shape(region.a.a))

    a0 = region.a.a[:]*1.
    region.PC = np.zeros((region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt))
    eps = 1.e-5
    epsi = 1./eps
    for z in range(0,region.nvars):
      for i in range(0,region.order[0]):
        for j in range(0,region.order[1]):
          for k in range(0,region.order[2]):
            for l in range(0,region.order[3]):
              regionManager.a[:] = regionManager.a0[:] #reset variables for entire region
              region.a.a[z,i,j,k,l] = a0[z,i,j,k,l] + eps  #perturb variable. This propgates to regionManager.a
              #regionManager.getRHS_REGION_INNER_ELEMENT(regionManager,eqns) #get RHS over entire region
              Rstar = nl_function(regionManager,regionManager.a)
              Rstar = np.reshape(Rstar[start_indx:end_indx],np.shape(region.a.a))
              region.PC[:,:,:,:,:,z,i,j,k,l] =  (Rstar - Rstar0)*epsi #fill in entries 
    region.PC = np.reshape(region.PC,(block_size,block_size,region.Npx,region.Npy,region.Npz,region.Npt) )
    region.PCinv = np.linalg.inv(np.rollaxis(np.rollaxis(region.PC,1,6),0,5))
    region.PCinv = np.rollaxis(np.rollaxis(region.PCinv,4,0),5,1)
  regionManager.a[:] = regionManager.a0[:]



def computeJacobian_elementb(regionManager,eqns):
  regionManager.a0[:] = regionManager.a[:]
  for region in regionManager.region:
    regionManager.a[:] = regionManager.a0[:]
    regionManager.getRHS_REGION_INNER_ELEMENT(regionManager,eqns)
    Rstar0 = region.RHS[:]*1.
    a0 = region.a.a[:]*1.
    region.J = np.zeros((region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.nvars,region.order[0],region.order[1],region.order[2],region.order[3],region.Npx,region.Npy,region.Npz,region.Npt))
    eps = 1.e-5
    epsi = 1./eps
    for z in range(0,region.nvars):
      for i in range(0,region.order[0]):
        for j in range(0,region.order[1]):
          for k in range(0,region.order[2]):
            for l in range(0,region.order[3]):
              regionManager.a[:] = regionManager.a0[:] #reset variables for entire region
              region.a.a[z,i,j,k,l] = a0[z,i,j,k,l] + eps  #perturb variable. This propgates to regionManager.a
              regionManager.getRHS_REGION_INNER_ELEMENT(regionManager,eqns) #get RHS over entire region
              region.J[:,:,:,:,:,z,i,j,k,l] =  (region.RHS - Rstar0)*epsi #fill in entries 
    block_size = region.nvars*np.prod(region.order)
    region.J = np.reshape(region.J,(block_size,block_size,region.Npx,region.Npy,region.Npz,region.Npt) )
    region.Jinv = np.linalg.inv(np.rollaxis(np.rollaxis(region.J,1,6),0,5))
    region.Jinv = np.rollaxis(np.rollaxis(region.Jinv,4,0),5,1)
  regionManager.a[:] = regionManager.a0[:]
  #return J#inv



### 
def getR(a,regionManager,eqns):
  regionManager.a = a[:]
  regionManager.getRHS_REGION_OUTER(regionManager,eqns)
  return regionManager.RHS[:]

def testAdolcJacobian1(regionManager,eqns):
  N = np.size(regionManager.a)
  #ax = numpy.array([adouble(0) for n in range(N)])
  ax = adouble(regionManager.a.flatten())
#  main_adolc = variables(main.Nel,main.order,main.quadpoints,eqns,main.mus,main.x,main.y,main.z,main.t,main.et,main.dt,main.iteration,main.save_freq,main.turb_str,main.procx,main.procy,main.BCs,main.fsource,main.source_mag,main.shock_capturing,main.mol_str,main.basis_args,ax.dtype)
  a0 = regionManager.a[:]*1.
  #R = getR(main,main,eqns)
  trace_on(1)
  independent(ax)
  ay = getR(ax,regionManager,eqns)
  dependent(ay)
  trace_off()
  #x = numpy.array([n+1 for n in range(N)])
  # compute jacobian of f at x
  #main.a.a[:] = a0[:]
  x = regionManager.a
#  y = getR(x,main_adolc,eqns)
#  y2 = adolc.function(0,x)
#  assert numpy.allclose(y,y2)
#  options = numpy.array([0,0,0,0],dtype=int)
#  pat = adolc.sparse.jac_pat(0,x,options)
#  J = colpack.sparse_jac_no_repeat(0,x,options)
  J = jacobian(1,x)#main.a.a.flatten())
  return J




def computeBlockJacobian(main,f):
  J = np.zeros((main.nvars,main.order[0],main.order[1],main.order[2],main.order[3],main.nvars,main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  RHS0 = np.zeros(np.shape(main.RHS))
  RHStmp = np.zeros(np.shape(main.RHS))
  Rstar0,R0,Rstar_glob = f(main,main.a.a)
  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  eps = 1.e-5
  epsi = 1./eps
  for z in range(0,main.nvars):
    for i in range(0,main.order[0]):
      for j in range(0,main.order[1]):
        for k in range(0,main.order[2]):
          for l in range(0,main.order[3]):
            main.a.a[:] = a0[:]
            main.a.a[z,i,j,k,l] = a0[z,i,j,k,l] + eps 
            Rstar,RHStmp,Rstar_glob = f(main,main.a.a)
            J[:,:,:,:,:,z,i,j,k,l] =  (Rstar - Rstar0)*epsi 
  #J = np.reshape(J, (main.nvars,main.order[0],main.order[1],main.order[2],main.order[3],main.order[0],main.order[1],main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  #J = np.reshape(J, (main.nvars,main.order[0],main.order[1],main.order[2],main.order[3],main.order[0],main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  #J = np.reshape(J, (main.nvars,main.order[0],main.order[1],main.order[2],main.order[3],main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  #J = np.reshape(J, (main.nvars,main.order[0],main.order[1],main.order[2]*main.order[3],main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  #J = np.reshape(J, (main.nvars,main.order[0],main.order[1]*main.order[2]*main.order[3],main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  #J = np.reshape(J, (main.nvars,main.order[0]*main.order[1]*main.order[2]*main.order[3],main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))

  main.a.a[:] = a0[:]
  return J#inv







def computeJacobianXT(main,f):
  J = np.zeros((main.nvars,main.order[0],main.order[3],main.order[0],main.order[3],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))

  RHS0 = np.zeros(np.shape(main.RHS))
  RHStmp = np.zeros(np.shape(main.RHS))
  Rstar0,R0,Rstar_glob = f(main,main.a.a)
  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  eps = 1.e-3
  epsi = 1./eps
  for i in range(0,main.order[0]):
    for j in range(0,main.order[3]):
      main.a.a[:] = a0[:]
      main.a.a[:,i,:,:,j] = a0[:,i,:,:,j] + eps 
      Rstar,RHStmp,Rstar_glob = f(main,main.a.a)
      J[:,:,:,i,j,:,:,:] = np.rollaxis( ( (Rstar - Rstar0)*epsi )[:,:,:,:,:] , 4 , 2)
  J = np.reshape(J, (main.nvars,main.order[0]*main.order[3],main.order[0],main.order[3],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
  J = np.reshape(J, (main.nvars,main.order[0]*main.order[3],main.order[0]*main.order[3],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
#  Jinv = np.rollaxis(np.rollaxis( np.linalg.inv(np.rollaxis(np.rollaxis(J,2,9),1,8)) , 7 , 1) , 8 , 2)
#  Jinv = np.reshape(Jinv, (main.nvars,main.order[0]*main.order[3],main.order[0],main.order[3],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
#  Jinv = np.reshape(Jinv, (main.nvars,main.order[0],main.order[3],main.order[0],main.order[3],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
#  J = np.reshape(J, (main.nvars,main.order[0]*main.order[3],main.order[0],main.order[3],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
#  J = np.reshape(J, (main.nvars,main.order[0],main.order[3],main.order[0],main.order[3],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
#  J = np.rollaxis(np.rollaxis(J,2,9),1,8)
#  Jinv = np.rollaxis(np.rollaxis(Jinv,2,7),3,7)
  main.a.a[:] = a0[:]
  return J#inv

def computeJacobianX(main,f):
  J = np.zeros((main.nvars,main.order[0],main.nvars,main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  RHS0 = np.zeros(np.shape(main.RHS))
  RHStmp = np.zeros(np.shape(main.RHS))
  Rstar0,R0,Rstar_glob = f(main,main.a.a)
  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  eps = 1.e-2
  for i in range(0,main.nvars):
    for j in range(0,main.order[0]):
      main.a.a[:] = a0[:]
      main.a.a[i,j] = a0[i,j] + eps 
      Rstar,RHStmp,Rstar_glob = f(main,main.a.a)
      J[:,:,i,j] = ( (Rstar - Rstar0)/eps )
  main.a.a[:] = a0[:]
  return J


def computeJacobianY(main,f):
  J = np.zeros((main.nvars,main.order[1],main.nvars,main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  RHS0 = np.zeros(np.shape(main.RHS))
  RHStmp = np.zeros(np.shape(main.RHS))
  Rstar0,R0,Rstar_glob = f(main,main.a.a)
  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  eps = 1e-3
  for i in range(0,main.nvars):
    for j in range(0,main.order[1]):
      main.a.a[:] = a0[:]
      main.a.a[i,:,j] = a0[i,:,j,:,:] + eps 
      Rstar,RHStmp,Rstar_glob = f(main,main.a.a)
      J[:,:,i,j] = np.rollaxis( (Rstar - Rstar0)/eps ,2,1) 
  main.a.a[:] = a0[:]
  return J

def computeJacobianZ(main,f):
  J = np.zeros((main.nvars,main.order[2],main.nvars,main.order[2],main.order[0],main.order[1],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  RHS0 = np.zeros(np.shape(main.RHS))
  RHStmp = np.zeros(np.shape(main.RHS))
  Rstar0,R0,Rstar_glob = f(main.a.a)
  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  eps = 1e-3
  for i in range(0,main.nvars):
    for j in range(0,main.order[2]):
      main.a.a[:] = a0[:]
      main.a.a[i,:,:,j] = a0[i,:,:,j,:] + eps 
      Rstar,RHStmp,Rstar_glob = f(main,main.a.a)
      J[:,:,i,j] = np.rollaxis( (Rstar - Rstar0)/eps ,2 , 1)
  main.a.a[:] = a0[:]
  return J


def computeJacobianT(main,f):
  J = np.zeros((main.nvars,main.order[3],main.nvars,main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
  RHS0 = np.zeros(np.shape(main.RHS))
  RHStmp = np.zeros(np.shape(main.RHS))
  Rstar0,R0,Rstar_glob = f(main,main.a.a)
  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  eps = 1e-3
  for i in range(0,main.nvars):
    for j in range(0,main.order[3]):
      main.a.a[:] = a0[:]
      main.a.a[i,:,:,:,j] = a0[i,:,:,:,j] + eps 
      Rstar,RHStmp,Rstar_glob = f(main,main.a.a)
      J[:,:,i,j] = np.rollaxis( (Rstar - Rstar0)/eps , 4 , 1)
  return J





def computeJacobianX_full(main,f):
  J = np.zeros((main.nvars,main.order[0],main.order[0],main.order[1],main.order[2],main.order[3],main.Nel[0],main.Nel[1],main.Nel[2],main.Nel[3]))
  RHS0 = np.zeros(np.shape(main.RHS))
  RHStmp = np.zeros(np.shape(main.RHS))
  Rstar0,R0,Rstar_glob = f(main.a.a)
  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  eps = 1e-3
  for i in range(0,main.order[0]):
    for j in range(0,main.Nel[0]):
      for k in range(0,main.Nel[1]):
        main.a.a[:] = a0[:]
        main.a.a[:,i,:,:,:,j,k] = a0[:,i,:,:,:,j,k] + eps 
        Rstar,RHStmp,Rstar_glob = f(main.a.a)
        J[:,:,i,:,:,:,j,k] = ( (Rstar - Rstar0)/eps )[:,:,:,:,:,j,k]
  Jinv = np.rollaxis(np.rollaxis( np.linalg.inv(np.rollaxis(np.rollaxis(J,2,10),1,9)) , 8 , 1) , 9 , 2)
  return Jinv


def computeJacobianY_full(main,f):
  J = np.zeros((main.nvars,main.order[0],main.order[1],main.order[1],main.order[2],main.order[3],main.Nel[0],main.Nel[1],main.Nel[2],main.Nel[3]))
  RHS0 = np.zeros(np.shape(main.RHS))
  RHStmp = np.zeros(np.shape(main.RHS))
  Rstar0,R0,Rstar_glob = f(main.a.a)
  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  eps = 1e-3
  for i in range(0,main.order[1]):
    for j in range(0,main.Nel[0]):
      for k in range(0,main.Nel[1]):
        main.a.a[:] = a0[:]
        main.a.a[:,:,i,:,:,j,k] = a0[:,:,i,:,:,j,k] + eps 
        Rstar,RHStmp,Rstar_glob = f(main.a.a)
        J[:,:,:,i,:,:,j,k] = ( (Rstar - Rstar0)/eps )[:,:,:,:,:,j,k]
  Jinv = np.rollaxis(np.rollaxis( np.linalg.inv(np.rollaxis(np.rollaxis(J,3,10),2,9)) , 8 , 2) , 9 , 3)
  return Jinv
