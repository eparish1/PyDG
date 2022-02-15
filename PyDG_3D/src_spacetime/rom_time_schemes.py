import numpy as np
from DG_core import * 
import scipy.linalg
from linear_solvers import *
import matplotlib.pyplot as plt
from mpi4py import MPI
import sys
from MPI_functions import *
from scipy.optimize import least_squares
from init_Classes import variables,equations
from nonlinear_solvers import *
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres,bicgstab
from scipy.sparse.linalg import lgmres
from scipy.optimize import newton_krylov
from scipy.optimize.nonlin import InverseJacobian
from scipy.optimize.nonlin import BroydenFirst, KrylovJacobian
from eos_functions import *
import time
from tensor_products import diffCoeffs
from DG_core import reconstructU_tensordot as reconstructU
from DG_core import volIntegrateGlob_tensordot as volIntegrateGlob
from MPI_functions import gatherSolSlab,gatherSolSpectral,gatherSolScalar,globalSum

#from jacobian_schemes import *
from navier_stokes_entropy import entropy_to_conservative, getEntropyMassMatrix,getEntropyMassMatrix_noinvert
import copy
from copy import deepcopy
try:
  import torch
except:
  pass
def backwardEulerRom(regionManager,regionManagerForJacobianMV,eqns,args):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  regionManager.a[:] = np.dot(regionManager.V,np.dot(regionManager.V.transpose(),regionManager.a))
  regionManager.a0[:] = regionManager.a[:]
  print(np.linalg.norm(regionManager.a))
  def unsteadyResidual(v):
    regionManager.a[:] = v[:]# np.dot(regionManager.V,v[:]) #set current state to be the solution iterate of the Newton Krylov solver
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions
    ## Construct the unsteady residual for Backward Euler in a few steps
    RHS_BE = np.zeros(np.size(regionManager.RHS))
    R1 = np.zeros(np.size(regionManager.RHS))
    R1[:] = regionManager.RHS[:]
    RHS_BE[:] = regionManager.dt*R1
    Rstar = ( regionManager.a[:]*1. - regionManager.a0[:] ) - RHS_BE
    return Rstar.flatten()

  nx,nbasis = np.shape(regionManager.V)
  JV = np.zeros((nx,nbasis) )
  regionManager.an = regionManager.a*1.
  # compute residual
  r = unsteadyResidual(regionManager.a)
  da_norm = 1.
  grad_norm = 1.

  r_norm = globalNorm(r,regionManager)
  an = regionManager.a*1.
  for i in range(0,nbasis):
    eps = 1.e-5
    regionManager.a[:] = an[:] + eps*regionManager.V[:,i]
    Jv_column = unsteadyResidual(regionManager.a[:])
    JV[:,i] = (Jv_column - r[:]) / eps
  regionManager.a[:] = an[:]*1.
  JVr = globalDot(JV.transpose(),r,regionManager)
  grad_norm_0 = np.linalg.norm(JVr)
  grad_norm = grad_norm_0*1.
  if (regionManager.mpi_rank == 0):
    print('Initial Residual = ' + str(r_norm) )

  while (grad_norm/grad_norm_0 > 1e-6 and da_norm >= 1e-6):
    eps = 1e-5
    an = regionManager.a*1.
    for i in range(0,nbasis):
      regionManager.a[:] = an[:] + eps*regionManager.V[:,i]
      Jv_column = unsteadyResidual(regionManager.a[:])
      JV[:,i] = (Jv_column - r[:]) / eps
    ## do global sum to get the dot product

    JSQ = np.dot(JV.transpose(),JV)
    data = regionManager.comm.gather(JSQ,root = 0)
    JSQ_glob = np.zeros(np.shape(JSQ) )
    if (regionManager.mpi_rank == 0):
      for j in range(0,regionManager.num_processes):
        JSQ_glob[:] += data[j]
      for j in range(1,regionManager.num_processes):
        comm.Send(JSQ_glob, dest=j)
    else:
      comm.Recv(JSQ_glob,source=0)
    JVr = globalDot(JV.transpose(),r,regionManager)
    da_pod = np.linalg.solve(JSQ_glob,-JVr)
    da_norm = np.linalg.norm(da_pod)
    grad_norm = np.linalg.norm(JVr)
    #print(np.linalg.norm(da_pod))
    regionManager.a[:] = an[:] + np.dot(regionManager.V,da_pod[:])
    r = unsteadyResidual(regionManager.a)
    r_norm = globalNorm(r,regionManager)
    if (regionManager.mpi_rank == 0):
      print('Residual = ' + str(r_norm), ' Change to solution is ' + str(da_norm), ' Gradient is ' + str(grad_norm), ' Gradient ratio is ' + str(grad_norm/grad_norm_0))

  regionManager.t += regionManager.dt
  regionManager.iteration += 1

## Collocated LSPG with Crank Nicolson 
def crankNicolsonRom(regionManager,eqns,args):
  schemes = ('central','Inviscid')
  eqnsTranspose = equations('Transpose Navier-Stokes',schemes,'DNS',None)
  
  if (regionManager.iteration == 0):
    print('Starting Simulation')
    #a0_pod = globalDot(regionManager.V.transpose(),regionManager.a,regionManager)
    a0 = regionManager.a[:]*1.
    for region in regionManager.region:
      region.a.a[:] = np.sum(region.M[None]*region.a.a[:,None,None,None,None],axis=(5,6,7,8) )
    regionManager.a0[:] = regionManager.a[:]
    a0_pod = globalDot(regionManager.V.transpose(),regionManager.a0,regionManager)
    regionManager.a[:] = np.dot(regionManager.V,a0_pod)#a0[:]
    regionManager.a0[:] = np.dot(regionManager.V,a0_pod) 
    regionManager.jacobian_iteration = 0
  else:
    a0_pod = regionManager.a_pod*1.
  regionManager.a_pod[:] = a0_pod[:]*1.
  regionManager.a0[:] = regionManager.a[:]*1.

  jacobian_update_freq = 1
  ## Get initial RHS
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling collocation
  #R0 = np.zeros(np.size(regionManager.RHS_hyper))
  #R0[:] = regionManager.RHS_hyper[:]
  R0 = np.zeros(np.size(regionManager.RHS[:]))
  R0[:] = regionManager.RHS[:]
  linVar = copy.deepcopy(regionManager.region[0].a)

  def applyJT(v): 
    linVar.a[:] = np.reshape(v,np.shape(linVar.a))
    mass_contrib = np.sum(regionManager.region[0].M[None]*linVar.a[:,None,None,None,None],axis=(5,6,7,8)).flatten()
    linVar.u[:] = reconstructUGeneral_tensordot(regionManager.region[0],linVar.a)
    rx,ry,rz = diffU_tensordot(linVar.a,regionManager.region[0])
    for region in regionManager.region:
      linVar.u[:] = reconstructUGeneral_tensordot(regionManager.region[0],linVar.a)
      linVar.uR[:],linVar.uL[:],linVar.uU[:],linVar.uD[:],linVar.uF[:],linVar.uB[:] = region.basis.reconstructEdgesGeneral(linVar.a,region)

    for region in regionManager.region:
      linVar.uR_edge[:],linVar.uL_edge[:],linVar.uU_edge[:],linVar.uD_edge[:],linVar.uF_edge[:],linVar.uB_edge[:] = sendEdgesGeneralSlab(linVar.uL,linVar.uR,linVar.uD,linVar.uU,linVar.uB,linVar.uF,region,regionManager)
    #regionManager.a[:] = regionManager.a0[:]
    regionManager.a[:] = np.dot(regionManager.V[:],regionManager.a_pod[:])
    regionManager.RHS[:] = 0.
    eqnsTranspose.getRHS(regionManager,eqnsTranspose,[linVar],[rx,ry,rz])
    JTv = mass_contrib - 0.5*regionManager.dt*regionManager.RHS[:]
    VTJTv =  globalDot(regionManager.V.transpose(),JTv,regionManager)
    return VTJTv 

  def unsteadyResidual(v): 
    regionManager.a[:] = v[:]*1.
    regionManager.getRHS_REGION_OUTER(regionManager,eqns)
    ## Construct the unsteady residual for Crank Nicolson in a few steps
    R1 = np.zeros(np.size(R0))
    R1[:] = regionManager.region[0].RHS[:].flatten()
    Rstar = ( regionManager.a[:]*1. - regionManager.a0[:] )# - 0.5*regionManager.dt*(R1 + R0)
    regionManager.a[:] = Rstar[:]*1.
    Rstar = np.sum(regionManager.region[0].M[None]*regionManager.region[0].a.a[:,None,None,None,None],axis=(5,6,7,8)).flatten()
    Rstar -= 0.5*regionManager.dt*(R1 + R0)
    regionManager.a[:] = v[:]
    return Rstar.flatten()*1.

  N = np.size(regionManager.a)
  nx,nbasis = np.shape(regionManager.V[:])
  JV = np.zeros((nx,nbasis) )
  regionManager.an = regionManager.a[:]*1.
  # compute residual
  r = unsteadyResidual(regionManager.a[:])
  r1 = copy.deepcopy(r)
  da_norm = 1.
  ls_iteration = 0
  r_norm = globalNorm(r,regionManager)
  alpha = 1.
  if (regionManager.mpi_rank == 0):
    print('Initial Residual = ' + str(r_norm) )
  r_norm = 1e10
  while (r_norm > 1e-4 and da_norm >= 1e-4):

   JVr2 = applyJT(r)
   regionManager.JVr2 = JVr2

   if (ls_iteration >= 40 or alpha <= 0.01):
     print('Didnt converge, alpha = ' + str(alpha) ,'  iteration count = ' + str(ls_iteration) )
     da_norm = 0.
   else:
    eps = 2e-6
    an = regionManager.a0[:]*1.
#    if (ls_iteration%jacobian_update_freq == 0 and regionManager.iteration%1 == 0):
#      for i in range(0,nbasis):
#        regionManager.a[:] = an[:] + eps*regionManager.V[:,i]
#        Jv_column = unsteadyResidual(regionManager.a[:])
#        JV[:,i] = (Jv_column - r) / (eps)
#      regionManager.JV = JV
#    JV = regionManager.JV 
    ## compute QR of JV
#    Q,R = np.linalg.qr(JV)
#    #now solve the problem R y = -Q^T JVr 
#    da_pod = np.linalg.solve(R,-np.dot(Q.transpose(),r) )
#    print(mpi_rank,np.linalg.norm(r1),np.linalg.norm(r))
    #linVar.a[:] = np.reshape(regionManager.V[:,0],np.shape(linVar.a))
#    JVr = globalDot(JV.transpose(),r,regionManager)
#    regionManager.JVr = JVr
#    regionManager.JV2 = regionManager.RHS[:].flatten()*1.
#    print(np.linalg.norm(JVr2),np.linalg.norm(JVr),np.linalg.norm(JVr2 - JVr))
#    #print(np.linalg.norm(regionManager.RHS),np.linalg.norm(JV[:,0]),np.linalg.norm(r1),np.linalg.norm(r))
#    JSQ = np.dot(JV.transpose(),JV)
#    ## do global sum to get the dot product
#    data = regionManager.comm.gather(JSQ,root = 0)
#    JSQ_glob = np.zeros(np.shape(JSQ) )
#    if (regionManager.mpi_rank == 0):
#      for j in range(0,regionManager.num_processes):
#        JSQ_glob[:] += data[j]
#      for j in range(1,regionManager.num_processes):
#        comm.Send(JSQ_glob, dest=j)
#    else:
#      comm.Recv(JSQ_glob,source=0)
#    da_pod = np.linalg.solve(JSQ_glob,-JVr)
    da_pod = -10.*JVr2
    da_norm = np.linalg.norm(da_pod)
    regionManager.a_pod[:] += da_pod[:]
    regionManager.a[:] = np.dot(regionManager.V[:],regionManager.a_pod[:])
    r = unsteadyResidual(regionManager.a)
    r_norm = globalNorm(r,regionManager)
    ls_iteration += 1
    if (regionManager.mpi_rank == 0):
      print('Residual = ' + str(r_norm), ' Change to solution is ' + str(da_norm))

  regionManager.t += regionManager.dt
  regionManager.iteration += 1

  ### Update the global state for saving
  if (regionManager.iteration%regionManager.save_freq == 0):
    regionManager.a[:] = np.dot(regionManager.V,regionManager.a_pod)

def linesearch(computeResidual,x0,dx,r0norm,alpha):
  rnorm = r0norm * 1.
  drnorm = 1.
  alpha_min = 0.
  alpha_max = alpha*1.

  while (drnorm >= 1e-4):
    rnew = computeResidual(x0+dx*alpha)
    rnewnorm = np.linalg.norm(rnew)
    drnorm = np.abs(rnewnorm - rnorm)
    if rnewnorm <= rnorm:
      alpha = alpha + abs(alpha_max - alpha_min)*0.5 
      alpha_min = alpha*1.
      rnorm = rnewnorm*1.
    else:
      alpha_max = alpha*1.
      alpha = (alpha_min + alpha_max)/2.
    print('Line search, rnorm = ' + str(rnewnorm) , ' alpha = ' + str(alpha))
  return alpha 

def bfgs(regionManager,computeResidual,applyGrad,x0,max_iterations = 30,dx_tol = 1e-3, g_tol = 1e-3):
  iteration_no = 0
  x = copy.deepcopy(x0)
  H = np.eye(np.size(x))
  r = computeResidual(x)
  JTr = applyGrad(x,r)
  JTr_old = copy.deepcopy(JTr)
  rnorm0 = np.linalg.norm(r)
  JTrnorm0 = np.linalg.norm(JTr)
  JTrnorm = JTrnorm0 * 1.
  step = 1.0
  xnorm = np.linalg.norm(x) 
  while  (iteration_no <= max_iterations and  JTrnorm / JTrnorm0 >= g_tol):
    p = np.linalg.solve(H,-JTr)
    step = linesearch(computeResidual,x,p*1.,np.linalg.norm(r),step)
    s = step*p
    x += s
    r = computeResidual(x)
    JTr = applyGrad(x,r)
    y = JTr - JTr_old
    num1 = np.outer(y,y)
    den1 = np.dot(y,s)
    num2 = np.dot(H, np.dot( np.outer(s,s) , H.transpose() ) )
    den2 = np.dot( s.transpose(), np.dot(H,s) )
    H = H + num1/den1 - num2/den2
    rnorm = np.linalg.norm(r)
    JTrnorm = np.linalg.norm(JTr)
    dG = JTr - JTr_old
    #step = np.abs(np.dot(dx,dG))/np.linalg.norm(dG)**2
    JTr_old = JTr*1. 
    print('Residual norm (abs) = ' + str(rnorm)  , 'Residual norm (rel) = ' + str(rnorm/rnorm0) ,  ' Gradient norm (abs) = ' + str(JTrnorm),  ' Gradient norm (rel) = ' + str(JTrnorm/JTrnorm0), ' Correction norm (rel) ' + str(np.linalg.norm(step)/xnorm), 'Step size = ' + str(step)) 
    iteration_no += 1 
  return x


def adaGrad(regionManager,computeResidual,applyGrad,x0,max_iterations = 50,dx_tol = 1e-5, g_tol = 1e-5, dr_tol = 1e-6):
  iteration_no = 0
  x = copy.deepcopy(x0)
  r = computeResidual(x)
  JTr = applyGrad(x,r)
  JTr_old = copy.deepcopy(JTr)
  rnorm0 = np.linalg.norm(r)
  rnorm_old = rnorm0*1.
  JTrnorm0 = np.linalg.norm(JTr)
  JTrnorm = JTrnorm0 * 1.
  step = 0.005#7./regionManager.dt 
  xnorm = np.linalg.norm(x) 
  drnorm = 100.
  velo = 0.
  delta = 1e-7
  rcum = 0.
  while  (iteration_no <= max_iterations):# and drnorm/rnorm0 >= dr_tol):
    rcum += JTr*JTr
    dx = -step / (delta + np.sqrt(rcum))*JTr
    x += dx 
    #velo = alpha_momentum*velo - step*JTr
    #dx = velo*1. 
    #x += dx
    r = computeResidual(x)
    JTr = applyGrad(x,r)
    rnorm = np.linalg.norm(r)
    drnorm = np.abs(rnorm - rnorm_old)
    rnorm_old = rnorm*1.
    JTrnorm = np.linalg.norm(JTr)
    dG = JTr - JTr_old
    #step = np.abs(np.dot(dx,dG))/np.linalg.norm(dG)**2
    JTr_old = JTr*1. 
    print('Iteration = ' + str(iteration_no), ' Residual norm (abs) = ' + str(rnorm)  , 'Residual norm (rel) = ' + str(rnorm/rnorm0) ,  ' dR norm = ' + str(drnorm/rnorm0) , ' Gradient norm (abs) = ' + str(JTrnorm),  ' Gradient norm (rel) = ' + str(JTrnorm/JTrnorm0), ' Correction norm (rel) ' + str(np.linalg.norm(dx)/xnorm), 'Step size = ' + str(step)) 
    iteration_no += 1 
  return x


def steepestDecentMomentum(regionManager,computeResidual,applyGrad,x0,max_iterations = 200,dx_tol = 1e-5, g_tol = 1e-5, dr_tol = 5e-6):
  iteration_no = 0
  x = copy.deepcopy(x0)
  r = computeResidual(x)
  JTr = applyGrad(x,r)
  JTr_old = copy.deepcopy(JTr)
  rnorm0 = np.linalg.norm(r)
  rnorm_old = rnorm0*1.
  JTrnorm0 = np.linalg.norm(JTr)
  JTrnorm = JTrnorm0 * 1.
  step = 5./regionManager.dt 
  xnorm = np.linalg.norm(x) 
  drnorm = 100.
  velo = 0.
  alpha_momentum = 0.95
  while  (iteration_no <= max_iterations and drnorm/rnorm0 >= dr_tol):
    velo = alpha_momentum*velo - step*JTr
    dx = velo*1. 
    x += dx
    xtilde = x + alpha_momentum*velo
    r = computeResidual(xtilde)
    JTr = applyGrad(xtilde,r)
    #r = computeResidual(x)
    #JTr = applyGrad(x,r)
    rnorm = np.linalg.norm(r)
    drnorm = np.abs(rnorm - rnorm_old)
    rnorm_old = rnorm*1.
    JTrnorm = np.linalg.norm(JTr)
    dG = JTr - JTr_old
    #step = np.abs(np.dot(dx,dG))/np.linalg.norm(dG)**2
    JTr_old = JTr*1. 
    print('Iteration = ' + str(iteration_no), ' Residual norm (abs) = ' + str(rnorm)  , 'Residual norm (rel) = ' + str(rnorm/rnorm0) ,  ' dR norm = ' + str(drnorm/rnorm0) , ' Gradient norm (abs) = ' + str(JTrnorm),  ' Gradient norm (rel) = ' + str(JTrnorm/JTrnorm0), ' Correction norm (rel) ' + str(np.linalg.norm(dx)/xnorm), 'Step size = ' + str(step)) 
    iteration_no += 1 
  return x


def steepestDecent(regionManager,computeResidual,applyGrad,x0,max_iterations = 50,dx_tol = 1e-5, g_tol = 1e-5, dr_tol = 1e-6):
  iteration_no = 0
  x = copy.deepcopy(x0)
  r = computeResidual(x)
  JTr = applyGrad(x,r)
  JTr_old = copy.deepcopy(JTr)
  rnorm0 = np.linalg.norm(r)
  rnorm_old = rnorm0*1.
  JTrnorm0 = np.linalg.norm(JTr)
  JTrnorm = JTrnorm0 * 1.
  step = 1./regionManager.dt 
  xnorm = np.linalg.norm(x) 
  drnorm = 100.
  while  (iteration_no <= max_iterations and drnorm/rnorm0 >= dr_tol):
    dx = step*JTr
    x -= dx
    r = computeResidual(x)
    JTr = applyGrad(x,r)
    rnorm = np.linalg.norm(r)
    drnorm = np.abs(rnorm - rnorm_old)
    rnorm_old = rnorm*1.
    JTrnorm = np.linalg.norm(JTr)
    dG = JTr - JTr_old
    step = np.abs(np.dot(dx,dG))/np.linalg.norm(dG)**2
    JTr_old = JTr*1. 
    print('Iteration = ' + str(iteration_no), ' Residual norm (abs) = ' + str(rnorm)  , 'Residual norm (rel) = ' + str(rnorm/rnorm0) ,  ' dR norm = ' + str(drnorm/rnorm0) , ' Gradient norm (abs) = ' + str(JTrnorm),  ' Gradient norm (rel) = ' + str(JTrnorm/JTrnorm0), ' Correction norm (rel) ' + str(np.linalg.norm(dx)/xnorm), 'Step size = ' + str(step)) 
    iteration_no += 1 
  return x


#def gaussNewton():
#  def applyA(v):
#    t1 = np.dot(regionManager.V[rec_stencil_list],v) 
#    t2 = jac_vec(1,ap,t1)
#    #t3 = globalDot(regionManager.V[rec_stencil_list].transpose(),t2,regionManager)
#    return t3
def initRegionManager(regionManager_existing,eqns):
  regionManager = blockClass(regionManager_existing.n_blocks,regionManager_existing.starting_rank,regionManager_existing.procx,regionManager_existing.procy,regionManager_existing.procz,regionManager_existing.et,regionManager_existing.dt,regionManager_existing.save_freq,regionManager_existing.turb_str,regionManager_existing.Nel_block,regionManager_existing.order,eqns,'O')
  region_counter = 0
  for i in regionManager.mpi_regions_owned:
    regionManager.region.append( variables(regionManager,region_counter,i,regionManager.Nel_block[i],regionManager.order,regionManager_existing.region[i].quadpoints,eqns,regionManager_existing.region[i].mu,regionManager_existing.region[i].x,regionManager_existing.region[i].y,regionManager_existing.region[i].z,regionManager.turb_str,regionManager.procx[i],regionManager.procy[i],regionManager.procz[i],regionManager.starting_rank[i],regionManager_existing.region[i].BCs,False, 0,False,False,['TensorDot','False']) )
    region_counter += 1
  return regionManager

def getSensitivities(regionManager,eqns):
  regionManagerPyAdolc = initRegionManager(regionManager,eqns)
  def getVelocity(v):
    regionManagerPyAdolc.a[regionManagerPyAdolc.region[0].rec_stencil_list] = v[:]
    regionManagerPyAdolc.getRHS_REGION_OUTER(regionManagerPyAdolc,eqns)
    return regionManagerPyAdolc.region[0].RHS_hyper.flatten()
  ax = adouble(regionManager.a[regionManager.region[0].rec_stencil_list].flatten())
  tape_no = 1
  trace_on(tape_no)
  independent(ax)
  ay = getVelocity(ax)
  dependent(ay)
  trace_off()
  print('taped tscheme here')
  



## Collocated LSPG with Crank Nicolson 
def crankNicolsonRomCollocation(regionManager,eqns,args):
  schemes = ('central','Inviscid')
  eqnsTranspose = equations('Transpose Navier-Stokes',schemes,'DNS',None)
  #getSensitivities(regionManager,eqns)
  if (regionManager.iteration == 0):
    print('Starting Simulation')
    #a0_pod = globalDot(regionManager.V.transpose(),regionManager.a,regionManager)
    a0 = regionManager.a[:]*1.
    for region in regionManager.region:
      region.a.a[:] = np.sum(region.M[None]*region.a.a[:,None,None,None,None],axis=(5,6,7,8) )
    regionManager.a0[:] = regionManager.a[:]
    a0_pod = globalDot(regionManager.V.transpose(),regionManager.a0,regionManager)
    regionManager.a[:] = np.dot(regionManager.V,a0_pod)#a0[:]
    regionManager.a0[:] = np.dot(regionManager.V,a0_pod) 
    regionManager.jacobian_update_freq = 5
    regionManager.jacobian_iteration = 0
  else:
    a0_pod = regionManager.a_pod*1.
  regionManager.a_pod[:] = a0_pod[:]*1.
  cell_list = regionManager.region[0].cell_list
  rec_stencil_list = regionManager.region[0].rec_stencil_list
  regionManager.a0[rec_stencil_list] = regionManager.a[rec_stencil_list]*1.

  ## Get initial RHS
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling collocation
  #R0 = np.zeros(np.size(regionManager.RHS_hyper))
  #R0[:] = regionManager.RHS_hyper[:]
  R0 = copy.deepcopy(regionManager.region[0].RHS_hyper[:].flatten())
  jacobian_update_freq = 5
  linVar = copy.deepcopy(regionManager.region[0].a)


  def applyJT(v):
    stencil_ijk = regionManager.region[0].stencil_ijk 
    cell_ijk = regionManager.region[0].cell_ijk 
    cell_list = regionManager.region[0].cell_list
    linVar.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = np.reshape(v,np.shape(linVar.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]))
    mass_contrib = np.sum(regionManager.region[0].M[None,:,:,:,:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*linVar.a[:,None,None,None,None,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],axis=(5,6,7,8)).flatten()
    for region in regionManager.region:
      linVar.u[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] =  reconstructUGeneral_tensordot(region,linVar.a[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]])
      rx,ry,rz = diffU_tensordot_sample(linVar.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region,region.Jinv[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]])
      v1,v2,v3,v4,v5,v6 = region.basis.reconstructEdgesGeneral(linVar.a[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]],region)
      linVar.uR[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v1[:]
      linVar.uL[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v2[:]
      linVar.uU[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v3[:]
      linVar.uD[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v4[:]
      linVar.uF[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v5[:]
      linVar.uB[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v6[:]

    for region in regionManager.region:
      linVar.uR_edge[:],linVar.uL_edge[:],linVar.uU_edge[:],linVar.uD_edge[:],linVar.uF_edge[:],linVar.uB_edge[:] = sendEdgesGeneralSlab(linVar.uL,linVar.uR,linVar.uD,linVar.uU,linVar.uB,linVar.uF,region,regionManager)

    regionManager.a[:] = np.dot(regionManager.V[:],regionManager.a_pod[:])
    regionManager.RHS[:] = 0.
    getRHSAdjointHyper(regionManager,eqnsTranspose,[linVar],[rx,ry,rz])
    JTv = mass_contrib.flatten() - 0.5*regionManager.dt*regionManager.region[0].RHS_hyper[:].flatten()
    VTJTv =  globalDot(regionManager.V[cell_list,:].transpose(),JTv,regionManager)
    return VTJTv 


  def local_getRhsAdjoint(v0,v):
      cell_ijk = regionManager.region[0].cell_ijk
      stencil_ijk = regionManager.region[0].stencil_ijk
      linVar.a[:] = 0.
      linVar.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = np.reshape(v,np.shape(linVar.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]))
      mass_contrib = np.sum(regionManager.region[0].M[None,:,:,:,:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*linVar.a[:,None,None,None,None,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],axis=(5,6,7,8)).flatten()
      
      for region in regionManager.region:
        linVar.u[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] =  reconstructUGeneral_tensordot(region,linVar.a[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]])
        rx,ry,rz = diffU_tensordot_sample(linVar.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],region,region.Jinv[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0]])
        v1,v2,v3,v4,v5,v6 = region.basis.reconstructEdgesGeneral(linVar.a[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]],region)
        linVar.uR[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v1[:]
        linVar.uL[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v2[:]
        linVar.uU[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v3[:]
        linVar.uD[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v4[:]
        linVar.uF[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v5[:]
        linVar.uB[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v6[:]

      for region in regionManager.region:
        linVar.uR_edge[:],linVar.uL_edge[:],linVar.uU_edge[:],linVar.uD_edge[:],linVar.uF_edge[:],linVar.uB_edge[:] = sendEdgesGeneralSlab(linVar.uL,linVar.uR,linVar.uD,linVar.uU,linVar.uB,linVar.uF,region,regionManager)
      regionManager.a[regionManager.region[0].rec_stencil_list] = np.dot(regionManager.V[regionManager.region[0].rec_stencil_list],v0)
      for region in regionManager.region:
        cell_ijk = region.cell_ijk
        stencil_ijk = region.viscous_stencil_ijk
        region.a.u_hyper_stencil =  reconstructUGeneral_tensordot(region,region.a.a[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]])
        region.a.u_hyper_cell =  reconstructUGeneral_tensordot(region,region.a.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]])
        v1,v2,v3,v4,v5,v6 = region.basis.reconstructEdgesGeneral(region.a.a[:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]],region)
        region.a.uR[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v1[:]
        region.a.uL[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v2[:]
        region.a.uU[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v3[:]
        region.a.uD[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v4[:]
        region.a.uF[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v5[:]
        region.a.uB[:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]] = v6[:]
        stencil_ijk = region.stencil_ijk

      for region in regionManager.region:
        region.a.uR_edge[:],region.a.uL_edge[:],region.a.uU_edge[:],region.a.uD_edge[:],region.a.uF_edge[:],region.a.uB_edge[:] = sendEdgesGeneralSlab(region.a.uL,region.a.uR,region.a.uD,region.a.uU,region.a.uB,region.a.uF,region,regionManager)
      
      regionManager.RHS[:] = 0.
      getRHSAdjointHyper(regionManager,eqnsTranspose,[linVar],[rx,ry,rz])
      return  mass_contrib.flatten()*1., regionManager.region[0].RHS_hyper[:].flatten()*1.


  def local_getRhsAdjoint_AD(v0,v):
      cell_ijk = regionManager.region[0].cell_ijk
      stencil_ijk = regionManager.region[0].stencil_ijk
      linVar.a[:] = 0.
      linVar.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] = np.reshape(v,np.shape(linVar.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]))
      mass_contrib = np.sum(regionManager.region[0].M[None,:,:,:,:,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]]*linVar.a[:,None,None,None,None,:,:,:,:,stencil_ijk[5][0],stencil_ijk[6][0],stencil_ijk[7][0],stencil_ijk[8][0]],axis=(5,6,7,8)).flatten()
      v0_phys = np.dot(regionManager.V[regionManager.region[0].rec_stencil_list],v0)
      JTf = vec_jac(1,v0_phys.flatten(),v)
      return  mass_contrib.flatten()*1., JTf#regionManager.region[0].RHS_hyper[:].flatten()*1.




  def applyWindowedJT(v0,v):
    vw = np.reshape(v,(numStepsInWindow,np.size(regionManager.region[0].cell_list)) )
    v0w = np.reshape(v0,(numStepsInWindow,np.size(regionManager.a_pod)) )

    VTJTvG = np.zeros((numStepsInWindow,np.size(regionManager.a_pod)) )
    for i in range(0,numStepsInWindow):
      Mv,fv = local_getRhsAdjoint(v0w[i,:],vw[i,:])
      JTv = Mv.flatten() - 0.5*regionManager.dt*fv
      VTJTvG[i,:] +=  globalDot(regionManager.V[cell_list,:].transpose(),JTv,regionManager)
      if (i + 1 < numStepsInWindow):
        Mvn,fvn = local_getRhsAdjoint(v0w[i+1,:],vw[i+1,:])
        JTv = -Mvn.flatten() - 0.5*regionManager.dt*fvn
        VTJTvG[i,:] +=  globalDot(regionManager.V[regionManager.region[0].cell_list,:].transpose(), JTv, regionManager) 
    return VTJTvG.flatten()



  def applyWindowedJT_AD(v0,v):
    vw = np.reshape(v,(numStepsInWindow,np.size(regionManager.region[0].cell_list)) )
    v0w = np.reshape(v0,(numStepsInWindow,np.size(regionManager.a_pod)) )

    VTJTvG = np.zeros((numStepsInWindow,np.size(regionManager.a_pod)) )
    for i in range(0,numStepsInWindow):
      Mv,fv = local_getRhsAdjoint_AD(v0w[i,:],vw[i,:])
      JTv = Mv.flatten() - 0.5*regionManager.dt*fv
      #print(np.shape(JTv),np.shape(Mv.flatten()),np.size(cell_list))
      VTJTvG[i,:] +=  globalDot(regionManager.V[regionManager.region[0].rec_stencil_list,:].transpose(),JTv,regionManager)
      if (i + 1 < numStepsInWindow):
        Mvn,fvn = local_getRhsAdjoint_AD(v0w[i+1,:],vw[i+1,:])
        JTv = -Mvn.flatten() - 0.5*regionManager.dt*fvn
        VTJTvG[i,:] +=  globalDot(regionManager.V[regionManager.region[0].rec_stencil_list,:].transpose(), JTv, regionManager) 
    return VTJTvG.flatten()
  
  def windowedResidualPod(v):
    v = np.reshape(v,(numStepsInWindow,np.size(regionManager.a_pod)))
    a_old = regionManager.a0[cell_list]*1.
    Rold = R0*1.
    Rg = np.zeros((numStepsInWindow,np.size(regionManager.region[0].cell_list)),dtype=regionManager.a.dtype)
    cell_ijk = regionManager.region[0].cell_ijk 
    for i in range(0,numStepsInWindow):
      regionManager.region[0].RHS_hyper[:] = 0.
      regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list] , v[i,:])
      regionManager.getRHS_REGION_OUTER(regionManager,eqns)
      ## Construct the unsteady residual for Crank Nicolson in a few steps
      #R1 = 1.*regionManager.region[0].RHS_hyper[:].flatten()
      ## compute the residual at the sample points
      Rstar = ( regionManager.a[cell_list]*1. - a_old) 
      Rstar = np.reshape(Rstar,np.shape(regionManager.region[0].a.a[:,None,None,None,None,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]) ) 
      Rstar = np.sum(regionManager.region[0].M[None,:,:,:,:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*Rstar,axis=(5,6,7,8)).flatten()
      Rstar -= 0.5*regionManager.dt*(regionManager.region[0].RHS_hyper[:].flatten() + Rold)
      Rg[i,:] = Rstar*1.
      Rold = regionManager.region[0].RHS_hyper[:].flatten()*1.
      a_old =  np.dot(regionManager.V[cell_list] , v[i,:])
    return Rg.flatten()
      

  def computeResidualPOD(v):
    return unsteadyResidual(np.dot(regionManager.V[rec_stencil_list],v[:]) )

  def unsteadyResidual(v):
    regionManager.a[rec_stencil_list] = v[:]
    regionManager.getRHS_REGION_OUTER(regionManager,eqns)
    cell_ijk = regionManager.region[0].cell_ijk 
    ## Construct the unsteady residual for Crank Nicolson in a few steps
    R1 = np.zeros(np.size(R0))
    R1[:] = regionManager.region[0].RHS_hyper[:].flatten()
    ## compute the residual at the sample points
    Rstar = ( regionManager.a[cell_list]*1. - regionManager.a0[cell_list] ) 
    regionManager.a[cell_list] = Rstar[:]*1.
    Rstar = np.sum(regionManager.region[0].M[None,:,:,:,:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*regionManager.region[0].a.a[:,None,None,None,None,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],axis=(5,6,7,8)).flatten()
    Rstar -= 0.5*regionManager.dt*(R1 + R0)
    regionManager.a[rec_stencil_list] = v[:]
    return Rstar.flatten()

  def initEuler(a_pod):
    a_pod_w = np.zeros((numStepsInWindow,np.size(regionManager.a_pod)))
    for i in range(0,numStepsInWindow):
      regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list],a_pod)
      regionManager.getRHS_REGION_OUTER(regionManager,eqns)
      a_pod = a_pod + regionManager.dt*globalDot(regionManager.V[cell_list].transpose(),regionManager.region[0].RHS_hyper.flatten(),regionManager)
      a_pod_w[i,:] = a_pod
    return a_pod_w

  N = np.size(regionManager.a)
  nx,nbasis = np.shape(regionManager.V[cell_list])
  JV = np.zeros((nx,nbasis) )
  regionManager.an = regionManager.a[rec_stencil_list]*1.
  method = 'SD'
  numStepsInWindow = 2
  if method == 'SD':
    #a0_pod = np.repeat(regionManager.a_pod[None,:],numStepsInWindow,axis=0)
    a0_pod = initEuler(regionManager.a_pod)
    a_pod = steepestDecentMomentum(regionManager,windowedResidualPod,applyWindowedJT,a0_pod.flatten())
    a_pod = np.reshape(a_pod,(numStepsInWindow,np.size(regionManager.a_pod)))
    regionManager.a_pod = a_pod[-1,:]
    regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list],regionManager.a_pod[:])
  if method == 'GN':
    if (regionManager.mpi_rank == 0):
      print('Initial Residual = ' + str(r_norm) )
    while (r_norm > 1e-4 and da_norm >= 5e-4):
     if (ls_iteration >= 20 or alpha <= 0.01):
       print('Didnt converge, alpha = ' + str(alpha) ,'  iteration count = ' + str(ls_iteration) )
       da_norm = 0.
     else:
      
      eps = 1e-5
      an = regionManager.a[rec_stencil_list]*1.
      if (ls_iteration%jacobian_update_freq == 0 and regionManager.iteration%1 == 0):
       for i in range(0,nbasis):
        regionManager.a[rec_stencil_list] = an[:] + eps*regionManager.V[rec_stencil_list,i]
        Jv_column = unsteadyResidual(regionManager.a[rec_stencil_list])
        JV[:,i] = (Jv_column - r[:]) / eps
       regionManager.JV = JV
      JV = regionManager.JV 
      ## compute QR of JV
      Q,R = np.linalg.qr(JV)
      #now solve the problem R y = -Q^T JVr 
      da_pod = np.linalg.solve(R,-np.dot(Q.transpose(),r) )
  #    JSQ = np.dot(JV.transpose(),JV)
      ## do global sum to get the dot product
  #    data = regionManager.comm.gather(JSQ,root = 0)
  #    JSQ_glob = np.zeros(np.shape(JSQ) )
  #    if (regionManager.mpi_rank == 0):
  #      for j in range(0,regionManager.num_processes):
  #        JSQ_glob[:] += data[j]
  #      for j in range(1,regionManager.num_processes):
  #        comm.Send(JSQ_glob, dest=j)
  #    else:
  #      comm.Recv(JSQ_glob,source=0)
  #    da_pod = np.linalg.solve(JSQ_glob,-JVr)
  #    JTr =  applyJT(r)
  #    regionManager.JTr = JTr
  #    regionManager.JTr2 = JTr2
  #    print(np.linalg.norm(JTr2),np.linalg.norm(JTr),np.linalg.norm(JTr2 - JTr))
  #    da_pod = -250.*JTr
      da_norm = np.linalg.norm(da_pod)
      regionManager.a_pod[:] += da_pod[:]
      regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list],regionManager.a_pod[:])
      r = unsteadyResidual(regionManager.a[rec_stencil_list])
      r_norm = globalNorm(r,regionManager)
      ls_iteration += 1
      if (regionManager.mpi_rank == 0):
        print('Residual = ' + str(r_norm), ' Change to solution is ' + str(da_norm))
  regionManager.t += regionManager.dt*numStepsInWindow
  regionManager.iteration += 1*numStepsInWindow

  ### Update the global state for saving
  if (regionManager.iteration%regionManager.save_freq == 0):
    #regionManager.a_pod[:] = np.dot(regionManager.V[cell_list].transpose(),regionManager.a[cell_list] )
    regionManager.a[:] = np.dot(regionManager.V,regionManager.a_pod)








## Collocated LSPG with Crank Nicolson 
def crankNicolsonRomScipy(regionManager,eqns,args):
  schemes = ('central','Inviscid')
  eqnsTranspose = equations('Transpose Navier-Stokes',schemes,'DNS',None)
  
  if (regionManager.iteration == 0):
    print('Starting Simulation')
    #a0_pod = globalDot(regionManager.V.transpose(),regionManager.a,regionManager)
    a0 = regionManager.a[:]*1.
    for region in regionManager.region:
      region.a.a[:] = np.sum(region.M[None]*region.a.a[:,None,None,None,None],axis=(5,6,7,8) )
    regionManager.a0[:] = regionManager.a[:]
    a0_pod = globalDot(regionManager.V.transpose(),regionManager.a0,regionManager)
    regionManager.a[:] = np.dot(regionManager.V,a0_pod)#a0[:]
    regionManager.a0[:] = np.dot(regionManager.V,a0_pod) 
    regionManager.jacobian_iteration = 0
  else:
    a0_pod = regionManager.a_pod*1.
  regionManager.a_pod[:] = a0_pod[:]*1.
  regionManager.a0[:] = regionManager.a[:]*1.

  jacobian_update_freq = 1
  ## Get initial RHS
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling collocation
  #R0 = np.zeros(np.size(regionManager.RHS_hyper))
  #R0[:] = regionManager.RHS_hyper[:]
  R0 = np.zeros(np.size(regionManager.RHS[:]))
  R0[:] = regionManager.RHS[:]
  linVar = copy.deepcopy(regionManager.region[0].a)


  def unsteadyResidual(v): 
    regionManager.a[:] = np.dot(regionManager.V,v[:])
    regionManager.getRHS_REGION_OUTER(regionManager,eqns)
    ## Construct the unsteady residual for Crank Nicolson in a few steps
    R1 = np.zeros(np.size(R0))
    R1[:] = regionManager.region[0].RHS[:].flatten()
    Rstar = ( regionManager.a[:]*1. - regionManager.a0[:] )# - 0.5*regionManager.dt*(R1 + R0)
    regionManager.a[:] = Rstar[:]*1.
    Rstar = np.sum(regionManager.region[0].M[None]*regionManager.region[0].a.a[:,None,None,None,None],axis=(5,6,7,8)).flatten()
    Rstar -= 0.5*regionManager.dt*(R1 + R0)
    #regionManager.a[:] = v[:]
    return Rstar.flatten()*1.


  def create_mv(v):
    def applyJ(v):
      eps = 1e-3
      R0 = unsteadyResidual(a0_pod)
      R1 = unsteadyResidual(a0_pod + eps*v)
      Jv = (R1 - R0)/eps
      return Jv 

    def applyJT(v): 
      linVar.a[:] = np.reshape(v,np.shape(linVar.a))
      mass_contrib = np.sum(regionManager.region[0].M[None]*linVar.a[:,None,None,None,None],axis=(5,6,7,8)).flatten()
      linVar.u[:] = reconstructUGeneral_tensordot(regionManager.region[0],linVar.a)
      rx,ry,rz = diffU_tensordot(linVar.a,regionManager.region[0])
      for region in regionManager.region:
        linVar.u[:] = reconstructUGeneral_tensordot(regionManager.region[0],linVar.a)
        linVar.uR[:],linVar.uL[:],linVar.uU[:],linVar.uD[:],linVar.uF[:],linVar.uB[:] = region.basis.reconstructEdgesGeneral(linVar.a,region)
  
      for region in regionManager.region:
        linVar.uR_edge[:],linVar.uL_edge[:],linVar.uU_edge[:],linVar.uD_edge[:],linVar.uF_edge[:],linVar.uB_edge[:] = sendEdgesGeneralSlab(linVar.uL,linVar.uR,linVar.uD,linVar.uU,linVar.uB,linVar.uF,region,regionManager)
      #regionManager.a[:] = regionManager.a0[:]
      regionManager.a[:] = np.dot(regionManager.V[:],regionManager.a_pod[:])
      regionManager.RHS[:] = 0.
      eqnsTranspose.getRHS(regionManager,eqnsTranspose,[linVar],[rx,ry,rz])
      JTv = mass_contrib - 0.5*regionManager.dt*regionManager.RHS[:]
      VTJTv =  globalDot(regionManager.V.transpose(),JTv,regionManager)
      return VTJTv 

    mdim = np.size(regionManager.a)
    ndim = np.shape(regionManager.V)[1]
    return  LinearOperator((mdim,ndim),matvec=applyJ,rmatvec=applyJT)

  N = np.size(regionManager.a)
  nx,nbasis = np.shape(regionManager.V[:])
  JV = np.zeros((nx,nbasis) )
  regionManager.an = regionManager.a[:]*1.
  # compute residual
#  r = unsteadyResidual(regionManager.a[:])
#  r1 = copy.deepcopy(r)
#  da_norm = 1.
#  ls_iteration = 0
#  r_norm = globalNorm(r,regionManager)
#  alpha = 1.
#  if (regionManager.mpi_rank == 0):
#    print('Initial Residual = ' + str(r_norm) )
#  r_norm = 1e10
  res_1 = least_squares(unsteadyResidual, a0_pod,ftol=1e-4,xtol=1e-8,jac=create_mv,method='dogbox',verbose=2)

  while (r_norm > 1e-6 and da_norm >= 1e-6):

   JVr2 = applyJT(r)
   regionManager.JVr2 = JVr2

   if (ls_iteration >= 400 or alpha <= 0.01):
     print('Didnt converge, alpha = ' + str(alpha) ,'  iteration count = ' + str(ls_iteration) )
     da_norm = 0.
   else:
    eps = 2e-6
    an = regionManager.a0[:]*1.
#    if (ls_iteration%jacobian_update_freq == 0 and regionManager.iteration%1 == 0):
#      for i in range(0,nbasis):
#        regionManager.a[:] = an[:] + eps*regionManager.V[:,i]
#        Jv_column = unsteadyResidual(regionManager.a[:])
#        JV[:,i] = (Jv_column - r) / (eps)
#      regionManager.JV = JV
#    JV = regionManager.JV 
    ## compute QR of JV
#    Q,R = np.linalg.qr(JV)
#    #now solve the problem R y = -Q^T JVr 
#    da_pod = np.linalg.solve(R,-np.dot(Q.transpose(),r) )
#    print(mpi_rank,np.linalg.norm(r1),np.linalg.norm(r))
    #linVar.a[:] = np.reshape(regionManager.V[:,0],np.shape(linVar.a))
#    JVr = globalDot(JV.transpose(),r,regionManager)
#    regionManager.JVr = JVr
#    regionManager.JV2 = regionManager.RHS[:].flatten()*1.
#    print(np.linalg.norm(JVr2),np.linalg.norm(JVr),np.linalg.norm(JVr2 - JVr))
#    #print(np.linalg.norm(regionManager.RHS),np.linalg.norm(JV[:,0]),np.linalg.norm(r1),np.linalg.norm(r))
#    JSQ = np.dot(JV.transpose(),JV)
#    ## do global sum to get the dot product
#    data = regionManager.comm.gather(JSQ,root = 0)
#    JSQ_glob = np.zeros(np.shape(JSQ) )
#    if (regionManager.mpi_rank == 0):
#      for j in range(0,regionManager.num_processes):
#        JSQ_glob[:] += data[j]
#      for j in range(1,regionManager.num_processes):
#        comm.Send(JSQ_glob, dest=j)
#    else:
#      comm.Recv(JSQ_glob,source=0)
#    da_pod = np.linalg.solve(JSQ_glob,-JVr)
    da_pod = -5.*JVr2
    da_norm = np.linalg.norm(da_pod)
    regionManager.a_pod[:] += da_pod[:]
    regionManager.a[:] = np.dot(regionManager.V[:],regionManager.a_pod[:])
    r = unsteadyResidual(regionManager.a)
    r_norm = globalNorm(r,regionManager)
    ls_iteration += 1
    if (regionManager.mpi_rank == 0):
      print('Residual = ' + str(r_norm), ' Change to solution is ' + str(da_norm))

  regionManager.t += regionManager.dt
  regionManager.iteration += 1

  ### Update the global state for saving
  if (regionManager.iteration%regionManager.save_freq == 0):
    regionManager.a[:] = np.dot(regionManager.V,regionManager.a_pod)



def ls_encoder(u,regionManager):
  def encoder_residual(xhat):
    uhat = decoder(xhat,regionManager)
    return (u - uhat)**2
  K = 10
  xhat = np.zeros(K)
  xhat = scipy.optimize.least_squares(encoder_residual,xhat.flatten(),verbose=2,ftol=1e-7, xtol=1e-7, gtol=1e-7).x
  return xhat

def least_squares_reconstruction(u,Phi,P):
  PPhi = Phi[P]
  LHS = np.dot(PPhi.transpose(),PPhi)
  RHS = np.dot(PPhi.transpose(),u)
  xhat = np.linalg.solve(LHS,RHS)
  u = np.dot(Phi,xhat)
  return u


def decoder(xhat,regionManager):
  U_ref = 0.
  U_scale = 1.
  if (np.size(np.shape(xhat)) == 1):
    return (regionManager.manifold_model.decoder(torch.tensor(xhat[None])).detach().numpy()*U_scale + U_ref)[0]
  else:
    return regionManager.manifold_model.decoder(torch.tensor(xhat)).detach().numpy()*U_scale + U_ref

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


## Collocated LSPG with Crank Nicolson on a manifold w/ pytorch model 
def crankNicolsonManifoldRomCollocation(regionManager,eqns,args):
  if (regionManager.iteration == 0):
    print('Starting Simulation')
    a_stencil = regionManager.a[regionManager.region[0].stencil_list]
    regionManager.ahat = ls_encoder(a_stencil,regionManager) 
  cell_ijk = regionManager.region[0].cell_ijk

  regionManager.a[regionManager.region[0].stencil_list] = decoder(regionManager.ahat,regionManager)
  a0 = regionManager.region[0].a.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*1.
  ## Get initial RHS
  regionManager.getRHS_REGION_INNER_QDEIM(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling collocation
  R0 = np.zeros(np.size(regionManager.region[0].RHS_hyper[:]))
  R0[:] = 1.*regionManager.region[0].RHS_hyper.flatten()
  def unsteadyResidual(v): 
    regionManager.a[regionManager.region[0].stencil_list] = decoder(v,regionManager)
    regionManager.getRHS_REGION_INNER_QDEIM(regionManager,eqns)
    R1 = np.zeros(np.size(R0))
    R1[:] = regionManager.region[0].RHS_hyper[:].flatten()
    Rstar = ( regionManager.region[0].a.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] - a0).flatten()
    Rstar -= 0.5*regionManager.dt*(R1 + R0)
    return Rstar.flatten()*1.

  K = 10
  numStepsInWindow = regionManager.numStepsInWindow
  def unsteadyResidualWindow(v):
    v = np.reshape(v,(numStepsInWindow,K))
    a_w = decoder(v,regionManager)
    residual = np.zeros(0)
    R1 = np.zeros(np.size(R0))
    Rold = R0*1.
    aold = a0*1.
    for i in range(0,numStepsInWindow):
      regionManager.a[regionManager.region[0].stencil_list] = a_w[i] 
      regionManager.getRHS_REGION_INNER_QDEIM(regionManager,eqns)
      R1[:] = regionManager.region[0].RHS_hyper[:].flatten()
      Rstar = ( regionManager.region[0].a.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] - aold).flatten()
      Rstar -= 0.5*regionManager.dt*(R1 + Rold)
      aold = regionManager.region[0].a.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*1.
      Rold = R1*1.
      residual = np.append(residual,Rstar.flatten())
    return residual 
  N = np.size(regionManager.region[0].cell_list)
  #J_sparsity = np.zeros((N*numStepsInWindow,K*numStepsInWindow))
  #J_sparsity[0:N,0:K] = 1
  #for i in range(1,numStepsInWindow):
  #  J_sparsity[N*i:N*(i+1),K*(i-1):K*(i+1)] = 1

  ahat0 = np.zeros((numStepsInWindow,K))
  ahat0[:] = regionManager.ahat[None]
  res_1 = scipy.optimize.least_squares(unsteadyResidualWindow, ahat0.flatten(),diff_step=1e-6,ftol=1e-6,xtol=1e-6,gtol=1e-6,verbose=2,jac_sparsity=regionManager.J_sparsity)
  ahatW = np.reshape(res_1.x,(numStepsInWindow,K))
  regionManager.ahat = ahatW[-1]
  regionManager.a[regionManager.region[0].stencil_list] = decoder(regionManager.ahat,regionManager)

  #### Update the global state for saving
  if (regionManager.iteration%regionManager.save_freq == 0):
   for i in range(0,numStepsInWindow):
    regionManager.t += regionManager.dt
    regionManager.iteration += 1
    regionManager.a[regionManager.region[0].stencil_list] = decoder(ahatW[i],regionManager)
    regionManager.a[:] = least_squares_reconstruction(regionManager.a[regionManager.region[0].stencil_list],regionManager.region[0].Phi,regionManager.region[0].stencil_list)
    for region in regionManager.region:
      reconstructU(region,region.a)
      uG = gatherSolSlab(region,eqns,region.a)
      aG = gatherSolSpectral(region.a.a,region)
      rG = gatherSolSpectral(region.RHS,region)
      if (regionManager.mpi_rank - region.starting_rank == 0):
        UG = getGlobU(uG)
        np.savez('Solution/npsol_block' + str(region.region_number) + '_' + str(regionManager.iteration),U=(UG),a=aG,RHS=rG,t=regionManager.t,iteration=regionManager.iteration)
        sys.stdout.flush()
  #regionManager.t += regionManager.dt*numStepsInWindow
  #regionManager.iteration += 1*numStepsInWindow





## Collocated LSPG with Crank Nicolson on a manifold w/ pytorch model 
def crankNicolsonManifoldRomCollocationb(regionManager,eqns,args):
  if (regionManager.iteration == 0):
    print('Starting Simulation')
    a_stencil = regionManager.a[regionManager.region[0].stencil_list]
    regionManager.ahat = ls_encoder(a_stencil,regionManager) 
  cell_ijk = regionManager.region[0].cell_ijk

  regionManager.a[regionManager.region[0].stencil_list] = decoder(regionManager.ahat,regionManager)
  a0 = regionManager.region[0].a.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*1.
  ## Get initial RHS
  regionManager.getRHS_REGION_INNER_QDEIM(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling collocation
  R0 = np.zeros(np.size(regionManager.region[0].RHS_hyper[:]))
  R0[:] = 1.*regionManager.region[0].RHS_hyper.flatten()
  def unsteadyResidual(v): 
    regionManager.a[regionManager.region[0].stencil_list] = decoder(v,regionManager)
    regionManager.getRHS_REGION_INNER_QDEIM(regionManager,eqns)
    ## Construct the unsteady residual for Crank Nicolson in a few steps
    R1 = np.zeros(np.size(R0))
    R1[:] = regionManager.region[0].RHS_hyper[:].flatten()
    Rstar = ( regionManager.region[0].a.a[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]] - a0).flatten()
    Rstar -= 0.5*regionManager.dt*(R1 + R0)
    return Rstar.flatten()*1.

  res_1 = scipy.optimize.least_squares(unsteadyResidual, regionManager.ahat,diff_step=1e-6,ftol=1e-6,xtol=1e-6,gtol=1e-6,verbose=2)
  regionManager.ahat = res_1.x
  regionManager.a[regionManager.region[0].stencil_list] = decoder(regionManager.ahat,regionManager)

  #### Update the global state for saving
  if (regionManager.iteration%regionManager.save_freq == 0):
    u = decoder(regionManager.ahat,regionManager)
    regionManager.a[:] = least_squares_reconstruction(regionManager.a[regionManager.region[0].stencil_list],regionManager.region[0].Phi,regionManager.region[0].stencil_list)
  regionManager.t += regionManager.dt
  regionManager.iteration += 1



## Collocated LSPG with Crank Nicolson on a manifold w/ pytorch model 
def crankNicolsonManifoldRomCollocation2(regionManager,eqns,args):
  if (regionManager.iteration == 0):
    print('Starting Simulation')
    a_stencil = regionManager.a[regionManager.region[0].stencil_list]
    regionManager.ahat = ls_encoder(a_stencil,regionManager) 
  #tmp = decoder(regionManager.ahat,regionManager)
  #regionManager.a[:] = least_squares_reconstruction(tmp,regionManager.region[0].Phi,regionManager.region[0].stencil_list)
  regionManager.a[regionManager.region[0].stencil_list] = decoder(regionManager.ahat,regionManager)

  regionManager.a0[:] = regionManager.a[:]*1.

  jacobian_update_freq = 1
  ## Get initial RHS
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling collocation
  R0 = np.zeros(np.size(regionManager.region[0].RHS_hyper[:]))
  R0[:] = regionManager.RHS[regionManager.region[0].cell_list].flatten()#_hyper[:].flatten()
  regionManager.getRHS_REGION_INNER_QDEIM(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling collocation
  print(np.linalg.norm(R0),np.linalg.norm(regionManager.region[0].RHS_hyper.flatten()),np.linalg.norm(R0 - regionManager.region[0].RHS_hyper.flatten()))
  print(np.shape(R0),np.shape(regionManager.region[0].RHS_hyper))
  def unsteadyResidual(v): 
    #tmp = decoder(regionManager.ahat,regionManager)
    #regionManager.a[:] = least_squares_reconstruction(tmp,regionManager.region[0].Phi,regionManager.region[0].stencil_list)
    regionManager.a[regionManager.region[0].stencil_list] = decoder(v,regionManager)
    regionManager.getRHS_REGION_OUTER(regionManager,eqns)
    ## Construct the unsteady residual for Crank Nicolson in a few steps
    R1 = np.zeros(np.size(R0))
    #R1[:] = regionManager.region[0].RHS_hyper[:].flatten()
    R1[:] = regionManager.RHS[regionManager.region[0].cell_list].flatten()#_hyper[:].flatten()
    Rstar = ( regionManager.a[regionManager.region[0].cell_list]*1. - regionManager.a0[regionManager.region[0].cell_list] )
    #regionManager.a[:] = Rstar[:]*1.
    #Rstar = np.sum(regionManager.region[0].M[None]*regionManager.region[0].a.a[:,None,None,None,None],axis=(5,6,7,8)).flatten()
    #print(np.linalg.norm(R1),np.linalg.norm(Rstar))
    Rstar -= regionManager.dt*R1# + R0)
    #regionManager.a[:] = v[:]
    return Rstar.flatten()*1.

  res_1 = scipy.optimize.least_squares(unsteadyResidual, regionManager.ahat,diff_step=1e-6,ftol=1e-6,xtol=1e-8,gtol=1e-8,verbose=2)
  regionManager.ahat = res_1.x
  regionManager.a[regionManager.region[0].stencil_list] = decoder(regionManager.ahat,regionManager)

  #### Update the global state for saving
  if (regionManager.iteration%regionManager.save_freq == 0):
    u = decoder(regionManager.ahat,regionManager)
    regionManager.a[:] = least_squares_reconstruction(regionManager.a[regionManager.region[0].stencil_list],regionManager.region[0].Phi,regionManager.region[0].stencil_list)
  regionManager.t += regionManager.dt
  regionManager.iteration += 1

## Collocated LSPG with Crank Nicolson on a manifold w/ pytorch model 
def crankNicolsonManifoldRomCollocation2(regionManager,eqns,args):
  if (regionManager.iteration == 0):
    print('Starting Simulation')
    a_stencil = regionManager.a[regionManager.region[0].stencil_list]
    ahat = ls_encoder(a_stencil,regionManager) 
    regionManager.ahat = ahat*1
  else:
    ahat = regionManager.ahat*1.
  tmp = decoder(regionManager.ahat,regionManager)
  regionManager.a[:] = least_squares_reconstruction(tmp,regionManager.region[0].Phi,regionManager.region[0].stencil_list)
  regionManager.a0[:] = regionManager.a[:]*1.

  jacobian_update_freq = 1
  ## Get initial RHS
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling collocation
  R0 = np.zeros(np.size(regionManager.region[0].RHS[:]))
  R0[:] = regionManager.region[0].RHS[:].flatten()


  def unsteadyResidual(v): 
    tmp = decoder(v,regionManager)
    regionManager.a[:] = least_squares_reconstruction(tmp,regionManager.region[0].Phi,regionManager.region[0].stencil_list)
    regionManager.getRHS_REGION_OUTER(regionManager,eqns)
    ## Construct the unsteady residual for Crank Nicolson in a few steps
    R1 = np.zeros(np.size(R0))
    R1[:] = regionManager.region[0].RHS[:].flatten()
    Rstar = ( regionManager.a[:]*1. - regionManager.a0[:] )
    Rstar -= regionManager.dt*R1
    return Rstar.flatten()*1.

  res_1 = scipy.optimize.least_squares(unsteadyResidual, ahat,diff_step=1e-4,ftol=1e-6,xtol=1e-8,verbose=2)
  regionManager.ahat = res_1.x
  regionManager.a[regionManager.region[0].stencil_list] = decoder(regionManager.ahat,regionManager)

  #### Update the global state for saving
  if (regionManager.iteration%regionManager.save_freq == 0):
    u = decoder(regionManager.ahat,regionManager)
    regionManager.a[:] = least_squares_reconstruction(regionManager.a[regionManager.region[0].stencil_list],regionManager.region[0].Phi,regionManager.region[0].stencil_list)
  regionManager.t += regionManager.dt
  regionManager.iteration += 1












## Collocated LSPG with Crank Nicolson 
def parametricSpaceTimeCrankNicolson(regionManager,eqns,args):
  if (regionManager.iteration == 0):
    print('Starting Simulation')
    a0 = regionManager.a[:]*1.
    for region in regionManager.region:
      region.a.a[:] = np.sum(region.M[None]*region.a.a[:,None,None,None,None],axis=(5,6,7,8) )
    regionManager.a0[:] = regionManager.a[:]
    a0_pod = globalDot(regionManager.V.transpose(),regionManager.a0,regionManager)
    regionManager.a[:] = np.dot(regionManager.V,a0_pod)
    regionManager.a0[:] = np.dot(regionManager.V,a0_pod) 
  else:
    a0_pod = regionManager.a_pod*1.
  cell_list = regionManager.region[0].cell_list
  rec_stencil_list = regionManager.region[0].rec_stencil_list

  
  def windowedResidualPod(v):
    v = np.reshape(v,(numStepsInWindow,np.size(regionManager.a_pod)))
    a_old = regionManager.a0[cell_list]*1.
    Rold = R0*1.
    Rg = np.zeros((nparams,numStepsInWindow,np.size(regionManager.region[0].cell_list)),dtype=regionManager.a.dtype)
    cell_ijk = regionManager.region[0].cell_ijk 
    for i in range(0,numStepsInWindow):
      for j in range(0,nparams):
        regionManager.region[0].RHS_hyper[:] = 0.
        regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list] , v[i,:])
        regionManager.getRHS_REGION_OUTER(regionManager,eqns)
        ## Construct the unsteady residual for Crank Nicolson in a few steps
        #R1 = 1.*regionManager.region[0].RHS_hyper[:].flatten()
        ## compute the residual at the sample points
        Rstar = ( regionManager.a[cell_list]*1. - a_old) 
        Rstar = np.reshape(Rstar,np.shape(regionManager.region[0].a.a[:,None,None,None,None,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]) ) 
        Rstar = np.sum(regionManager.region[0].M[None,:,:,:,:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*Rstar,axis=(5,6,7,8)).flatten()
        Rstar -= 0.5*regionManager.dt*(regionManager.region[0].RHS_hyper[:].flatten() + Rold)
        Rg[i,:] = Rstar*1.
        Rold = regionManager.region[0].RHS_hyper[:].flatten()*1.
        a_old =  np.dot(regionManager.V[cell_list] , v[i,:])
    return Rg.flatten()
      

  def computeResidualPOD(v):
    return unsteadyResidual(np.dot(regionManager.V[rec_stencil_list],v[:]) )

  def unsteadyResidual(v):
    regionManager.a[rec_stencil_list] = v[:]
    regionManager.getRHS_REGION_OUTER(regionManager,eqns)
    cell_ijk = regionManager.region[0].cell_ijk 
    ## Construct the unsteady residual for Crank Nicolson in a few steps
    R1 = np.zeros(np.size(R0))
    R1[:] = regionManager.region[0].RHS_hyper[:].flatten()
    ## compute the residual at the sample points
    Rstar = ( regionManager.a[cell_list]*1. - regionManager.a0[cell_list] ) 
    regionManager.a[cell_list] = Rstar[:]*1.
    Rstar = np.sum(regionManager.region[0].M[None,:,:,:,:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]*regionManager.region[0].a.a[:,None,None,None,None,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]],axis=(5,6,7,8)).flatten()
    Rstar -= 0.5*regionManager.dt*(R1 + R0)
    regionManager.a[rec_stencil_list] = v[:]
    return Rstar.flatten()

  def initEuler(a_pod):
    a_pod_w = np.zeros((numStepsInWindow,np.size(regionManager.a_pod)))
    for i in range(0,numStepsInWindow):
      regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list],a_pod)
      regionManager.getRHS_REGION_OUTER(regionManager,eqns)
      a_pod = a_pod + regionManager.dt*globalDot(regionManager.V[cell_list].transpose(),regionManager.region[0].RHS_hyper.flatten(),regionManager)
      a_pod_w[i,:] = a_pod
    return a_pod_w

  N = np.size(regionManager.a)
  nx,nbasis = np.shape(regionManager.V[cell_list])
  JV = np.zeros((nx,nbasis) )
  regionManager.an = regionManager.a[rec_stencil_list]*1.
  method = 'SD'
  numStepsInWindow = 2
  if method == 'SD':
    #a0_pod = np.repeat(regionManager.a_pod[None,:],numStepsInWindow,axis=0)
    a0_pod = initEuler(regionManager.a_pod)
    a_pod = steepestDecentMomentum(regionManager,windowedResidualPod,applyWindowedJT,a0_pod.flatten())
    a_pod = np.reshape(a_pod,(numStepsInWindow,np.size(regionManager.a_pod)))
    regionManager.a_pod = a_pod[-1,:]
    regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list],regionManager.a_pod[:])
  if method == 'GN':
    if (regionManager.mpi_rank == 0):
      print('Initial Residual = ' + str(r_norm) )
    while (r_norm > 1e-4 and da_norm >= 5e-4):
     if (ls_iteration >= 20 or alpha <= 0.01):
       print('Didnt converge, alpha = ' + str(alpha) ,'  iteration count = ' + str(ls_iteration) )
       da_norm = 0.
     else:
      
      eps = 1e-5
      an = regionManager.a[rec_stencil_list]*1.
      if (ls_iteration%jacobian_update_freq == 0 and regionManager.iteration%1 == 0):
       for i in range(0,nbasis):
        regionManager.a[rec_stencil_list] = an[:] + eps*regionManager.V[rec_stencil_list,i]
        Jv_column = unsteadyResidual(regionManager.a[rec_stencil_list])
        JV[:,i] = (Jv_column - r[:]) / eps
       regionManager.JV = JV
      JV = regionManager.JV 
      ## compute QR of JV
      Q,R = np.linalg.qr(JV)
      #now solve the problem R y = -Q^T JVr 
      da_pod = np.linalg.solve(R,-np.dot(Q.transpose(),r) )
  #    JSQ = np.dot(JV.transpose(),JV)
      ## do global sum to get the dot product
  #    data = regionManager.comm.gather(JSQ,root = 0)
  #    JSQ_glob = np.zeros(np.shape(JSQ) )
  #    if (regionManager.mpi_rank == 0):
  #      for j in range(0,regionManager.num_processes):
  #        JSQ_glob[:] += data[j]
  #      for j in range(1,regionManager.num_processes):
  #        comm.Send(JSQ_glob, dest=j)
  #    else:
  #      comm.Recv(JSQ_glob,source=0)
  #    da_pod = np.linalg.solve(JSQ_glob,-JVr)
  #    JTr =  applyJT(r)
  #    regionManager.JTr = JTr
  #    regionManager.JTr2 = JTr2
  #    print(np.linalg.norm(JTr2),np.linalg.norm(JTr),np.linalg.norm(JTr2 - JTr))
  #    da_pod = -250.*JTr
      da_norm = np.linalg.norm(da_pod)
      regionManager.a_pod[:] += da_pod[:]
      regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list],regionManager.a_pod[:])
      r = unsteadyResidual(regionManager.a[rec_stencil_list])
      r_norm = globalNorm(r,regionManager)
      ls_iteration += 1
      if (regionManager.mpi_rank == 0):
        print('Residual = ' + str(r_norm), ' Change to solution is ' + str(da_norm))
  regionManager.t += regionManager.dt*numStepsInWindow
  regionManager.iteration += 1*numStepsInWindow

  ### Update the global state for saving
  if (regionManager.iteration%regionManager.save_freq == 0):
    #regionManager.a_pod[:] = np.dot(regionManager.V[cell_list].transpose(),regionManager.a[cell_list] )
    regionManager.a[:] = np.dot(regionManager.V,regionManager.a_pod)












