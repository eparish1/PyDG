import numpy as np
from pylab import *
#import sys, petsc4py
#petsc4py.init(sys.argv)
#from petsc4py import PETSc
from DG_core import * 
import scipy.linalg
from linear_solvers import *
import matplotlib.pyplot as plt
from mpi4py import MPI
import sys
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
#from jacobian_schemes import *
from navier_stokes_entropy import entropy_to_conservative, getEntropyMassMatrix,getEntropyMassMatrix_noinvert
#from block_classes import getRHS_REGION_INNER_ROM_MORTHOGONAL
def gatherResid(Rstar,regionManager):
  ## Create Global residual
  data = regionManager.comm.gather(np.linalg.norm(Rstar)**2,root = 0)
  if (regionManager.mpi_rank == 0):
    Rstar_glob = 0.
    for j in range(0,regionManager.num_processes):
      Rstar_glob += data[j]
    Rstar_glob = np.sqrt(Rstar_glob)
    for j in range(1,regionManager.num_processes):
      regionManager.comm.send(Rstar_glob, dest=j)
  else:
    Rstar_glob = regionManager.comm.recv(source=0)
  #print(np.size(Rstar_glob),Rstar_glob)
  return Rstar_glob

def spaceTime(main,MZ,eqns,args=None):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  main.a0[:] = main.a.a[:]
  alpha = 0.1
  def unsteadyResidual_element(main,v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    getRHS_element(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    ## now integrate the volume term ( \int ( dw/dt * u) )
    main.basis.reconstructU(main,main.a)
    volint_t = main.basis.volIntegrateGlob(main,main.a.u[:]*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.wp3)*2./main.dt
    uFuture,uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)
    futureFlux = main.basis.faceIntegrateGlob(main,uFuture*main.Jdet[None,:,:,:,:,:,:,None],main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    Rstar = volint_t - (futureFlux[:,:,:,:,None])*2./main.dt + R1[:]
    main.basis.applyMassMatrix(main,Rstar)
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob


  def create_MF_Jacobian_element(v,args,main):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-3
    main.a.a[:] = an + eps*vr
    getRHS_element(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    vr_phys = main.basis.reconstructUGeneral(main,vr)
    volint_t = main.basis.volIntegrateGlob(main,vr_phys*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.wp3)*2./main.dt
    uFuture,uPast = main.basis.reconstructEdgesGeneralTime(vr,main)
    futureFlux = main.basis.faceIntegrateGlob(main,uFuture*main.Jdet[None,:,:,:,:,:,:,None],main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    Av = volint_t - (futureFlux[:,:,:,:,None])*2./main.dt + 1./eps * (R1 - Rn)
    main.basis.applyMassMatrix(main,Av)
    return Av.flatten()

  def unsteadyResidual_element2(main,v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    getRHS_element(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    ## now integrate the volume term ( \int ( dw/dt * u) )
    main.basis.reconstructU(main,main.a)
    volint_t = main.basis.volIntegrateGlob(main,main.a.u[:]*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.wp3)*2./main.dt
    uFuture,uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)
    futureFlux = main.basis.faceIntegrateGlob(main,uFuture*main.Jdet[None,:,:,:,:,:,:,None],main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    Rstar = volint_t - (futureFlux[:,:,:,:,None])*2./main.dt + main.RHS[:]
    main.basis.applyMassMatrix(main,Rstar)
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob


  def create_MF_Jacobian_element2(v,args,main):
    an = args[0]
    Rn = args[1]
    #print(np.linalg.norm(JinvX))
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-5
#    Rstar0,R10,Rstar_glob0 = unsteadyResidual_element(main,main.a.a)
#    Rstar1,R11,Rstar_glob1 = unsteadyResidual_element(main,main.a.a+eps*vr)
#    Av= 1./eps*(Rstar1 - Rstar0)
    main.a.a[:] = an + eps*vr
    getRHS_element(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    vr_phys = main.basis.reconstructUGeneral(main,vr)
    volint_t = main.basis.volIntegrateGlob(main,vr_phys*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.wp3)*2./main.dt
    uFuture,uPast = main.basis.reconstructEdgesGeneralTime(vr,main)
    futureFlux = main.basis.faceIntegrateGlob(main,uFuture*main.Jdet[None,:,:,:,:,:,:,None],main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    Av = volint_t - (futureFlux[:,:,:,:,None])*2./main.dt + 1./eps * (R1 - Rn)
    main.basis.applyMassMatrix(main,Av)
    Av = np.reshape(Av, (main.nvars*main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
#    main.a.a[:] = an[:]

    return Av




#  JXT02 = computeJacobianXT(main,eqns,unsteadyResidual_element)
#  JXT02 = np.reshape(JXT02,(main.nvars,main.order[0],main.order[3],main.order[0]*main.order[3],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
#  JXT02 = np.reshape(JXT02,(main.nvars,main.order[0],main.order[3],main.order[0],main.order[3],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
#  J = computeBlockJacobian(main,unsteadyResidual_element) #get the Jacobian
#  J2 = np.zeros(np.shape(J))
#  J2[:] = J[:]
#  main.J = J
  def create_Dinv2(f,main,args=None):
    ff = np.reshape(f*1.,np.shape(main.a.a))
    main.basis.applyMassMatrix(main,ff)
    return ff.flatten()


  def create_Dinv(fs,main,args=None):
    f = np.zeros(np.shape(fs))
    f[:] = fs[:]
    #if (args[2]%100 == 0):
    #  print('computing')
    #  J2[:] = computeBlockJacobian(main,unsteadyResidual_element) #get the Jacobian
    J = np.reshape(J2[0:main.nvars,0:main.order[0],0:main.order[1],0:main.order[2],0:main.order[3],0:main.nvars,0:main.order[0],0:main.order[1],0:main.order[2],0:main.order[3] ,0:main.Npx,0:main.Npy,0:main.Npz,0:main.Npt], (main.nvars*main.order[0]*main.order[1]*main.order[2]*main.order[3],main.nvars*main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    J = np.rollaxis(np.rollaxis(J ,1,6),0,5)
    f = np.reshape(f,np.shape(main.a.a))
    f = np.reshape(f, (main.nvars*main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    f = np.rollaxis(f,0,5)
    f = np.linalg.solve(J,f)
    f = np.rollaxis(f,4,0)
    f = np.reshape(f,np.shape(main.a.a) )
    #main.basis.applyMassMatrix(main,f)
    return f.flatten()

  def create_Dinv2(f,main,args):
    tol = args[1]
    a0 = np.zeros(np.shape(main.a.a))
    a0[:] = main.a.a[:]
    f0 = np.zeros(np.size(f))
    f0[:] = f[:]
    f = np.reshape(f,np.shape(main.a.a))
    f0r = np.reshape(f0,np.shape(main.a.a))
    #print('before unsteady residual')
    Rstarn_pc,Rn_pc,Rstar_glob_pc = unsteadyResidual_element(main,a0)
    #print('e unsteady residual')

    MF_Jacobian_args2 = [a0,Rn_pc]
    f0r2 = np.reshape(f0r, (main.nvars*main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    #ferror = GMRes(create_MF_Jacobian_element, f.flatten() - Rstarn_pc.flatten(),-a0.flatten(),main,MF_Jacobian_args2,None,None,1e-6,1,200,False)
    #ff = a0[:] + np.reshape(ferror,np.shape(main.a.a))
    #ferror = GMRes(create_MF_Jacobian, f.flatten(),f.flatten()*0.,main,MF_Jacobian_args2,None,None,tol,1,15,False)
    #ff = np.reshape(ferror,np.shape(main.a.a))
    #ferror = GMRes_element(create_MF_Jacobian_element2, f0r2 - np.reshape(Rstarn_pc,np.shape(f0r2)), -np.reshape(a0,np.shape(f0r2)),main,MF_Jacobian_args2,None,None,1e-10,1,500,False)
    #ff = a0 + np.reshape(ferror,np.shape(main.a.a))
    ferror = GMRes_element(create_MF_Jacobian_element2, f0r2, -np.reshape(a0,np.shape(f0r2))*0.,main,MF_Jacobian_args2,None,None,1e-10,1,50,False)
    ff = np.reshape(ferror,np.shape(main.a.a))
    return ff.flatten()

  def unsteadyResidual(main,v):
    main.a.a[:] = np.reshape(v*1.,np.shape(main.a.a))
    eqns.getRHS(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    ## now integrate the volume term ( \int ( dw/dt * u) )
    volint_t = main.basis.volIntegrateGlob(main,main.a.u*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.wp3)*2./main.dt
    uFuture,uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)
    futureFlux = main.basis.faceIntegrateGlob(main,uFuture*main.Jdet[None,:,:,:,:,:,:,None],main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    uPast[:,:,:,:,:,:,:,0] = main.a.uFuture[:,:,:,:,:,:,:,-1]
    if (main.Npt > 1):
      uPast[:,:,:,:,:,:,:,1::] = uFuture[:,:,:,:,:,:,:,0:-1]
    pastFlux   = main.basis.faceIntegrateGlob(main,uPast*main.Jdet[None,:,:,:,:,:,:,None] ,main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    Rstar = volint_t - (futureFlux[:,:,:,:,None] - pastFlux[:,:,:,:,None]*main.altarray3[None,None,None,None,:,None,None,None,None])*2./main.dt + main.RHS[:]
#    Rstar = np.einsum('zpqrij...,zpqrj...->zpqri...',JinvT,Rstar)
#    Rstar = np.einsum('zij...,zj...->zi...',JinvX,Rstar)
#    Rstar = np.einsum('ziljm...,zjpqm...->zipql...',JinvXT,Rstar)
    main.basis.applyMassMatrix(main,Rstar)
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob
    #return Rstar.flatten()


  def create_MF_Jacobian(v,args,main):
    an = args[0]
    Rn = args[1]
    #print(np.linalg.norm(JinvX))
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-5
#    Rstar0,R10,Rstar_glob0 = unsteadyResidual(main,an)
#    Rstar1,R11,Rstar_glob1 = unsteadyResidual(main,an+eps*vr*1.)
#    Av = 1./eps*(Rstar1 - Rstar0)
    main.a.a[:] = an + eps*vr
    eqns.getRHS(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    vr_phys = main.basis.reconstructUGeneral(main,vr)
    volint_t = main.basis.volIntegrateGlob(main,vr_phys*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.wp3)*2./main.dt
    uFuture,uPast = main.basis.reconstructEdgesGeneralTime(vr,main)
    futureFlux = main.basis.faceIntegrateGlob(main,uFuture*main.Jdet[None,:,:,:,:,:,:,None],main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    uPast[:,:,:,:,:,:,:,0] = 0
    if (main.Npt > 1):
      uPast[:,:,:,:,:,:,:,1::] = uFuture[:,:,:,:,:,:,:,0:-1]
    pastFlux   = main.basis.faceIntegrateGlob(main,uPast*main.Jdet[None,:,:,:,:,:,:,None]  ,main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    Av = volint_t - (futureFlux[:,:,:,:,None] - pastFlux[:,:,:,:,None]*main.altarray3[None,None,None,None,:,None,None,None,None])*2./main.dt + 1./eps * (R1 - Rn)
    main.basis.applyMassMatrix(main,Av)
    return Av.flatten()

  nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns,create_Dinv)

  #def printer(f,x):
  #  print(np.linalg.norm(f),np.linalg.norm(x))
  #scipy.optimize.newton_krylov(unsteadyResidualDum,main.a.a.flatten(),method='gmres',verbose=4)
  main.t += main.dt*main.Npt
  main.iteration += 1
  main.a.uFuture[:],uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)


##########=============================================================

def limiter_MF(main):
   main.basis.reconstructU(main,main.a)
   main.a.u[5::] = np.fmax(main.a.u[5::],1e-7)
   main.a.u[5::] = np.fmin(main.a.u[5::],main.a.u[None,0]*np.ones(np.shape(main.a.u[5::])))
   ord_arrx= np.linspace(0,main.order[0]-1,main.order[0])
   ord_arry= np.linspace(0,main.order[1]-1,main.order[1])
   ord_arrz= np.linspace(0,main.order[2]-1,main.order[2])
   ord_arrt= np.linspace(0,main.order[3]-1,main.order[3])
   scale =  (2.*ord_arrx[:,None,None,None] + 1.)*(2.*ord_arry[None,:,None,None] + 1.)*(2.*ord_arrz[None,None,:,None] + 1.)*(2.*ord_arrt[None,None,None,:] + 1.)/16.
   main.a.a[:] = main.basis.volIntegrateGlob(main,main.a.u,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]


def limiter_characteristic(main):
  gamma = 1.4
#  R = 8314.4621/1000.
  atmp = np.zeros(np.shape(main.a.a))
#  atmp[:,0,0,0] = main.a.a[:,0,0,0]
  atmp[:] = main.a.a[:] 
#  main.basis.reconstructU(main,main.a)
#  u0 = np.zeros(np.shape(main.a.u))
#  u0[:] = main.a.u[:]
#  Y0 = np.zeros(np.shape(main.a.u[5::]))
#  Y0[:] = main.a.u[5::]/main.a.u[None,0]
  U = main.basis.reconstructUGeneral(main,atmp)
#  Y_N2 = 1. - np.sum(U[5::]/U[None,0],axis=0)
#  Winv =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],U[5::]/U[None,0]) + 1./main.W[-1]*Y_N2
#  Cp = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],U[5::]/U[None,0]) + main.Cp[-1]*Y_N2
#  Cv = Cp - R*Winv
#  gamma = Cp/Cv
  
  p = (gamma - 1.)*(U[4] - 0.5*U[1]**2/U[0] - 0.5*U[2]**2/U[0] - 0.5*U[3]**2/U[0])
  c = np.sqrt(gamma*p/U[0])
  u = U[1] / U[0]
  v = U[2] / U[0]
  w = U[3] / U[0]
  nx = 1.
  ny = 0.
  nz = 0.
  mx = 1.
  my = 0.
  mz = 0.
  lx = 1.
  ly = 0.
  lz = 0.
  K = gamma - 1. 
  ql = u*1.
  qm = u*1.
  qn = u*1.

  sizeu = np.array([5,5])#np.shape(main.a.u)[0]
  sizeu = np.append(sizeu,np.shape(main.a.u[0]))
  L = np.zeros(sizeu)
  R = np.zeros(sizeu)

  q = np.sqrt(u**2 + v**2 + w**2)
  L[0,0] = K*q**2/(4.*c**2) + qn/(2.*c)
  L[0,1] = -(K/(2.*c**2)*u + nx/(2.*c))
  L[0,2] = -(K/(2.*c**2)*v + ny/(2.*c))
  L[0,3] = -(K/(2.*c**2)*w + nz/(2.*c))
  L[0,4] = K/(2.*c**2)
  L[1,0] = 1. - K*q**2/(2.*c**2)
  L[1,1] = K*u/c**2
  L[1,2] = K*v/c**2
  L[1,3] = K*w/c**2
  L[1,4] = -K/c**2
  L[2,0] = K*q**2/(4.*c**2) - qn/(2.*c)
  L[2,1] = -(K/(2.*c**2)*u - nx/(2.*c) )
  L[2,2] = -(K/(2.*c**2)*v - ny/(2.*c) )
  L[2,3] = -(K/(2.*c**2)*w - nz/(2.*c) )
  L[2,4] = K/(2.*c**2)
  L[3,0] = -ql
  L[3,1] = lx
  L[3,2] = ly
  L[3,3] = lz
  L[4,0] = -qm
  L[4,1] = mx
  L[4,2] = my
  L[4,3] = mz

  # compute H in three steps (H = E + p/rho)
  H = (gamma - 1.)*(U[4] - 0.5*U[0]*q**2) #compute pressure
  H += U[4]
  H /= U[0]

  R[0,0] = 1.
  R[0,1] = 1.
  R[0,2] = 1.
  R[1,0] = u - c*nx
  R[1,1] = u
  R[1,2] = u + c*nx
  R[1,3] = lx
  R[1,4] = mx
  R[2,0] = v - c*ny
  R[2,1] = v
  R[2,2] = v + c*ny
  R[2,3] = ly
  R[2,4] = my
  R[3,0] = w - c*nz
  R[3,1] = w
  R[3,2] = w + c*nz
  R[3,3] = lz
  R[3,4] = mz
  R[4,0] = H - qn*c 
  R[4,1] = q**2/2.
  R[4,2] = H + qn*c
  R[4,3] = ql
  R[4,4] = qm


  w = np.einsum('ij...,j...->i...',L,main.a.u[0:5])
  ord_arrx= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arry= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arrz= np.linspace(0,main.order[2]-1,main.order[2])
  ord_arrt= np.linspace(0,main.order[3]-1,main.order[3])
  scale =  (2.*ord_arrx[:,None,None,None] + 1.)*(2.*ord_arry[None,:,None,None] + 1.)*(2.*ord_arrz[None,None,:,None] + 1.)*(2.*ord_arrt[None,None,None,:] + 1.)/16.
  Cw = main.basis.volIntegrateGlob(main,w,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]
  Cwf = np.zeros(np.shape(main.a.a[0:5]))
  Cwf[:] = Cw[:]
  dcR = np.zeros(np.shape(main.a.a[0:5]))
  dcL = np.zeros(np.shape(main.a.a[0:5]))
  dcR[:,:,:,:,:,0:-1] = Cw[:,:,:,:,:,1::] - Cw[:,:,:,:,:,0:-1]   
  dcR[:,:,:,:,:,-1] =   dcR[:,:,:,:,:,-2]
  dcL[:,:,:,:,:,1::] = dcR[:,:,:,:,:,0:-1]
  dcL[:,:,:,:,:,0] = dcL[:,:,:,:,:,1]

  ## perform filtering
  pindx = np.shape(main.a.a[0:5])[1] - 1
  Cwf0 = np.zeros(np.shape(main.a.a[0:5]))
  check = 0
  indx_eq = np.zeros(np.shape(main.a.a[0:5,0]),dtype=bool)
  while (pindx >= 1 and check == 0):
    Cwf0[:] = Cwf[:]
    indx = (np.sign(Cwf0[:,pindx])==np.sign(dcR[:,pindx-1])) & (np.sign(dcR[:,pindx-1])==np.sign(dcL[:,pindx-1] ))
    #print(np.size(Cw[:,-1][indx]),pindx) 
    w_limit = np.zeros(np.shape(main.a.u) )
    alpha = 1.#2./(main.dx*main.order[0])
    Cwf[:,pindx] = 0.
    Cwf[:,pindx][indx] = np.sign(Cwf0[:,pindx][indx])*np.fmin( np.abs(Cwf0[:,pindx][indx]), np.fmin(np.abs(alpha*dcR[:,pindx-1][indx]),np.abs(alpha*dcL[:,pindx-1][indx]) ))
    #print(np.size(Cwf[indx_eq]),pindx)
    Cwf[:,pindx][indx_eq] = Cwf0[:,pindx][indx_eq]
    indx_eq = np.isclose(Cwf[:,pindx],Cwf0[:,pindx],rtol=1e-7,atol = 1e-11)
    pindx -= 1
  w = main.basis.reconstructUGeneral(main,Cwf)
  u2 = np.zeros(np.shape(main.a.u))
  u2[0:5] = np.einsum('ij...,j...->i...',R,w)
#  u2[5::] = u2[None,0]*Y0

  #print(np.linalg.norm(main.a.a))
  main.a.a[:] = main.basis.volIntegrateGlob(main,u2,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]
  #print(np.linalg.norm(main.a.a))

  #print('hi') 
def limiter(main):
#   main.basis.reconstructU(main,main.a)
#   u0 = main.a.u*1.
   a_f = np.zeros(np.shape(main.a.a))
   a_f[:] = main.a.a[:]
   a_f[:,-1] = 0.
   dcR = np.zeros(np.shape(main.a.a))
   dcL = np.zeros(np.shape(main.a.a))
   dcR[:,:,:,:,:,0:-1] = main.a.a[:,:,:,:,:,1::] - main.a.a[:,:,:,:,:,0:-1]   
   dcR[:,:,:,:,:,-1] =   dcR[:,:,:,:,:,-2]
   dcL[:,:,:,:,:,1::] = dcR[:,:,:,:,:,0:-1]
   dcL[:,:,:,:,:,0] = dcL[:,:,:,:,:,1]
   indx = (sign(main.a.a[:,-1])==sign(alpha*dcR[:,-2])) & (sign(alpha*dcR[:,-2])==sign(alpha*dcL[:,-2] ))
   u_limit = np.zeros(np.shape(main.a.u) )
   alpha = 1.#2./(main.dx*main.order[0])
   a_f[:,-1][indx] = np.sign(main.a.a[:,-1][indx])*np.fmin( np.abs(main.a.a[:,-1][indx]),np.abs(alpha*dcR[:,-2][indx]),np.abs(alpha*dcL[:,-2][indx]) )
   main.a.a[:] = a_f[:]

#   main.basis.reconstructU(main,main.a)
#   u1 = main.a.u*1.

def SSP_RK3_Entropy(main,MZ,eqns,args=None):

  #======= First Stage
  main.getRHS(main,MZ,eqns)  
  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  #getEntropyMassMatrix(main)
  M = getEntropyMassMatrix_noinvert(main)
  R = np.reshape(main.RHS[:],(main.nvars*main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt) )
  R = np.rollaxis(R,0,5)
  R = np.linalg.solve(M,R)
  R = np.rollaxis(R,4,0)
  #R = np.einsum('ij...,j...->i...',main.EMM,R)
  R = np.reshape(R,np.shape(main.a.a))
  a1 = main.a.a[:]  + main.dt*(R[:])
  #a1[:,main.order[0]/2::,main.order[1]/2::,:] = 0.
  main.a.a[:] = a1[:]
  #========= Second Stage
  main.getRHS(main,MZ,eqns)
  #getEntropyMassMatrix(main)
  M = getEntropyMassMatrix_noinvert(main)
  R = np.reshape(main.RHS[:],(main.nvars*main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt) )
  R = np.rollaxis(R,0,5)
  R = np.linalg.solve(M,R)
  R = np.rollaxis(R,4,0)
  #R = np.einsum('ij...,j...->i...',main.EMM,R)
  R = np.reshape(R,np.shape(main.a.a))
  a1[:] = 3./4.*a0 + 1./4.*(a1 + main.dt*R[:]) #reuse a1 vector
  #a1[:,main.order[0]/2::,main.order[1]/2::,:] = 0.
  main.a.a[:] = a1[:]
 
  #========== Third Stage
  main.getRHS(main,MZ,eqns)  
  #getEntropyMassMatrix(main)
  M = getEntropyMassMatrix_noinvert(main)
  R = np.reshape(main.RHS[:],(main.nvars*main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt) )
  R = np.rollaxis(R,0,5)
  R = np.linalg.solve(M,R)
  R = np.rollaxis(R,4,0)
  #R = np.einsum('ij...,j...->i...',main.EMM,R)
  R = np.reshape(R,np.shape(main.a.a))
  main.a.a[:] = 1./3.*a0 + 2./3.*(a1[:] + main.dt*R[:])
  #main.a.a[:,main.order[0]/2::,main.order[1]/2::] = 0.
  main.t += main.dt
  main.iteration += 1

def ExplicitRK2(main,MZ,eqns,args=None):
  main.a0[:] = main.a.a[:]
  rk4const = np.array([1./2,1.])
  for i in range(0,2):
    main.rkstage = i
    main.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
    main.basis.applyMassMatrix(main,main.RHS) 
    main.a.a[:] = main.a0 + main.dt*rk4const[i]*main.RHS
    #limiter_MF(main)
  main.t += main.dt
  main.iteration += 1

def ExplicitRK4(regionManager,eqns,args=None):
  region_counter = 0
  regionManager.a0[:] = regionManager.a
  rk4const = np.array([1./4,1./3,1./2,1.])
  for i in range(0,4):
    region_counter = 0
    regionManager.getRHS_REGION_OUTER(regionManager,eqns)
    regionManager.a[:] = regionManager.a0[:] + regionManager.dt*rk4const[i]*regionManager.RHS
  regionManager.t += regionManager.dt
  regionManager.iteration += 1


def filter_shock(regionManager,eqns,args=None):
  for region in regionManager.region:
    pmax = np.amax(region.order)
    filt_cells = np.zeros(np.shape(region.a.a[0,0,0,0,0]),dtype=bool)
    u = region.basis.reconstructUGeneral(region,region.a.a)*1.
    for pf in range(1,pmax):
      filtarray = np.zeros(np.shape(region.a.a))
      fx0 = np.fmax(region.order[0]-pf,1)
      fx1 = np.fmax(region.order[1]-pf,1)
      fx2 = np.fmax(region.order[2]-pf,1)
      fx3 = np.fmax(region.order[3]-pf,1)
      filtarray[:,0:fx0,0:fx1,0:fx2,0:fx3] = 1.
      af = region.a.a*filtarray
      uf = region.basis.reconstructUGeneral(region,af)
      Se = region.basis.volIntegrate(region.weights0,region.weights1,region.weights2,region.weights3, (u - uf)*(u - uf) )/( region.basis.volIntegrate( region.weights0,region.weights1,region.weights2,region.weights3, u*u) ) + 1e-30
      sensor = np.log10(Se[0])
      kappa = 2. 
      s0  = 1./np.amax(region.order)**4 - kappa
      tol = s0
      if (pf == 1):
        filt_cells[sensor>tol] = True
      else:
        filt_cells[np.logical_and( sensor>tol,filt_cells==True)] = True 
      region.a.a[:,:,:,:,:,filt_cells] = region.a.a[:,:,:,:,:,filt_cells]*filtarray[:,:,:,:,:,filt_cells]
#      print(pf,np.size(filt_cells[filt_cells]))
#    print(np.shape(filt)) 
#    plot(sensor.flatten())
#    plot(region.a.u[0,0,0].flatten())
#    plot(np.ones(100)*tol)
#    
#    pause(0.001)  
#    clf()

def SSP_RK3(regionManager,eqns,args=None):
  ### get a0
  regionManager.a0[:] = regionManager.a[:]
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions

  regionManager.a1 = regionManager.a[:]  + regionManager.dt*regionManager.RHS[:]
  regionManager.a[:] = regionManager.a1[:]
  #filter_shock(regionManager,eqns)

  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions

  regionManager.a1 = 3./4.*regionManager.a0[:]  + 1./4.*(regionManager.a1 + regionManager.dt*regionManager.RHS[:])
  regionManager.a[:] = regionManager.a1[:]
  #filter_shock(regionManager,eqns)

  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions

  regionManager.a[:] = 1./3.*regionManager.a0[:]  + 2./3.*(regionManager.a1 + regionManager.dt*regionManager.RHS[:])
  regionManager.t += regionManager.dt
  regionManager.iteration += 1
  #filter_shock(regionManager,eqns)

def SSP_RK3_POD_QDEIM(regionManager,eqns,args=None):
  regionManager.rk_stage = 0
  a0 = regionManager.a[:]*1.
  for region in regionManager.region:
    region.a.a[:] = np.sum(region.M[None]*region.a.a[:,None,None,None,None],axis=(5,6,7,8) )
  regionManager.a0[:] = regionManager.a[:]
  a0_pod = globalDot(regionManager.V.transpose(),regionManager.a0,regionManager)
  regionManager.a[:] = a0[:]

  cell_ijk = regionManager.region[0].cell_ijk
  stencil_list = regionManager.region[0].stencil_list
  stencil_ijk = regionManager.region[0].stencil_ijk
  '''

  rec_stencil_list = regionManager.region[0].rec_stencil_list
  cell_list = regionManager.region[0].cell_list
  viscous_stencil_list = regionManager.region[0].viscous_stencil_list
  regionManager.rk_stage = 0
  if (regionManager.iteration == 0):
    print('Starting Simulation')
    regionManager.a0[:] = regionManager.a[:]
    a0_pod = globalDot(regionManager.V.transpose(),regionManager.a0,regionManager)
    regionManager.MR_rec = regionManager.MR[rec_stencil_list]
    regionManager.V_rec = regionManager.V[rec_stencil_list]

  else:
    a0_pod = regionManager.a_pod*1.
  #a0_pod = np.dot(regionManager.V[stencil_list,:].transpose(),regionManager.a[stencil_list]*1.)
  ## First Stage
  '''
  print(np.linalg.norm(regionManager.a))
  t0 = time.time()
  for i in range(0,100):
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  print('walltime = ' , time.time() - t0)
  norm1 = np.linalg.norm(regionManager.region[0].RHS_hyper)

  regionManager.getRHS_REGION_INNER_ROM_MORTHOGONAL(regionManager,eqns)
  norm2 = np.linalg.norm(regionManager.region[0].RHS[:,:,:,:,:,cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]]) 
  print(norm1, norm2, norm1 - norm2)
  #regionManager.rk_stage += 1
  '''
  R = np.dot(regionManager.V[cell_list].transpose(),regionManager.RHS[cell_list])
  #R = np.einsum('ij,j->i',regionManager.VR,regionManager.region[0].RHS[cell_ijk[0][0],cell_ijk[1][0],cell_ijk[2][0],cell_ijk[3][0],cell_ijk[4][0],cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]].flatten() )
  a1_pod = a0_pod  + regionManager.dt*R.flatten()#globalDot(regionManager.V.transpose(),regionManager.RHS[:],regionManager) 
  regionManager.a[rec_stencil_list] = np.dot(regionManager.V_rec,a1_pod)
  regionManager.a[:] = np.dot(regionManager.V[:,:],a1_pod)
  ## Second Stage
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  R = np.dot(regionManager.V[cell_list].transpose(),regionManager.RHS[cell_list])
  #R = np.einsum('ij,j->i',regionManager.VR,regionManager.region[0].RHS[cell_ijk[0][0],cell_ijk[1][0],cell_ijk[2][0],cell_ijk[3][0],cell_ijk[4][0],cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]].flatten() )

  regionManager.rk_stage += 1
  a1_pod = 3./4.*a0_pod + 1./4.*(a1_pod + regionManager.dt*R ) #reuse a1 vector
  regionManager.a[rec_stencil_list] = np.dot(regionManager.V_rec,a1_pod)
  #regionManager.a[:] = np.dot(regionManager.V[:,:],a1_pod)

  ## Third Stage
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  R = np.dot(regionManager.V[cell_list].transpose(),regionManager.RHS[cell_list])
  #R = np.einsum('ij,j->i',regionManager.VR,regionManager.region[0].RHS[cell_ijk[0][0],cell_ijk[1][0],cell_ijk[2][0],cell_ijk[3][0],cell_ijk[4][0],cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]].flatten() )

  af_pod = 1./3.*a0_pod + 2./3.*(a1_pod[:] + regionManager.dt*R )
  regionManager.a[rec_stencil_list] = np.dot(regionManager.V_rec,af_pod*1.)
  #regionManager.a[:] = np.dot(regionManager.V[:,:],af_pod*1.)
  regionManager.iteration += 1
  regionManager.a_pod[:] = af_pod[:]

  ### Update the global state for saving
  if (regionManager.iteration%regionManager.save_freq == 0):
    regionManager.a[:] = np.dot(regionManager.V,af_pod)
  '''
  regionManager.t += regionManager.dt





def SSP_RK3_POD_QDEIM3(regionManager,eqns,args=None):
  cell_ijk = regionManager.region[0].cell_ijk
  stencil_list = regionManager.region[0].stencil_list
  stencil_ijk = regionManager.region[0].stencil_ijk
  rec_stencil_list = regionManager.region[0].rec_stencil_list
  cell_list = regionManager.region[0].cell_list
  viscous_stencil_list = regionManager.region[0].viscous_stencil_list

  regionManager.rk_stage = 0
  if (regionManager.iteration == 0):
    print('Starting Simulation')
    regionManager.a0[:] = regionManager.a[:]
    a0_pod = globalDot(regionManager.V.transpose(),regionManager.a0,regionManager)
    regionManager.MR_rec = regionManager.MR[rec_stencil_list]
    regionManager.V_rec = regionManager.V[rec_stencil_list]

  else:
    a0_pod = regionManager.a_pod*1.
  #a0_pod = np.dot(regionManager.V[stencil_list,:].transpose(),regionManager.a[stencil_list]*1.)
  ## First Stage
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  regionManager.rk_stage += 1

  R = np.dot(regionManager.VR,regionManager.RHS)
  #R = np.einsum('ij,j->i',regionManager.VR,regionManager.region[0].RHS[cell_ijk[0][0],cell_ijk[1][0],cell_ijk[2][0],cell_ijk[3][0],cell_ijk[4][0],cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]].flatten() )
  a1_pod = a0_pod  + regionManager.dt*R.flatten()#globalDot(regionManager.V.transpose(),regionManager.RHS[:],regionManager) 
  regionManager.a[rec_stencil_list] = np.dot(regionManager.V_rec,a1_pod)
  regionManager.a[:] = np.dot(regionManager.V[:,:],a1_pod)
  ## Second Stage
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  R = np.dot(regionManager.VR,regionManager.RHS)
  #R = np.einsum('ij,j->i',regionManager.VR,regionManager.region[0].RHS[cell_ijk[0][0],cell_ijk[1][0],cell_ijk[2][0],cell_ijk[3][0],cell_ijk[4][0],cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]].flatten() )

  regionManager.rk_stage += 1
  a1_pod = 3./4.*a0_pod + 1./4.*(a1_pod + regionManager.dt*R ) #reuse a1 vector
  regionManager.a[rec_stencil_list] = np.dot(regionManager.V_rec,a1_pod)
  #regionManager.a[:] = np.dot(regionManager.V[:,:],a1_pod)

  ## Third Stage
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  R = np.dot(regionManager.VR,regionManager.RHS)
  #R = np.einsum('ij,j->i',regionManager.VR,regionManager.region[0].RHS[cell_ijk[0][0],cell_ijk[1][0],cell_ijk[2][0],cell_ijk[3][0],cell_ijk[4][0],cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]].flatten() )

  af_pod = 1./3.*a0_pod + 2./3.*(a1_pod[:] + regionManager.dt*R )
  regionManager.a[rec_stencil_list] = np.dot(regionManager.V_rec,af_pod*1.)
  #regionManager.a[:] = np.dot(regionManager.V[:,:],af_pod*1.)
  regionManager.t += regionManager.dt
  regionManager.iteration += 1
  regionManager.a_pod[:] = af_pod[:]

  ### Update the global state for saving
  if (regionManager.iteration%regionManager.save_freq == 0):
    regionManager.a[:] = np.dot(regionManager.V,af_pod)




def SSP_RK3_POD_QDEIM2(regionManager,eqns,args=None):
  cell_ijk = regionManager.region[0].cell_ijk
  stencil_list = regionManager.region[0].stencil_list
  stencil_ijk = regionManager.region[0].stencil_ijk
  rec_stencil_list = regionManager.region[0].rec_stencil_list
  cell_list = regionManager.region[0].cell_list
  viscous_stencil_list = regionManager.region[0].viscous_stencil_list

  regionManager.rk_stage = 0
  if (regionManager.iteration == 0):
    print('Starting Simulation')
    regionManager.a0[:] = regionManager.a[:]
    a0_pod = globalDot(regionManager.V.transpose(),regionManager.a0,regionManager)
    regionManager.MR_rec = regionManager.MR[rec_stencil_list]
    regionManager.V_rec = regionManager.V[rec_stencil_list]

  else:
    a0_pod = regionManager.a_pod*1.
  #a0_pod = np.dot(regionManager.V[stencil_list,:].transpose(),regionManager.a[stencil_list]*1.)
  ## First Stage
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  regionManager.rk_stage += 1

  R = np.dot(regionManager.VR,regionManager.RHS[cell_list])
  #R = np.einsum('ij,j->i',regionManager.VR,regionManager.region[0].RHS[cell_ijk[0][0],cell_ijk[1][0],cell_ijk[2][0],cell_ijk[3][0],cell_ijk[4][0],cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]].flatten() )
  a1_pod = a0_pod  + regionManager.dt*R.flatten()#globalDot(regionManager.V.transpose(),regionManager.RHS[:],regionManager) 
  regionManager.a[rec_stencil_list] = np.dot(regionManager.V_rec,a1_pod)
  regionManager.a[:] = np.dot(regionManager.V[:,:],a1_pod)
  ## Second Stage
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  R = np.dot(regionManager.VR,regionManager.RHS[cell_list])
  #R = np.einsum('ij,j->i',regionManager.VR,regionManager.region[0].RHS[cell_ijk[0][0],cell_ijk[1][0],cell_ijk[2][0],cell_ijk[3][0],cell_ijk[4][0],cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]].flatten() )

  regionManager.rk_stage += 1
  a1_pod = 3./4.*a0_pod + 1./4.*(a1_pod + regionManager.dt*R ) #reuse a1 vector
  regionManager.a[rec_stencil_list] = np.dot(regionManager.V_rec,a1_pod)
  #regionManager.a[:] = np.dot(regionManager.V[:,:],a1_pod)

  ## Third Stage
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  R = np.dot(regionManager.VR,regionManager.RHS[cell_list])
  #R = np.einsum('ij,j->i',regionManager.VR,regionManager.region[0].RHS[cell_ijk[0][0],cell_ijk[1][0],cell_ijk[2][0],cell_ijk[3][0],cell_ijk[4][0],cell_ijk[5][0],cell_ijk[6][0],cell_ijk[7][0],cell_ijk[8][0]].flatten() )

  af_pod = 1./3.*a0_pod + 2./3.*(a1_pod[:] + regionManager.dt*R )
  regionManager.a[rec_stencil_list] = np.dot(regionManager.V_rec,af_pod*1.)
  #regionManager.a[:] = np.dot(regionManager.V[:,:],af_pod*1.)
  regionManager.t += regionManager.dt
  regionManager.iteration += 1
  regionManager.a_pod[:] = af_pod[:]

  ### Update the global state for saving
  if (regionManager.iteration%regionManager.save_freq == 0):
    regionManager.a[:] = np.dot(regionManager.V,af_pod)
 


def SSP_RK3_POD_QDEIM_VALIDATE(regionManager,eqns,args=None):
  cell_list = regionManager.region[0].cell_list
  cell_ijk = regionManager.region[0].cell_ijk
  stencil_list = regionManager.region[0].stencil_list
  rec_stencil_list = regionManager.region[0].rec_stencil_list

  regionManager.rk_stage = 0
  regionManager.a0[:] = regionManager.a[:]
#  print('0',np.linalg.norm(regionManager.region[0].a.a))
  if (regionManager.iteration == 0):
    print('Starting Simulation')
    a0_pod = globalDot(regionManager.V.transpose(),regionManager.a0,regionManager)
  else:
    a0_pod = regionManager.a_pod*1.
  #a0_pod = np.dot(regionManager.V[stencil_list,:].transpose(),regionManager.a[stencil_list]*1.)
  ## First Stage
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  regionManager.rk_stage += 1
  Rsample = regionManager.RHS[cell_list]*1.
  R = np.dot(regionManager.VR,Rsample)
  #R = np.dot(regionManager.V.transpose(),Rreconstruct)

  a1_pod = a0_pod  + regionManager.dt*R.flatten()#globalDot(regionManager.V.transpose(),regionManager.RHS[:],regionManager) 

  #regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list,:],a1_pod)
  regionManager.a[:] = np.dot(regionManager.V[:,:],a1_pod)

  ## Second Stage
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions

  Rsample = regionManager.RHS[cell_list]*1.
  R = np.dot(regionManager.VR,Rsample)
  #R = np.dot(regionManager.V.transpose(),Rreconstruct)

  regionManager.rk_stage += 1
  a1_pod = 3./4.*a0_pod + 1./4.*(a1_pod + regionManager.dt*R ) #reuse a1 vector
  #regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list,:],a1_pod)
  regionManager.a[:] = np.dot(regionManager.V[:,:],a1_pod)

  ## Third Stage
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions

  Rsample = regionManager.RHS[cell_list]*1.
  R = np.dot(regionManager.VR,Rsample)
  #R = np.dot(regionManager.V.transpose(),Rreconstruct)

  af_pod = 1./3.*a0_pod + 2./3.*(a1_pod[:] + regionManager.dt*R )
  #regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list,:],af_pod*1.)
  regionManager.a[:] = np.dot(regionManager.V[:,:],af_pod*1.)
  regionManager.t += regionManager.dt
  regionManager.iteration += 1
  regionManager.a_pod[:] = af_pod[:]

  ### Update the global state for saving
  if (regionManager.iteration%regionManager.save_freq == 0):
    regionManager.a[:] = np.dot(regionManager.V,af_pod)

def SSP_RK3_POD_COLLOCATE(regionManager,eqns,args=None):
  print('not yet implemented')

def SSP_RK3_POD(regionManager,eqns,args=None):
  regionManager.rk_stage = 0
  a0 = regionManager.a[:]*1.
  for region in regionManager.region:
    region.a.a[:] = np.sum(region.M[None]*region.a.a[:,None,None,None,None],axis=(5,6,7,8) )
  regionManager.a0[:] = regionManager.a[:]
  a0_pod = globalDot(regionManager.V.transpose(),regionManager.a0,regionManager)
  regionManager.a[:] = a0[:]
  ## First Stage
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  regionManager.rk_stage += 1
  a1_pod = a0_pod  + regionManager.dt*globalDot(regionManager.V.transpose(),regionManager.RHS[:],regionManager) 
  regionManager.a[:] = np.dot(regionManager.V,a1_pod)
  ## Second Stage
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  regionManager.rk_stage += 1
  a1_pod = 3./4.*a0_pod + 1./4.*(a1_pod + globalDot(regionManager.V.transpose(),regionManager.dt*regionManager.RHS[:],regionManager) ) #reuse a1 vector
  regionManager.a[:] = np.dot(regionManager.V,a1_pod)
  ## Third Stage
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  af_pod = 1./3.*a0_pod + 2./3.*(a1_pod[:] + regionManager.dt*globalDot(regionManager.V.transpose(), regionManager.RHS[:],regionManager) )
  regionManager.a[:] = np.dot(regionManager.V,af_pod)

  regionManager.t += regionManager.dt
  regionManager.iteration += 1


def SSP_RK3_POD_unsteady(regionManager,eqns,args=None):
  regionManager.rk_stage = 0
  regionManager.a0[:] = regionManager.a[:]
  a0_pod = globalDot(regionManager.V.transpose(),regionManager.a0,regionManager)
  K0_pod = globalDot(regionManager.V.transpose(),regionManager.K*1.,regionManager)

  ## First Stage
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  regionManager.rk_stage += 1
  a1_pod = a0_pod  + regionManager.dt*globalDot(regionManager.V.transpose(),regionManager.RHS[:],regionManager) 
  K1_pod = K0_pod  + regionManager.dt*globalDot(regionManager.V.transpose(),regionManager.RHS2[:],regionManager) 

  regionManager.a[:] = np.dot(regionManager.V,a1_pod)
  regionManager.K[:] = np.dot(regionManager.V,K1_pod)

  ## Second Stage
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  regionManager.rk_stage += 1
  a1_pod = 3./4.*a0_pod + 1./4.*(a1_pod + globalDot(regionManager.V.transpose(),regionManager.dt*regionManager.RHS[:],regionManager) ) #reuse a1 vector
  K1_pod = 3./4.*K0_pod + 1./4.*(K1_pod + globalDot(regionManager.V.transpose(),regionManager.dt*regionManager.RHS2[:],regionManager) ) 

  regionManager.a[:] = np.dot(regionManager.V,a1_pod)
  regionManager.K[:] = np.dot(regionManager.V,K1_pod)

  ## Third Stage
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  af_pod = 1./3.*a0_pod + 2./3.*(a1_pod[:] + regionManager.dt*globalDot(regionManager.V.transpose(), regionManager.RHS[:],regionManager) )
  Kf_pod = 1./3.*K0_pod + 2./3.*(K1_pod[:] + regionManager.dt*globalDot(regionManager.V.transpose(), regionManager.RHS2[:],regionManager) )

  regionManager.a[:] = np.dot(regionManager.V,af_pod)
  regionManager.K[:] = np.dot(regionManager.V,Kf_pod)

  regionManager.t += regionManager.dt
  regionManager.iteration += 1


def SteadyState(regionManager,eqns,args):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  regionManager.a0[:] = regionManager.a[:]
  #eqns.getRHS(region,region,eqns)
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  R0 = np.zeros(np.shape(regionManager.RHS))
  R0[:] = regionManager.RHS[:]

  def unsteadyResidual(regionManager,v):
    regionManager.a[:] = v[:] #set current state to be the solution iterate of the Newton Krylov solver
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions
    ## Construct the unsteady residual for Crank Nicolson in a few steps
    RHS = np.zeros(np.size(regionManager.RHS))
    R1 = np.zeros(np.size(regionManager.RHS))
    R1[:] = regionManager.RHS[:]
    Rstar = R1*1.
    Rstar_glob = gatherResid(Rstar,regionManager) #gather the residual from all the different mpi_ranks
    return Rstar,R1,Rstar_glob

  ## Function to to a Matrix free approximation to the mat-vec product [dR/du][v] 
  def create_MF_Jacobian(v,args,regionManager):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(regionManager.a))
    eps = 5.e-4
    regionManager.a[:] = an + eps*vr
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
    Av  = (regionManager.RHS - Rn)/eps
    return Av

  nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,regionManager,linear_solver,sparse_quadrature,eqns,None)

  regionManager.t += regionManager.dt
  regionManager.iteration += 1


def SteadyStateExperimental(main,MZ,eqns,args):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  main.a0[:] = main.a.a[:]
  eqns.getRHS(main,main,eqns)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  def unsteadyResidual(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    eqns.getRHS(main,main,eqns)
    R1= np.zeros(np.shape(main.RHS))
    Rstar = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Rstar[:] = main.RHS[:]
    main.basis.applyMassMatrix(main,Rstar)
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  def create_MF_Jacobian(v,args,main):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-2
    main.a.a[:] = an + eps*vr
    eqns.getRHS(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Av = (R1 - Rn)/eps
    main.basis.applyMassMatrix(main,Av)
    return Av.flatten()

  Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a)
  an = np.zeros(np.shape(main.a0))
  an[:] = main.a0[:]
  MF_Jacobian_args = [an,Rn]
  if (main.t == 0):
    sol = linear_solver.solve(create_MF_Jacobian, -Rstarn.flatten(),0.*an.flatten(),main,MF_Jacobian_args,1e-9,200,20,True)
    main.a.a[:] = an[:] + 1.0*np.reshape(sol,np.shape(main.a.a))
  #nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns)
  main.t += main.dt
  main.iteration += 1



## NOT UPDATED FOR MULTIBLOCK YET
def CrankNicolsonEntropy(main,MZ,eqns,args):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  main.a0[:] = main.a.a[:]
  main.getRHS(main,MZ,eqns)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  U0 = entropy_to_conservative(main.a.u)
  t0 = time.time()
  if (main.iteration%1 == 0):
    getEntropyMassMatrix(main)
  if (main.mpi_rank == 0): print('MM time = ' + str(time.time() - t0))
  def unsteadyResidual(main,v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    main.getRHS(main,MZ,eqns)
    U = entropy_to_conservative(main.a.u)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    #if (main.mpi_rank == 0): print(np.amax(R1) )
    ## compute volume integral term
    time_integral = main.basis.volIntegrateGlob(main, (U - U0)*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.w3)
    Rstar = time_integral  - 0.5*main.dt*(R0 + R1)
    Rstar = np.reshape(Rstar,(main.nvars*main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt) )
    Rstar = np.einsum('ij...,j...->i...',main.EMM,Rstar)
    Rstar = np.reshape(Rstar,np.shape(main.a.a))
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,Rstar,Rstar_glob

  def create_MF_Jacobian(v,args,main):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-7
    main.a.a[:] = an + eps*vr
    R1,dum,dum = unsteadyResidual(main,main.a.a)
    Av = (R1 - Rn)/eps
    return Av.flatten()

  nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns)
  main.t += main.dt
  main.iteration += 1

def CrankNicolsonEntropyMZ(main,MZ,eqns,args):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  main.a0[:] = main.a.a[:]
  eqns.getRHS(main,main,eqns)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  main.basis.reconstructU(main,main.a)
  U0 = entropy_to_conservative(main.a.u)*1.
  t0 = time.time()
  getEntropyMassMatrix(main)
  MZ.a.a[:] = 0.
  MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2],:] = main.a.a[:]
  MZ.basis.reconstructU(MZ,MZ.a)

  getEntropyMassMatrix(MZ)
  if (main.mpi_rank == 0): print('MM time = ' + str(time.time() - t0))
  eps = 1.e-3
  def unsteadyResidual(main,v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    ## First compute M^{-1}R(a) @ a = atilde
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2],:] = main.a.a[:]
    eqns.getRHS(MZ,MZ,eqns)
    R1 = np.zeros(np.shape(MZ.RHS))
    R1s = np.zeros(np.shape(MZ.RHS))
    R1[:] =MZ.RHS[:]
    R1s[:] =MZ.RHS[:]
    R1 = np.reshape(R1,(MZ.nvars*MZ.order[0]*MZ.order[1]*MZ.order[2]*MZ.order[3],MZ.Npx,MZ.Npy,MZ.Npz,MZ.Npt) )
    R1 = np.einsum('ij...,j...->i...',MZ.EMM,R1)
    R1 = np.reshape(R1,np.shape(MZ.a.a))
    ## now compute R(a) @ a = atilde + eps M^{-1}R(atilde)
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2],:] = main.a.a[:]
    MZ.a.a[:] += eps*R1
    eqns.getRHS(MZ,MZ,eqns)
    R2 = np.zeros(np.shape(MZ.RHS))
    R2[:] =MZ.RHS[:]
    ## Now compute R(a) @ a = tilde( atilde + eps M^{-1}R(atilde) ) 
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2],:] = main.a.a[:]
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2],:] += eps*R1[:,0:main.order[0],0:main.order[1],0:main.order[2]]
    eqns.getRHS(MZ,MZ,eqns)
    R3 = np.zeros(np.shape(MZ.RHS))
    R3[:] =MZ.RHS[:]
    PLQLu =  1./eps*( R2 - R3)

    #### Now do dynamic procedure for tau
    #==========================================
    testfilter = np.zeros(np.shape(main.a.a))
    testfilterMZ = np.zeros(np.shape(MZ.a.a))
    testscale = np.array([main.order[0]-2,main.order[1]-2,main.order[2]-2])
    testscale[:] = np.fmax(1,testscale)
    testfilter[:,0:testscale[0],0:testscale[1],0:testscale[2]] = 1.
    testfilterMZ[:,0:testscale[0],0:testscale[1],0:testscale[2]] = 1.
    # First compute M^{-1}R(a) @ a = abar
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2],:] = main.a.a[:]*testfilter
    eqns.getRHS(MZ,MZ,eqns)
    R1_dtau = np.zeros(np.shape(MZ.RHS))
    R1s_dtau = np.zeros(np.shape(MZ.RHS))
    R1_dtau[:] =MZ.RHS[:]
    R1s_dtau[:] =MZ.RHS[:]
    R1_dtau = np.reshape(R1_dtau,(MZ.nvars*MZ.order[0]*MZ.order[1]*MZ.order[2]*MZ.order[3],MZ.Npx,MZ.Npy,MZ.Npz,MZ.Npt) )
    R1_dtau = np.einsum('ij...,j...->i...',MZ.EMM,R1_dtau)
    R1_dtau = np.reshape(R1_dtau,np.shape(MZ.a.a))
    ## now compute R(a) @ a = abar + eps M^{-1}R(abar)
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2],:] = main.a.a[:]*testfilter
    MZ.a.a[:] += eps*R1_dtau
    eqns.getRHS(MZ,MZ,eqns)
    R2_dtau = np.zeros(np.shape(MZ.RHS))
    R2_dtau[:] =MZ.RHS[:]
    ## Now compute R(a) @ a = bar( abar + eps M^{-1}R(abar) ) 
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2],:] = main.a.a[:]*testfilter
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2],:] += eps*R1_dtau[:,0:main.order[0],0:main.order[1],0:main.order[2]]*testfilter
    eqns.getRHS(MZ,MZ,eqns)
    R3_dtau = np.zeros(np.shape(MZ.RHS))
    R3_dtau[:] =MZ.RHS[:]
    PLQLu_f = 1./eps*( R2_dtau - R3_dtau)
    # Now compute tau based on entropy transfer, (V^T,PLQLu)
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2],:] = main.a.a[:]*testfilter
    V = MZ.basis.reconstructUGeneral(MZ,MZ.a.a)
    MZ.basis.applyMassMatrix(MZ,PLQLu)
    R1sf = R1s*testfilterMZ
    MZ.basis.applyMassMatrix(MZ,PLQLu_f)
    MZ.basis.applyMassMatrix(MZ,R1s_dtau)
    MZ.basis.applyMassMatrix(MZ,R1sf)

    PLQLu_phys = MZ.basis.reconstructUGeneral(MZ,PLQLu)
    PLQLu_fphys = MZ.basis.reconstructUGeneral(MZ,PLQLu_f)
    R1s_phys = MZ.basis.reconstructUGeneral(MZ,R1sf)#*testfilterMZ)
    R1s_dtauphys = MZ.basis.reconstructUGeneral(MZ,R1s_dtau)

    eqns.getRHS(main,main,eqns)
    tr = np.reshape(main.RHS[:]*1.,(main.nvars*main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt) )
    tr = np.einsum('ij...,j...->i...',main.EMM,tr)
    tr = np.reshape(tr,np.shape(main.a.a))
    test_phys = main.basis.reconstructUGeneral(main,tr)
    print('hi')

    test_phys = entropy_to_conservative(-test_phys)
    Vtest = main.basis.reconstructUGeneral(main,main.a.a)
    def entropyInnerProduct(a,b):
      tmp = (a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + a[4]*b[4])[None,:]*main.Jdet[None,:,:,:,None,:,:,:,None]
      return main.basis.volIntegrate(main.weights0,main.weights1,main.weights2,main.weights3,tmp)
    tester = globalSum( entropyInnerProduct(Vtest,test_phys) , main)
    print(tester)

    numerator = globalSum( entropyInnerProduct(V,R1s_dtauphys - R1s_phys) ,MZ)
    denominator1 = globalSum( entropyInnerProduct(V,PLQLu_phys),MZ)
    denominator2 = globalSum( entropyInnerProduct(V,PLQLu_fphys),MZ)
    scale = (main.order[0]*1./(testscale[0]))**1.5
    tau =  numerator/(denominator1 - scale*denominator2 - 1e-8) 
    #if (main.mpi_rank == 0): print(tau)
    main.tau = tau
    tau = max(tau,0.)
    #if (main.mpi_rank == 0): print(tau)
    #====================================================
 
    ## Now form RHS
    RHS = R1s[:,0:main.order[0],0:main.order[1],0:main.order[2]] #tau/eps*( R2[:,0:main.order[0],0:main.order[1],0:main.order[2]] - R3[:,0:main.order[0],0:main.order[1],0:main.order[2]])
 
    main.basis.reconstructU(main,main.a)
    U = entropy_to_conservative(main.a.u)*1.
    ## compute volume integral term
    time_integral = main.basis.volIntegrateGlob(main, (U - U0)*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.w3)
    Rstar = time_integral  - 0.5*main.dt*(R0 + RHS)
    Rstar = np.reshape(Rstar,(main.nvars*main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt) )
    Rstar = np.einsum('ij...,j...->i...',main.EMM,Rstar)
    Rstar = np.reshape(Rstar,np.shape(main.a.a))
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,Rstar,Rstar_glob

  def create_MF_Jacobian(v,args,main):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-5
    main.a.a[:] = an + eps*vr
    R1,dum,dum = unsteadyResidual(main,main.a.a)
    Av = (R1 - Rn)/eps
    return Av.flatten()

  nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns)
  if (main.mpi_rank == 0): print(main.tau)
  main.t += main.dt
  main.iteration += 1



def backwardEuler_DLS(regionManager,eqns,args=None):
  for region in regionManager.region:
    region.basis.reconstructU(region,region.a)

  u0 = []
  for region  in regionManager.region:
    u0.append(region.a.u*1.)

  ## Function to evaluate the unsteady residual
  def unsteadyResidual(v):
    regionManager.a[:] = v[:] #set current state to be the solution iterate of the Newton Krylov solver
    vol_resid,flux_resid = getRHS_strong(regionManager,eqns) #
    ## Construct the unsteady residual for Crank Nicolson in a few steps
    resid = 1./regionManager.dt*(regionManager.region[0].a.u - u0[0]) + vol_resid[0] 
    resid = resid.flatten()
    resid = np.append(resid,100.*flux_resid)
    #print(np.shape(resid))
    #print(np.linalg.norm(flux_resid))
    return resid
  r0 = unsteadyResidual(regionManager.a)
  Nr = np.size(r0)
  ## Function to to a Matrix free approximation to the mat-vec product [dR/du][v]
  def create_mv(v):
    def Jv(v):
      vr = np.reshape(v,np.shape(regionManager.a))
      eps = 5.e-2
      a0 = regionManager.a[:]*1.
      resid0 = unsteadyResidual(a0)
      a1 = a0 + eps*vr
      resid1 = unsteadyResidual(a1)
      Jv = (resid1 - resid0)/eps
      regionManager.a[:] = a0[:] 
      return Jv.flatten() 
    def vTJ(v):
      return v[0:np.size(regionManager.a)]#regionManager.a0.flatten()
    mdim = Nr
    ndim = np.size(regionManager.a)
    print(mdim,ndim)
    return LinearOperator((mdim,ndim),matvec=Jv,rmatvec=vTJ)

  def Jacobian(v):
    J = np.zeros((Nr,np.size(regionManager.a)))
    eps = 1.#e-3
    r0 = unsteadyResidual(v)
    for i in range(0,np.size(v)):
      v[i] += eps
      rtmp = unsteadyResidual(v)
      J[:,i] = (rtmp - r0)/eps
      v[i] -= eps
    return J
  if (regionManager.iteration == 0):
    J0 = Jacobian(regionManager.a*1.)
    print('Computed Jac!')
    regionManager.jac_pattern = np.ones( (Nr, np.size(regionManager.a))) 
    regionManager.jac_pattern[np.abs(J0) <= 1e-9] = 0.
  sol = scipy.optimize.least_squares(unsteadyResidual,x0=regionManager.a*1.,verbose=2,ftol=1e-10,xtol=1e-10,gtol=1e-10,jac_sparsity=regionManager.jac_pattern,method='trf').x
 
  regionManager.a[:] = sol[:]
  regionManager.t += regionManager.dt
  regionManager.iteration += 1








### FUNCTION THAT ADVANCES THE SOLUTION VIA BACKWARD EULER IMPLICIT TIMESCHEME
## a^{n+1} - a^{n} =  dt*R(a^n)
def backwardEuler(regionManager,eqns,args=None):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  regionManager.a0[:] = regionManager.a[:] 
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  R0 = np.zeros(np.shape(regionManager.RHS))
  R0[:] = regionManager.RHS[:]
  ## Function to evaluate the unsteady residual
  def unsteadyResidual_element(regionManager,v):
    regionManager.a[:] = v[:] #set current state to be the solution iterate of the Newton Krylov solver
    regionManager.getRHS_REGION_INNER_ELEMENT(regionManager,eqns) # evaluate the RHS includes loop over all regions
#    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions
    ## Construct the unsteady residual for Backward Euler in a few steps
    RHS_BE = np.zeros(np.size(regionManager.RHS))
    R1 = np.zeros(np.size(regionManager.RHS))
    R1[:] = regionManager.RHS[:]
    RHS_BE[:] = regionManager.dt*R1
    Rstar = ( regionManager.a[:] - regionManager.a0 ) - RHS_BE
    return Rstar
#  J = computeJacobian_full_element(regionManager,unsteadyResidual_element) 
#  Jinv = np.linalg.inv(J)
  '''
  jac_freq = 1
  if (regionManager.iteration%jac_freq == 0):
    computeJacobian_element(regionManager,unsteadyResidual_element) 
    if (regionManager.mpi_rank == 0):
      sys.stdout.write('Computing Block Jacobians' +  '\n')

  def blockJacobian_PC(v,regionManager,args=None):
    z = np.zeros(np.size(v))
    #computeJacobian_element(regionManager,unsteadyResidual_element) 
    for region in regionManager.region:
      block_size = region.nvars*np.prod(region.order)
      start_indx = regionManager.solution_start_indx[region.region_counter]
      end_indx = regionManager.solution_end_indx[region.region_counter]
      tmp = np.reshape(v[start_indx:end_indx],np.shape(region.a.a))
      tmp = np.reshape(tmp,(block_size,region.Npx,region.Npy,region.Npz,region.Npt))
      tmp = np.einsum('ij...,j...->i...',region.PCinv,tmp)
      z[start_indx:end_indx] = tmp.flatten()
    return z
  '''
  ## Function to evaluate the unsteady residual
  def unsteadyResidual(regionManager,v):
    regionManager.a[:] = v[:] #set current state to be the solution iterate of the Newton Krylov solver
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions
    ## Construct the unsteady residual for Backward Euler in a few steps
    RHS_BE = np.zeros(np.size(regionManager.RHS))
    R1 = np.zeros(np.size(regionManager.RHS))
    R1[:] = regionManager.RHS[:]
    RHS_BE[:] = regionManager.dt*R1
    Rstar = (regionManager.a[:] - regionManager.a0 ) - RHS_BE
#    for region in regionManager.region:
#      block_size = region.nvars*np.prod(region.order)
#      start_indx = regionManager.solution_start_indx[region.region_counter]
#      end_indx = regionManager.solution_end_indx[region.region_counter]
#      tmp = np.reshape(Rstar[start_indx:end_indx],np.shape(region.a.a))
#      tmp = np.reshape(tmp,(block_size,region.Npx,region.Npy,region.Npz,region.Npt))
#      tmp = np.einsum('ij...,j...->i...',region.PCinv,tmp)
#      Rstar[start_indx:end_indx] = tmp.flatten()
    Rstar_glob = gatherResid(Rstar,regionManager) #gather the residual from all the different mpi_ranks
    return Rstar,Rstar,Rstar_glob

  ## Function to to a Matrix free approximation to the mat-vec product [dR/du][v] 
  def create_MF_Jacobian(v,args,regionManager):
    an = args[0]
    Rn = args[1]
    eps = 1.e-5
#    regionManager.a[:] = an[:]
#    R0,dum,dum = unsteadyResidual(regionManager,regionManager.a) #includes loop over all regions
    regionManager.a[:] = an + eps*v
    R1,dum,dum = unsteadyResidual(regionManager,regionManager.a) #includes loop over all regions
    Av = (R1-  Rn)/eps
    return Av 



  def create_MF_Jacobian2(v,args,regionManager):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(regionManager.a))
    eps = 1.e-4
    regionManager.a[:] = an + eps*vr
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
    RHS_BE = np.zeros(np.shape(regionManager.RHS))
    RHS_BE[:] = regionManager.dt*(regionManager.RHS - Rn)/eps
    Av =vr - RHS_BE
    return Av 

  nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,regionManager,linear_solver,sparse_quadrature,eqns)

  regionManager.t += regionManager.dt
  regionManager.iteration += 1


def CrankNicolson(regionManager,eqns,args=None):

  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  regionManager.a0[:] = regionManager.a[:] 
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
  R0 = np.zeros(np.shape(regionManager.RHS))
  R0[:] = regionManager.RHS[:]
  ## Function to evaluate the unsteady residual
  def unsteadyResidual(regionManager,v):
    regionManager.a[:] = v[:] #set current state to be the solution iterate of the Newton Krylov solver
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions
    ## Construct the unsteady residual for Crank Nicolson in a few steps
    RHS_CN = np.zeros(np.size(regionManager.RHS))
    R1 = np.zeros(np.size(regionManager.RHS))
    R1[:] = regionManager.RHS[:]
    RHS_CN[:] = 0.5*regionManager.dt*(R0 + R1)
    Rstar = ( regionManager.a[:] - regionManager.a0 ) - RHS_CN 
    Rstar_glob = gatherResid(Rstar,regionManager) #gather the residual from all the different mpi_ranks
    return Rstar,R1,Rstar_glob

  ## Function to to a Matrix free approximation to the mat-vec product [dR/du][v] 
  def create_MF_Jacobian(v,args,regionManager):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(regionManager.a))
    eps = 5.e-4
    regionManager.a[:] = an + eps*vr
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) #includes loop over all regions
    RHS_CN = np.zeros(np.shape(regionManager.RHS))
    RHS_CN[:] = 0.5*regionManager.dt*(regionManager.RHS - Rn)/eps
    Av = vr - RHS_CN
    return Av

  nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,regionManager,linear_solver,sparse_quadrature,eqns,None)

  regionManager.t += regionManager.dt
  regionManager.iteration += 1

def StrangSplitting(main,MZ,eqns,args):
  ## First run SSP_RK4 for half a time step with no source
  main.fsource = 0.
  main.getRHS(main,MZ,eqns)
  tau = 0.5*main.dt
  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  a1 = main.a.a[:]  + tau*(main.RHS[:])
  main.a.a[:] = a1[:]

  main.getRHS(main,MZ,eqns)
  a1[:] = 3./4.*a0 + 1./4.*(a1 + tau*main.RHS[:]) #reuse a1 vector
  main.a.a[:] = a1[:]

  main.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
  main.a.a[:] = 1./3.*a0 + 2./3.*(a1[:] + tau*main.RHS[:])

  ## Now run implicit on the source term for a full time step
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  main.a0[:] = main.a.a[:]
  getRHS_SOURCE(main,main,eqns)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  def unsteadyResidual(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    getRHS_SOURCE(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Rstar = ( main.a.a[:] - main.a0 ) - 0.5*main.dt*(R0 + R1)
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  def create_MF_Jacobian(v,args,main):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-2
    main.a.a[:] = an + eps*vr
    getRHS_SOURCE(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Av = vr - main.dt/2.*(R1 - Rn)/eps
    return Av.flatten()

  nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns)

  ## Finish with SSP_RK4 for a final half time step
  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  main.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
  a1 = main.a.a[:]  + tau*(main.RHS[:])
  main.a.a[:] = a1[:]

  main.getRHS(main,MZ,eqns)
  a1[:] = 3./4.*a0 + 1./4.*(a1 + tau*main.RHS[:]) #reuse a1 vector
  main.a.a[:] = a1[:]

  main.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
  main.a.a[:] = 1./3.*a0 + 2./3.*(a1[:] + tau*main.RHS[:])


  main.t += main.dt
  main.iteration += 1


def SDIRK2(main,MZ,eqns,args):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  main.a0[:] = main.a.a[:]
  main.getRHS(main,MZ,eqns)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  alpha = (2. - np.sqrt(2.))/2.
  
  def STAGE1_unsteadyResidual(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    main.getRHS(main,MZ,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Rstar = ( main.a.a[:] - main.a0 ) - alpha*main.dt*R1
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  def create_MF_Jacobian(v,args,main):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-2
    main.a.a[:] = an[:] + eps*vr
    main.getRHS(main,MZ,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Av = vr - main.dt*alpha*(R1 - Rn)/eps
    return Av.flatten()
  #stage 1
  nonlinear_solver.solve(STAGE1_unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns)
  main.getRHS(main,MZ,eqns)
  R0[:] = main.RHS[:]
  main.a0[:] = main.a.a[:]
  def STAGE2_unsteadyResidual(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    main.getRHS(main,MZ,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Rstar = ( main.a.a[:] - main.a0 ) - main.dt*( (1. - alpha)*R0 + alpha*R1 )
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  nonlinear_solver.solve(STAGE2_unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns)
  main.t += main.dt
  main.iteration += 1


def SDIRK4(main,MZ,eqns,args):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  main.a0[:] = main.a.a[:]
  main.getRHS(main,MZ,eqns)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  gam = 9./40.
  c2 = 7./13.
  c3 = 11./15.
  b2 = -(-2. + 3.*c3 + 9.*gam - 12.*c3*gam - 6.*gam**2 + 6.*c3*gam**2)/(6.*(c2 - c3)*(c2 - gam))
  b3 =  (-2. + 3.*c2 + 9.*gam - 12.*c2*gam - 6.*gam**2 + 6.*c2*gam**2)/(6.*(c2 - c3)*(c3 - gam))
  a32 = -(c2 - c3)*(c3 - gam)*(-1. + 9.*gam - 18.*gam**2 + 6.*gam**3)/ \
       ( (c2 - gam)*(-2. + 3.*c2 + 9.*gam - 12.*c2*gam - 6.*gam**2 + 6.*c2*gam**2) )

  def create_MF_Jacobian(v,args,main):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-2
    main.a.a[:] = an[:] + eps*vr
    main.getRHS(main,MZ,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Av = vr - main.dt*gam*(R1 - Rn)/eps
    return Av.flatten()


  #========== STAGE 1 ======================

  def STAGE1_unsteadyResidual(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    main.getRHS(main,MZ,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Rstar = ( main.a.a[:] - main.a0 ) - gam*main.dt*R1
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  nonlinear_solver.solve(STAGE1_unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns)
  main.getRHS(main,MZ,eqns)
  R1 = np.zeros( np.shape(main.RHS) )
  R1[:] = main.RHS[:]
  main.a0[:] = main.a.a[:]

  #========= STAGE 2 ==========================
  def STAGE2_unsteadyResidual(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    main.getRHS(main,MZ,eqns)
    R2 = np.zeros(np.shape(main.RHS))
    R2[:] = main.RHS[:]
    Rstar = ( main.a.a[:] - main.a0 ) - main.dt*( (c2 - gam)*R1 + gam*R2 )
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R2,Rstar_glob

  nonlinear_solver.solve(STAGE2_unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns)
  main.getRHS(main,MZ,eqns)
  R2 = np.zeros( np.shape(main.RHS) )
  R2[:] = main.RHS[:]
  main.a0[:] = main.a.a[:]

  #============ STAGE 3 ===============
  def STAGE3_unsteadyResidual(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    main.getRHS(main,MZ,eqns)
    R3 = np.zeros(np.shape(main.RHS))
    R3[:] = main.RHS[:]
    Rstar = ( main.a.a[:] - main.a0 ) - main.dt*( (c3 - a32 -  gam)*R1 + a32*R2 + gam*R3 )
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R3,Rstar_glob

  nonlinear_solver.solve(STAGE3_unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns)
  main.getRHS(main,MZ,eqns)
  R3 = np.zeros( np.shape(main.RHS) )
  R3[:] = main.RHS[:]
  main.a0[:] = main.a.a[:]

  #============ STAGE 4 ============
  def STAGE4_unsteadyResidual(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    main.getRHS(main,MZ,eqns)
    R4 = np.zeros(np.shape(main.RHS))
    R4[:] = main.RHS[:]
    Rstar = ( main.a.a[:] - main.a0 ) - main.dt*( (1. - b2 - b3 - gam)*R1 + b2*R2 + b3*R3 + gam*R4 )
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R4,Rstar_glob
  nonlinear_solver.solve(STAGE4_unsteadyResidual,create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns)
  main.t += main.dt
  main.iteration += 1


### This is for PETSc. It's not MPI so don't use - just have it for comparrison
def advanceSolImplicit_PETsc(main,MZ,eqns):
  old = np.zeros(np.shape(main.a0))
  old[:] = main.a0[:]
  main.a0[:] = main.a.a[:]
  main.getRHS(main,MZ,eqns)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  def unsteadyResid( snes, V, R):
    v = V[...] #+ main.a0.flatten()
    Resid = R[...]
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    main.getRHS(main,MZ,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Resid[:] = (v - main.a0.flatten() ) - 0.5*main.dt*(R0 + R1).flatten()


  # create PETSc nonlinear solver
  snes = PETSc.SNES().create()
  # register the function in charge of
  # computing the nonlinear residual
  f = PETSc.Vec().createSeq(np.size(main.a.a))
  snes.setFunction(unsteadyResid, f)
  snes.setType('newtonls')
  # configure the nonlinear solver
  # to use a matrix-free Jacobian
  snes.setUseMF(True)
  snes.getKSP().setType('gmres')
  snes.ksp.setGMRESRestart(50)
  opts = PETSc.Options()
  opts['snes_ngmres_restart_it'] = 80
  opts['snes_ngmres_m'] = 80
  opts['snes_ngmres_gammaC'] = 1e-10
  opts['snes_ngmres_epsilonB'] = 1e-10
  opts['snes_ngmres_deltaB'] = 1e-10
  opts['snes_monitor']  = ''
  opts['ksp_monitor']   = ''
  snes.setFromOptions()
  def monitor(snes, its, fgnorm):
    print(its,fgnorm)

  snes.setMonitor(monitor)
  snes.setTolerances(1e-7,1e-8,1e-7)
  snes.getKSP().setTolerances(1e-3,1e-50,10000.0,10000)
  tols2 = snes.getKSP().getTolerances()
#  snes_ngmres_restart_it
  tols = snes.getTolerances()
  # solve the nonlinear problem
  b, x = None, f.duplicate()
  x[...] = main.a.a[:]
#  x.set(0) # zero inital guess
  snes.solve(b, x)
  its = snes.getIterationNumber()
  lits = snes.getLinearSolveIterations()

  #print "Number of SNES iterations = %d" % its
  #print "Number of Linear iterations = %d" % lits

  sol = x[...]
  main.a.a =   np.reshape(sol,np.shape(main.a.a))
  main.a0[:] = np.reshape(sol,np.shape(main.a.a))
  main.t += main.dt
  main.iteration += 1





## function to implement LSPG for backward euler timescheme with custom least squares solver
## uses scipys least_squares functionality
def backwardEuler_LSPG(regionManager,eqns,args):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  #regionManager.a0[:] = regionManager.a[:]
  #regionManager.a[:] = np.dot(regionManager.V,np.dot(regionManager.V.transpose(),regionManager.a))
  a0 = regionManager.a[:]*1.
  for region in regionManager.region:
    region.a.a[:] = np.sum(region.M[None]*region.a.a[:,None,None,None,None],axis=(5,6,7,8) )
  regionManager.a0[:] = regionManager.a[:]
  a0_pod = globalDot(regionManager.V.transpose(),regionManager.a0,regionManager)
  regionManager.a[:] = a0[:]


  regionManager.a0[:] = regionManager.a[:]
  print('here')
  print(np.linalg.norm(regionManager.a))
  def unsteadyResidual(v):
    regionManager.a[:] = v[:]# np.dot(regionManager.V,v[:]) #set current state to be the solution iterate of the Newton Krylov solver
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions
    ## Construct the unsteady residual for Backward Euler in a few steps
    RHS_BE = np.zeros(np.size(regionManager.RHS))
    R1 = np.zeros(np.size(regionManager.RHS))
    R1[:] = regionManager.RHS[:]
    RHS_BE[:] = regionManager.dt*R1
    Rstar = 0.*( regionManager.a[:]*1. - regionManager.a0[:] ) - RHS_BE
#    Rstar_glob = gatherResid(Rstar,regionManager) #gather the residual from all the different mpi_ranks
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


## GNAT Crank Nicolson 
def crankNicolson_GNAT(regionManager,eqns,args):
  if (regionManager.iteration == 0):
    print('Starting Simulation')
    a0_pod = globalDot(regionManager.V.transpose(),regionManager.a,regionManager)
    regionManager.jacobian_update_freq = 10
    regionManager.jacobian_iteration = 0
  else:
    a0_pod = regionManager.a_pod*1.
  regionManager.a_pod[:] = a0_pod[:]*1.
  cell_list = regionManager.region[0].cell_list
  rec_stencil_list = regionManager.region[0].rec_stencil_list
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  regionManager.a0[rec_stencil_list] = regionManager.a[rec_stencil_list]*1.
  ## Get initial RHS
  
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling QDEIM
  R0 = np.zeros(np.size(regionManager.RHS[cell_list]))
  R0[:] = regionManager.RHS[cell_list]

  def unsteadyResidual(v):
    regionManager.a[rec_stencil_list] = v[:]# np.dot(regionManager.V[rec_stencil_list],v[:]) #set current state to be the solution iterate of the Newton Krylov solver
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling QDEIM
    ## Construct the unsteady residual for Crank Nicolson in a few steps
    R1 = np.zeros(np.size(regionManager.RHS[cell_list]))
    R1[:] = regionManager.RHS[cell_list]
    ## compute the residual at the sample points
    Rstar = ( regionManager.a[cell_list]*1. - regionManager.a0[cell_list] ) - 0.5*regionManager.dt*(R1 + R0)
    return Rstar.flatten()
  N = np.size(regionManager.a)
  nx,nbasis = np.shape(regionManager.V[cell_list])
  JV = np.zeros((nx,nbasis) )
  regionManager.an = regionManager.a[rec_stencil_list]*1.
  # compute residual
  r = unsteadyResidual(regionManager.a[rec_stencil_list])
  da_norm = 1.
  #r_norm = globalNorm(r,regionManager)
  ls_iteration = 0
  alpha = 1.
  rbar = np.dot(regionManager.ZW_pinv,r)
  r_norm = globalNorm(rbar,regionManager)
  if (regionManager.mpi_rank == 0):
    print('Initial Residual = ' + str(r_norm) )

  while ( r_norm > 1e-6 and da_norm >= 1e-6):
   if (ls_iteration >= 20 or alpha <= 0.01):
     print('Didnt converge, alpha = ' + str(alpha) ,'  iteration count = ' + str(ls_iteration) )
     da_norm = 0.
   else:
    r_norm_old = r_norm*1.
    rbar = np.dot(regionManager.ZW_pinv,r)
    eps = 1e-5
    an = regionManager.a[rec_stencil_list]*1.
    if (ls_iteration%5 == 0 and regionManager.iteration%10 == 0):
     for i in range(0,nbasis):
      regionManager.a[rec_stencil_list] = an[:] + eps*regionManager.V[rec_stencil_list,i]
      Jv_column = unsteadyResidual(regionManager.a[rec_stencil_list])
      JV[:,i] = (Jv_column - r[:]) / eps
     regionManager.JV = JV
    JV = regionManager.JV 
    ## now compute Jbar and rbar
    JVbar = np.dot(regionManager.ZW_pinv,JV)

    Q,R = np.linalg.qr(JVbar)

    da_pod = np.linalg.solve(R,-np.dot(Q.transpose(),rbar) )
#    JSQ = np.dot(JVbar.transpose(),JVbar)
#    print(numpy.linalg.cond(JSQ))
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
#    JVrbar = globalDot(JVbar.transpose(),rbar,regionManager)
#    da_pod = np.linalg.solve(JSQ_glob,-JVrbar)
    da_norm = np.linalg.norm(da_pod)
    regionManager.a_pod[:] += alpha*da_pod[:]
    regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list],regionManager.a_pod[:])
    r = unsteadyResidual(regionManager.a[rec_stencil_list])
    r_norm = globalNorm(rbar,regionManager)
#    if (r_norm >= r_norm_old and ls_iteration > 0):
#      print('Residual grew, lowering step size')
#      regionManager.a_pod[:] -= alpha*da_pod[:]
#      regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list],regionManager.a_pod[:])
#      r = unsteadyResidual(regionManager.a[rec_stencil_list])
#      alpha /= 2.
    ls_iteration += 1
    if (regionManager.mpi_rank == 0):
      print('Residual = ' + str(r_norm), ' Change to solution is ' + str(da_norm))

  regionManager.t += regionManager.dt
  regionManager.iteration += 1

  ### Update the global state for saving
  if (regionManager.iteration%regionManager.save_freq == 0):
    #regionManager.a_pod[:] = np.dot(regionManager.V[cell_list].transpose(),regionManager.a[cell_list] )
    regionManager.a[:] = np.dot(regionManager.V,regionManager.a_pod)





## Collocated LSPG with Crank Nicolson 
def crankNicolson_LSPG_QDEIM(regionManager,eqns,args):
  if (regionManager.iteration == 0):
    print('Starting Simulation')
    a0_pod = globalDot(regionManager.V.transpose(),regionManager.a,regionManager)
    regionManager.jacobian_update_freq = 10
    regionManager.jacobian_iteration = 0
  else:
    a0_pod = regionManager.a_pod*1.
  regionManager.a_pod[:] = a0_pod[:]*1.
  cell_list = regionManager.region[0].cell_list
  rec_stencil_list = regionManager.region[0].rec_stencil_list
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  regionManager.a0[rec_stencil_list] = regionManager.a[rec_stencil_list]*1.
  ## Get initial RHS
  
  regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling QDEIM
  R0 = np.zeros(np.size(regionManager.RHS[cell_list]))
  R0[:] = regionManager.RHS[cell_list]

  def unsteadyResidual(v):
    regionManager.a[rec_stencil_list] = v[:]# np.dot(regionManager.V[rec_stencil_list],v[:]) #set current state to be the solution iterate of the Newton Krylov solver
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling QDEIM
    ## Construct the unsteady residual for Crank Nicolson in a few steps
    R1 = np.zeros(np.size(regionManager.RHS[cell_list]))
    R1[:] = regionManager.RHS[cell_list]
    ## compute the residual at the sample points
    Rstar = ( regionManager.a[cell_list]*1. - regionManager.a0[cell_list] ) - 0.5*regionManager.dt*(R1 + R0)
    return Rstar.flatten()
  N = np.size(regionManager.a)
  nx,nbasis = np.shape(regionManager.V[cell_list])
  JV = np.zeros((nx,nbasis) )
  regionManager.an = regionManager.a[rec_stencil_list]*1.
  # compute residual
  r = unsteadyResidual(regionManager.a[rec_stencil_list])
  da_norm = 1.
  ls_iteration = 0
  r_norm = globalNorm(r,regionManager)
  alpha = 1.
  if (regionManager.mpi_rank == 0):
    print('Initial Residual = ' + str(r_norm) )
  while (r_norm > 1e-6 and da_norm >= 1e-6):
   if (ls_iteration >= 20 or alpha <= 0.01):
     print('Didnt converge, alpha = ' + str(alpha) ,'  iteration count = ' + str(ls_iteration) )
     da_norm = 0.
   else:
    eps = 1e-5
    an = regionManager.a[rec_stencil_list]*1.
    if (ls_iteration%1 == 0 and regionManager.iteration%1 == 0):
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
#    JVr = globalDot(JV.transpose(),r,regionManager)
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
    da_norm = np.linalg.norm(da_pod)
    regionManager.a_pod[:] += da_pod[:]
    regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list],regionManager.a_pod[:])
    r = unsteadyResidual(regionManager.a[rec_stencil_list])
    r_norm = globalNorm(r,regionManager)
    ls_iteration += 1
    if (regionManager.mpi_rank == 0):
      print('Residual = ' + str(r_norm), ' Change to solution is ' + str(da_norm))

  regionManager.t += regionManager.dt
  regionManager.iteration += 1

  ### Update the global state for saving
  if (regionManager.iteration%regionManager.save_freq == 0):
    #regionManager.a_pod[:] = np.dot(regionManager.V[cell_list].transpose(),regionManager.a[cell_list] )
    regionManager.a[:] = np.dot(regionManager.V,regionManager.a_pod)


## using QDEIM 
def backwardEuler_LSPG_collocation_validate(regionManager,eqns,args):
  if (regionManager.iteration == 0):
    print('Starting Simulation')
    a0_pod = globalDot(regionManager.V.transpose(),regionManager.a,regionManager)
    regionManager.jacobian_update_freq = 1
    regionManager.jacobian_iteration = 0
  else:
    a0_pod = regionManager.a_pod*1.
  regionManager.a_pod[:] = a0_pod[:]*1.
  cell_list = regionManager.region[0].cell_list
  rec_stencil_list = regionManager.region[0].rec_stencil_list
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  regionManager.a0[rec_stencil_list] = regionManager.a[rec_stencil_list]*1.
  def unsteadyResidual(v):
    regionManager.a[rec_stencil_list] = v[:]# np.dot(regionManager.V[rec_stencil_list],v[:]) #set current state to be the solution iterate of the Newton Krylov solver
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling QDEIM
    ## Construct the unsteady residual for Backward Euler in a few steps
    RHS_BE = np.zeros(np.size(regionManager.RHS[cell_list]))
    R1 = np.zeros(np.size(regionManager.RHS[cell_list]))
    R1[:] = regionManager.RHS[cell_list]
    RHS_BE[:] = regionManager.dt*R1
    Rstar = ( regionManager.a[cell_list]*1. - regionManager.a0[cell_list] ) - RHS_BE
    return Rstar.flatten()

  nx,nbasis = np.shape(regionManager.V[cell_list])
  JV = np.zeros((nx,nbasis) )
  regionManager.an = regionManager.a[rec_stencil_list]*1.
  # compute residual
  r = unsteadyResidual(regionManager.a[rec_stencil_list])
  da_norm = 1.
  r_norm = globalNorm(r,regionManager)
  if (regionManager.mpi_rank == 0):
    print('Initial Residual = ' + str(r_norm) )

  while (r_norm > 1e-4 and da_norm >= 1e-4):
    eps = 1e-5
    an = regionManager.a[rec_stencil_list]*1.
    for i in range(0,nbasis):
      regionManager.a[rec_stencil_list] = an[:] + eps*regionManager.V[rec_stencil_list,i]
      Jv_column = unsteadyResidual(regionManager.a[rec_stencil_list])
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
    #print(np.linalg.norm(da_pod))
    regionManager.a_pod[:] += da_pod[:]
    regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list],regionManager.a_pod[:])
    #regionManager.a[rec_stencil_list] = regionManager.a0[rec_stencil_list]
    #regionManager.a[rec_stencil_list] = regionManager.a[rec_stencil_list] + np.dot(regionManager.V[rec_stencil_list],da_pod[:])
    r = unsteadyResidual(regionManager.a[rec_stencil_list])
    r_norm = globalNorm(r,regionManager)
    if (regionManager.mpi_rank == 0):
      print('Residual = ' + str(r_norm), ' Change to solution is ' + str(da_norm))

  regionManager.t += regionManager.dt
  regionManager.iteration += 1

  ### Update the global state for saving
  if (regionManager.iteration%regionManager.save_freq == 0):
    #regionManager.a_pod[:] = np.dot(regionManager.V[cell_list].transpose(),regionManager.a[cell_list] )
    regionManager.a[:] = np.dot(regionManager.V,regionManager.a_pod)




## using QDEIM 
def backwardEuler_LSPG_QDEIM(regionManager,eqns,args):
  if (regionManager.iteration == 0):
    print('Starting Simulation')
    a0_pod = globalDot(regionManager.V.transpose(),regionManager.a,regionManager)
    regionManager.jacobian_update_freq = 1
    regionManager.jacobian_iteration = 0
  else:
    a0_pod = regionManager.a_pod*1.
  regionManager.a_pod[:] = a0_pod[:]*1.
  cell_list = regionManager.region[0].cell_list
  rec_stencil_list = regionManager.region[0].rec_stencil_list
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  regionManager.a0[rec_stencil_list] = regionManager.a[rec_stencil_list]*1.
  def unsteadyResidual(v):
    regionManager.a[rec_stencil_list] = v[:]# np.dot(regionManager.V[rec_stencil_list],v[:]) #set current state to be the solution iterate of the Newton Krylov solver
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling QDEIM
    ## Construct the unsteady residual for Backward Euler in a few steps
    RHS_BE = np.zeros(np.size(regionManager.RHS[cell_list]))
    R1 = np.zeros(np.size(regionManager.RHS[cell_list]))
    R1[:] = regionManager.RHS[cell_list]
    RHS_BE[:] = regionManager.dt*R1
    Rstar = ( regionManager.a[cell_list]*1. - regionManager.a0[cell_list] ) - RHS_BE
    return Rstar.flatten()

  nx,nbasis = np.shape(regionManager.V[cell_list])
  JV = np.zeros((nx,nbasis) )
  regionManager.an = regionManager.a[rec_stencil_list]*1.
  # compute residual
  r = unsteadyResidual(regionManager.a[rec_stencil_list])
  da_norm = 1.
  r_norm = globalNorm(r,regionManager)
  if (regionManager.mpi_rank == 0):
    print('Initial Residual = ' + str(r_norm) )

  while (r_norm > 1e-4 and da_norm >= 1e-4):
    eps = 1e-5
    an = regionManager.a[rec_stencil_list]*1.
    for i in range(0,nbasis):
      regionManager.a[rec_stencil_list] = an[:] + eps*regionManager.V[rec_stencil_list,i]
      Jv_column = unsteadyResidual(regionManager.a[rec_stencil_list])
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
    #print(np.linalg.norm(da_pod))
    regionManager.a_pod[:] += da_pod[:]
    regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list],regionManager.a_pod[:])
    #regionManager.a[rec_stencil_list] = regionManager.a0[rec_stencil_list]
    #regionManager.a[rec_stencil_list] = regionManager.a[rec_stencil_list] + np.dot(regionManager.V[rec_stencil_list],da_pod[:])
    r = unsteadyResidual(regionManager.a[rec_stencil_list])
    r_norm = globalNorm(r,regionManager)
    if (regionManager.mpi_rank == 0):
      print('Residual = ' + str(r_norm), ' Change to solution is ' + str(da_norm))

  regionManager.t += regionManager.dt
  regionManager.iteration += 1

  ### Update the global state for saving
  if (regionManager.iteration%regionManager.save_freq == 0):
    #regionManager.a_pod[:] = np.dot(regionManager.V[cell_list].transpose(),regionManager.a[cell_list] )
    regionManager.a[:] = np.dot(regionManager.V,regionManager.a_pod)


## uses scipys least_squares functionality
def backwardEuler_LSPG_QDEIM_SCIPY(regionManager,eqns,args):
  if (regionManager.iteration == 0):
    print('Starting Simulation')
    a0_pod = globalDot(regionManager.V.transpose(),regionManager.a,regionManager)
    regionManager.jacobian_update_freq = 10
    regionManager.jacobian_iteration = 0
  else:
    a0_pod = regionManager.a_pod*1.
  regionManager.a_pod[:] = a0_pod[:]*1.
  cell_list = regionManager.region[0].cell_list
  rec_stencil_list = regionManager.region[0].rec_stencil_list
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  regionManager.a0[rec_stencil_list] = regionManager.a[rec_stencil_list]*1.
  def unsteadyResidual(v):
    regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list], v[:]  )
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling QDEIM
    ## Construct the unsteady residual for Backward Euler in a few steps
    RHS_BE = np.zeros(np.size(regionManager.RHS[cell_list]))
    R1 = np.zeros(np.size(regionManager.RHS[cell_list]))
    R1[:] = regionManager.RHS[cell_list]
    RHS_BE[:] = regionManager.dt*R1
    Rstar = ( regionManager.a[cell_list]*1. - regionManager.a0[cell_list] ) - RHS_BE
    return Rstar.flatten()

  res_1 = least_squares(unsteadyResidual, regionManager.a_pod,ftol=1e-4,xtol=1e-8,method='dogbox',verbose=2)

  regionManager.a_pod[:] = res_1.x[:]
  regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list],regionManager.a_pod[:])

  regionManager.t += regionManager.dt
  regionManager.iteration += 1

  ### Update the global state for saving
  if (regionManager.iteration%regionManager.save_freq == 0):
    #regionManager.a_pod[:] = np.dot(regionManager.V[cell_list].transpose(),regionManager.a[cell_list] )
    regionManager.a[:] = np.dot(regionManager.V,regionManager.a_pod)




## uses scipys least_squares functionality
def backwardEuler_Galerkin_QDEIM(regionManager,eqns,args):
  if (regionManager.iteration == 0):
    print('Starting Simulation')
    a0_pod = globalDot(regionManager.V.transpose(),regionManager.a,regionManager)
    regionManager.jacobian_update_freq = 10
    regionManager.jacobian_iteration = 0
  else:
    a0_pod = regionManager.a_pod*1.
  regionManager.a_pod[:] = a0_pod[:]*1.
  cell_list = regionManager.region[0].cell_list
  rec_stencil_list = regionManager.region[0].rec_stencil_list
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  regionManager.a0[rec_stencil_list] = regionManager.a[rec_stencil_list]*1.
  def unsteadyResidual(v):
    regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list], v[:]  )
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling QDEIM
    ## Construct the unsteady residual for Backward Euler in a few steps
    RHS_BE = np.zeros(np.size(regionManager.RHS[cell_list]))
    R1 = np.zeros(np.size(regionManager.RHS[cell_list]))
    R1[:] = regionManager.RHS[cell_list]
    RHS_BE[:] = regionManager.dt*R1
    Rstar = np.dot(regionManager.V[cell_list].transpose(), ( regionManager.a[cell_list]*1. - regionManager.a0[cell_list] ) - RHS_BE )
    return Rstar.flatten()

  res = scipy.optimize.newton_krylov(unsteadyResidual,regionManager.a_pod,verbose=4)
  regionManager.a_pod[:] = res[:] 


  regionManager.a[rec_stencil_list] = np.dot(regionManager.V[rec_stencil_list],regionManager.a_pod[:])

  regionManager.t += regionManager.dt
  regionManager.iteration += 1
  ### Update the global state for saving
  if (regionManager.iteration%regionManager.save_freq == 0):
    #regionManager.a_pod[:] = np.dot(regionManager.V[cell_list].transpose(),regionManager.a[cell_list] )
    regionManager.a[:] = np.dot(regionManager.V,regionManager.a_pod)




## function to implement LSPG for backward euler timescheme
## uses scipys least_squares functionality
def backwardEuler_LSPG2(regionManager,eqns,args):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  regionManager.a0[:] = regionManager.a[:]
  regionManager.getRHS_REGION_OUTER(regionManager,eqns)
  R0 = np.zeros(np.size(regionManager.RHS))
  R0[:] = regionManager.RHS[:]
  print(np.linalg.norm(regionManager.a))
  def unsteadyResidual(v):
    regionManager.a[:] = np.dot(regionManager.V,v[:]) #set current state to be the solution iterate of the Newton Krylov solver
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions
    ## Construct the unsteady residual for Backward Euler in a few steps
    RHS_BE = np.zeros(np.size(regionManager.RHS))
    R1 = np.zeros(np.size(regionManager.RHS))
    R1[:] = regionManager.RHS[:]
    RHS_BE[:] = regionManager.dt*R1
    Rstar = ( regionManager.a[:]*1. - regionManager.a0[:] ) - RHS_BE
#    Rstar_glob = gatherResid(Rstar,regionManager) #gather the residual from all the different mpi_ranks
    return np.sum((np.abs(Rstar.flatten())))

  def create_mv(v):
    def MF_Jacobian(v):
      an = regionManager.a0*1.
      #vr = np.reshape( np.dot(main.V,v) , np.shape(main.a.a) )
      ##vr = np.reshape(v,np.shape(main.a.a))
      #eps = 5.e-7
      #main.a.a[:] = an + eps*vr
      #main.getRHS(main,MZ,eqns)
      #RHS_BE = np.zeros(np.shape(main.RHS))
      #R1 = np.zeros(np.shape(main.RHS))
      #R1[:] = main.RHS[:]
      #RHS_BE[:] = main.dt*(R1 - Rn)/eps
      #main.basis.applyMassMatrix(main,RHS_BE)
      #Av = vr - RHS_BE
      return np.dot(regionManager.V,v)
      #return Av.flatten()

    def rmatvec(v):
      return globalDot(regionManager.V.transpose(),v,regionManager)
    mdim = np.size(regionManager.a)
    ndim = np.shape(regionManager.V)[1]
    return  LinearOperator((mdim,ndim),matvec=MF_Jacobian,rmatvec=rmatvec)

  #a0_pod = globalDot(regionManager.V.transpose(),regionManager.a.flatten(),regionManager)
  a0 = regionManager.a[:]*1.
  for region in regionManager.region:
    region.a.a[:] = np.sum(region.M[None]*region.a.a[:,None,None,None,None],axis=(5,6,7,8) )
  regionManager.a0[:] = regionManager.a[:]
  a0_pod = globalDot(regionManager.V.transpose(),regionManager.a0,regionManager)

  #A = create_mv(regionManager.a.flatten())
  #res_1 = least_squares(unsteadyResidual, a0_pod,ftol=1e-4,xtol=1e-8,jac=create_mv,method='dogbox',verbose=2)
  res_1 = least_squares(unsteadyResidual, a0_pod,ftol=1e-4,xtol=1e-8,method='dogbox',verbose=2)
  #res_1 = scipy.optimize.minimize(unsteadyResidual, a0_pod)

  a_sol_pod = res_1.x
  regionManager.a[:] = np.dot(regionManager.V,a_sol_pod)
  regionManager.t += regionManager.dt
  regionManager.iteration += 1


## function to implement LSPG for backward euler timescheme
## uses scipys least_squares functionality
def crankNicolson_LSPG_windowed(regionManager,eqns,args):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  regionManager.a0[:] = regionManager.a[:]
  regionManager.getRHS_REGION_OUTER(regionManager,eqns)
  R0 = np.zeros(np.size(regionManager.RHS))
  R0[:] = regionManager.RHS[:]
  nwindows = regionManager.tau
  def unsteadyResidual(v):
    K = np.shape(regionManager.V)[1]
    Kt = np.shape(regionManager.Vt)[1]

    #vw = np.reshape(v,(K,Kt),order='F')
    vw = np.reshape(v,(K,Kt))

    ## reconstruct generalized coordinates
    #a_tmp = np.einsum('ij,jl->il',regionManager.V,vw)
    #a_spacetime = np.einsum('kl,il->ik',regionManager.Vt,a_tmp)
    a_spacetime = np.einsum('ijkl,jl->ik',VST,vw)
    ### DO FIRST LSPG WINDOW   
    an  = a_spacetime[:,0] #set current state to be the solution iterate of the Newton Krylov solver
    regionManager.a[:] = an[:]
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions
    ## Construct the unsteady residual for Backward Euler in a few steps
    RHS_BE = np.zeros(np.size(regionManager.RHS))
    R1 = np.zeros(np.size(regionManager.RHS))
    R1[:] = regionManager.RHS[:]
    #Rstar = ( regionManager.a[:]*1. - regionManager.a0[:] ) - RHS_BE
    R0m = R0*1.
#    Rstar = ( regionManager.a[:] - regionManager.a0[:] ) - 0.5*regionManager.dt*(R1 + R0m)
    Rstar = ( regionManager.a[:] - regionManager.a0[:] ) - regionManager.dt*(R1)
    Rcombined = Rstar.flatten()*1.
   ## now compute the same residual over all windows
    for window in regionManager.Pt:
     if (window == 0):
      pass
     else:
    #for window in range(1,regionManager.tau):

      #anp1 = a_spacetime[:,window] 
      regionManager.a[:] = a_spacetime[:,window]*1.
      regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions
      #RHS_BE2 = np.zeros(np.size(regionManager.RHS))
      #R0m[:] = R1*1.
      R1 = regionManager.RHS[:]*1.

      #regionManager.a[:] = a_spacetime[:,window-1]
      #regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions
      #R0[:] = regionManager.RHS[:]

      #Rstar[:] = ( regionManager.a[:]*1. - a_spacetime[:,window-1] )  - 0.5*regionManager.dt*(R1[:] + R0[:])
      Rstar = ( a_spacetime[:,window] - a_spacetime[:,window-1] )  - regionManager.dt*(R1[:])
      Rcombined = np.append(Rcombined.flatten(),1.*Rstar[:].flatten())
    return Rcombined

  def create_mv(v):
    def MF_Jacobian(v):
      an = regionManager.a0*1
      N,K = np.shape(regionManager.V)[0],np.shape(regionManager.V)[1]
      V2 = np.zeros((nwindows*N,nwindows*K))
      for i in range(0,nwindows):
#        print(i*N,(i+1)*N,i*K,(i+1)*K,np.shape(V2),np.shape(V2[:,i*K:(i+1)*K]))
        V2[i*N:(i+1)*N,i*K:(i+1)*K] = regionManager.V[:,:]
      #V2[0:N,K] = 
      #vr = np.reshape( np.dot(main.V,v) , np.shape(main.a.a) )
      ##vr = np.reshape(v,np.shape(main.a.a))
      #eps = 5.e-7
      #main.a.a[:] = an + eps*vr
      #main.getRHS(main,MZ,eqns)
      #RHS_BE = np.zeros(np.shape(main.RHS))
      #R1 = np.zeros(np.shape(main.RHS))
      #R1[:] = main.RHS[:]
      #RHS_BE[:] = main.dt*(R1 - Rn)/eps
      #main.basis.applyMassMatrix(main,RHS_BE)
      #Av = vr - RHS_BE
      return np.dot(V2,v)
      #return Av.flatten()

    def rmatvec(v):
      an = regionManager.a0*1
      N,K = np.shape(regionManager.V)[0],np.shape(regionManager.V)[1]
      V2 = np.zeros((nwindows*N,nwindows*K))
      for i in range(0,nwindows):
        V2[i*N:(i+1)*N,i*K:(i+1)*K] = regionManager.V[:]
      return globalDot(V2.transpose(),v,regionManager)

    mdim = np.size(regionManager.a)*nwindows
    ndim = np.shape(regionManager.V)[1]*nwindows
    return  LinearOperator((mdim,ndim),matvec=MF_Jacobian,rmatvec=rmatvec)

  ## make windowed ST generalized coordinates
  # normal spacetime projection
  regionManager.V = regionManager.VsG[regionManager.window_counter]
  regionManager.Vt = regionManager.VtG[regionManager.window_counter]
  regionManager.Pt = regionManager.PtG[regionManager.window_counter]

  #regionManager.Vt = np.eye(nwindows)
  K = np.shape(regionManager.V)[1]
  Kt = np.shape(regionManager.Vt)[1]
  #print(regionManager.Vt)
  VST = regionManager.V[:,:,None,None]*regionManager.Vt[None,None,:,:]
  K = np.shape(regionManager.V)[1]
  N = np.shape(regionManager.V)[0]
  a_st = np.zeros((N,nwindows) )  
  #a0_pod = globalDot(regionManager.V.transpose(),regionManager.a.flatten(),regionManager) # a0_pod \in R^K
  ## rezise into a set of spacetime coordinates \in R^K \times R^Kt
  a0_pod_st = np.zeros((N,nwindows))
  for i in range(1,nwindows):
    a_st[:,i] = regionManager.a[:] 

  
  ## now put into temporal coords
  #print(np.shape(regionManager.Vt),np.shape(a0_pod_st))
  #a0_pod_st = np.einsum('it,ki->kt', regionManager.Vt, a0_pod_st) 
  a0_pod_st = np.einsum('ijkl,ik->jl',VST,a_st) 
  a_tmp = np.einsum('ij,jl->il',regionManager.V,a0_pod_st)
  a_spacetime = np.einsum('kl,il->ik',regionManager.Vt,a_tmp)
  ast2 = np.einsum('ijkl,jl->ik',VST,a0_pod_st)
  #print(np.linalg.norm(a_spacetime - a_st)  )
  print(np.linalg.norm(ast2 - a_st)  ,np.linalg.norm(ast2),np.linalg.norm(a_st) )

  #
  #K = np.size(a0_pod)
  #A = create_mv(regionManager.a.flatten())
  #res_1 = least_squares(unsteadyResidual, a0_pod,ftol=1e-4,xtol=1e-8,jac=create_mv,method='dogbox',verbose=2)
  res_1 = least_squares(unsteadyResidual, a0_pod_st.flatten(),ftol=1e-8,xtol=1e-8,method='trf',verbose=2)
  a_sol_pod = np.reshape( res_1.x[:] ,(K,Kt) )[:,:]
  regionManager.a[:] = np.einsum('ijkl,jl->ik',VST,a_sol_pod)[:,-1]
  regionManager.t += nwindows*regionManager.dt
  regionManager.iteration += nwindows
  regionManager.window_counter += 1




## function to implement LSPG for backward euler timescheme
## uses scipys least_squares functionality
def crankNicolson_LSPG_windowed_Rt(regionManager,eqns,args):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  regionManager.a0[:] = regionManager.a[:]
  regionManager.getRHS_REGION_OUTER(regionManager,eqns)
  R0 = np.zeros(np.size(regionManager.RHS))
  R0[:] = regionManager.RHS[:]
  nwindows = regionManager.tau 
  def unsteadyResidual(v):
    K = np.shape(regionManager.V)[1]
    vw = np.reshape(v,(K,nwindows),order='F')
    ### DO FIRST LSPG WINDOW    
    an  = np.dot(regionManager.V,v[0:K]) #set current state to be the solution iterate of the Newton Krylov solver
    regionManager.a[:] = an[:]
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions
    ## Construct the unsteady residual for Backward Euler in a few steps
    RHS_BE = np.zeros(np.size(regionManager.RHS))
    R1 = np.zeros(np.size(regionManager.RHS))
    R1[:] = regionManager.RHS[:]
    #Rstar = ( regionManager.a[:]*1. - regionManager.a0[:] ) - RHS_BE
    R0m = R0*1.
    Rstar = ( regionManager.a[:] - regionManager.a0[:] ) - 0.5*regionManager.dt*(R1 + R0m)
    Rcombined = Rstar.flatten()*1.
    ## now compute the same residual over all windows
    for window in range(1,nwindows):
      anp1 = np.dot(regionManager.V,vw[:,window])
      #anp1 = np.dot(regionManager.V,v[K::])
      regionManager.a[:] = anp1[:]
      regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions
      RHS_BE2 = np.zeros(np.size(regionManager.RHS))
      R0m[:] = R1*1.
      R1[:] = regionManager.RHS[:]
      #Rstar[:] = ( regionManager.a[:]*1. - np.dot(regionManager.V,vw[:,window-1] ) ) - 0.5*regionManager.dt(R1[:] + R0m[:])
      #print(np.shape(v),np.shape(vw), np.shape( np.dot(regionManager.V,v[K*window:(K+1)*window] ) ) )
      Rstar[:] = ( regionManager.a[:]*1. - np.dot(regionManager.V,vw[:,window-1] ) ) - 0.5*regionManager.dt*(R1[:] + R0m[:])

      #Rstar[:] = R1[:]
      Rcombined = np.append(Rcombined.flatten(),Rstar[:].flatten()) 
    return Rcombined

  def create_mv(v):
    def MF_Jacobian(v):
      an = regionManager.a0*1
      N,K = np.shape(regionManager.V)[0],np.shape(regionManager.V)[1]
      V2 = np.zeros((nwindows*N,nwindows*K))
      for i in range(0,nwindows):
#        print(i*N,(i+1)*N,i*K,(i+1)*K,np.shape(V2),np.shape(V2[:,i*K:(i+1)*K]))
        V2[i*N:(i+1)*N,i*K:(i+1)*K] = regionManager.V[:,:]
      #V2[0:N,K] = 
      #vr = np.reshape( np.dot(main.V,v) , np.shape(main.a.a) )
      ##vr = np.reshape(v,np.shape(main.a.a))
      #eps = 5.e-7
      #main.a.a[:] = an + eps*vr
      #main.getRHS(main,MZ,eqns)
      #RHS_BE = np.zeros(np.shape(main.RHS))
      #R1 = np.zeros(np.shape(main.RHS))
      #R1[:] = main.RHS[:]
      #RHS_BE[:] = main.dt*(R1 - Rn)/eps
      #main.basis.applyMassMatrix(main,RHS_BE)
      #Av = vr - RHS_BE
      return np.dot(V2,v)
      #return Av.flatten()

    def rmatvec(v):
      an = regionManager.a0*1
      N,K = np.shape(regionManager.V)[0],np.shape(regionManager.V)[1]
      V2 = np.zeros((nwindows*N,nwindows*K))
      for i in range(0,nwindows):
        V2[i*N:(i+1)*N,i*K:(i+1)*K] = regionManager.V[:]
      return globalDot(V2.transpose(),v,regionManager)

    mdim = np.size(regionManager.a)*nwindows
    ndim = np.shape(regionManager.V)[1]*nwindows
    return  LinearOperator((mdim,ndim),matvec=MF_Jacobian,rmatvec=rmatvec)


  a0_pod = globalDot(regionManager.V.transpose(),regionManager.a.flatten(),regionManager)
  K = np.size(a0_pod)
  for i in range(1,nwindows):
    a0_pod = np.append(a0_pod,a0_pod[0:K])
  #A = create_mv(regionManager.a.flatten())
  #res_1 = least_squares(unsteadyResidual, a0_pod,ftol=1e-4,xtol=1e-8,jac=create_mv,method='dogbox',verbose=2)
  res_1 = least_squares(unsteadyResidual, a0_pod,ftol=1e-8,xtol=1e-8,method='dogbox',verbose=2)

  a_sol_pod = res_1.x[-K::]
  regionManager.a[:] = np.dot(regionManager.V,a_sol_pod)
  regionManager.t += nwindows*regionManager.dt
  regionManager.iteration += nwindows








## function to implement LSPG for backward euler timescheme
## uses scipys least_squares functionality
def backwardEuler_LSPG_windowed(regionManager,eqns,args):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  regionManager.a0[:] = regionManager.a[:]
  regionManager.getRHS_REGION_OUTER(regionManager,eqns)
  R0 = np.zeros(np.size(regionManager.RHS))
  R0[:] = regionManager.RHS[:]
  nwindows = regionManager.tau 
  def unsteadyResidual(v):
    K = np.shape(regionManager.V)[1]
    vw = np.reshape(v,(K,nwindows),order='F')
    ### DO FIRST LSPG WINDOW    
    an  = np.dot(regionManager.V,v[0:K]) #set current state to be the solution iterate of the Newton Krylov solver
    regionManager.a[:] = an[:]
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions
    ## Construct the unsteady residual for Backward Euler in a few steps
    RHS_BE = np.zeros(np.size(regionManager.RHS))
    R1 = np.zeros(np.size(regionManager.RHS))
    R1[:] = regionManager.RHS[:]
    RHS_BE[:] = regionManager.dt*R1
    Rstar = ( regionManager.a[:]*1. - regionManager.a0[:] ) - RHS_BE
    Rcombined = Rstar.flatten()*1.
    ## now compute the same residual over all windows
    for window in range(1,nwindows):
      anp1 = np.dot(regionManager.V,vw[:,window])
      #anp1 = np.dot(regionManager.V,v[K::])
      regionManager.a[:] = anp1[:]
      regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions
      RHS_BE2 = np.zeros(np.size(regionManager.RHS))
      R1_2 = np.zeros(np.size(regionManager.RHS))
      R1_2[:] = regionManager.RHS[:]
      RHS_BE2[:] = regionManager.dt*R1_2
      Rstar2 = ( regionManager.a[:]*1. - np.dot(regionManager.V,vw[:,window-1] ) ) - RHS_BE2
      #Rstar2 = ( regionManager.a[:]*1. - an ) - RHS_BE2
      Rcombined = np.append(Rcombined.flatten(),Rstar2.flatten()) 
    return Rcombined

  def create_mv(v):
    def MF_Jacobian(v):
      an = regionManager.a0*1
      N,K = np.shape(regionManager.V)[0],np.shape(regionManager.V)[1]
      V2 = np.zeros((nwindows*N,nwindows*K))
      for i in range(0,nwindows):
#        print(i*N,(i+1)*N,i*K,(i+1)*K,np.shape(V2),np.shape(V2[:,i*K:(i+1)*K]))
        V2[i*N:(i+1)*N,i*K:(i+1)*K] = regionManager.V[:,:]
      #V2[0:N,K] = 
      #vr = np.reshape( np.dot(main.V,v) , np.shape(main.a.a) )
      ##vr = np.reshape(v,np.shape(main.a.a))
      #eps = 5.e-7
      #main.a.a[:] = an + eps*vr
      #main.getRHS(main,MZ,eqns)
      #RHS_BE = np.zeros(np.shape(main.RHS))
      #R1 = np.zeros(np.shape(main.RHS))
      #R1[:] = main.RHS[:]
      #RHS_BE[:] = main.dt*(R1 - Rn)/eps
      #main.basis.applyMassMatrix(main,RHS_BE)
      #Av = vr - RHS_BE
      return np.dot(V2,v)
      #return Av.flatten()

    def rmatvec(v):
      an = regionManager.a0*1
      N,K = np.shape(regionManager.V)[0],np.shape(regionManager.V)[1]
      V2 = np.zeros((nwindows*N,nwindows*K))
      for i in range(0,nwindows):
        V2[i*N:(i+1)*N,i*K:(i+1)*K] = regionManager.V[:]
      return globalDot(V2.transpose(),v,regionManager)

    mdim = np.size(regionManager.a)*nwindows
    ndim = np.shape(regionManager.V)[1]*nwindows
    return  LinearOperator((mdim,ndim),matvec=MF_Jacobian,rmatvec=rmatvec)


  a0_pod = globalDot(regionManager.V.transpose(),regionManager.a.flatten(),regionManager)
  K = np.size(a0_pod)
  for i in range(1,nwindows):
    a0_pod = np.append(a0_pod,a0_pod[0:K])
  #A = create_mv(regionManager.a.flatten())
  #res_1 = least_squares(unsteadyResidual, a0_pod,ftol=1e-4,xtol=1e-8,jac=create_mv,method='dogbox',verbose=2)
  res_1 = least_squares(unsteadyResidual, a0_pod,ftol=1e-8,xtol=1e-8,method='trf',verbose=2)

  a_sol_pod = res_1.x[-K::]
  regionManager.a[:] = np.dot(regionManager.V,a_sol_pod)
  regionManager.t += nwindows*regionManager.dt
  regionManager.iteration += nwindows


## function to implement POD for backward Euler timescheme
## uses scipys GMRES functionality
def backwardEuler_POD(regionManager,eqns,args):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  regionManager.a0[:] = regionManager.a[:]
  def unsteadyResidual(v):
    regionManager.a[:] = np.dot(regionManager.V,v[:])# np.dot(regionManager.V[rec_stencil_list],v[:]) #set current state to be the solution iterate of the Newton Krylov solver
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling QDEIM
    ## Construct the unsteady residual for Crank Nicolson in a few steps
    R1 = np.zeros(np.size(regionManager.RHS[:]))
    R1[:] = regionManager.RHS[:]
    ## compute the residual at the sample points
    Rstar = np.dot(regionManager.V.transpose(), ( regionManager.a[:]*1. - regionManager.a0[:] ) - regionManager.dt*R1 )
    return Rstar.flatten()


  a0_pod = globalDot(regionManager.V.transpose(),regionManager.a.flatten(),regionManager)
  #A = create_mv(regionManager.a.flatten())
  #res_1 = least_squares(unsteadyResidual, a0_pod,ftol=1e-4,xtol=1e-8,jac=create_mv,method='dogbox',verbose=0)
  #res_1 = least_squares(unsteadyResidual, a0_pod,ftol=1e-8,xtol=1e-8,method='dogbox',verbose=2)
  res = scipy.optimize.newton_krylov(unsteadyResidual,a0_pod,verbose=0)
  a_sol_pod = res[:]
  regionManager.a_pod[:] = res[:] 
  regionManager.a[:] = np.dot(regionManager.V,a_sol_pod)
  regionManager.t += regionManager.dt
  regionManager.iteration += 1



## uses scipys GMRES functionality
def crankNicolson_POD(regionManager,eqns,args):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  regionManager.a0[:] = regionManager.a[:]
  regionManager.getRHS_REGION_OUTER(regionManager,eqns)
  R0 = np.zeros(np.size(regionManager.RHS))
  R0[:] = regionManager.RHS[:]
  def unsteadyResidual(v):
    regionManager.a[:] = np.dot(regionManager.V,v[:])# np.dot(regionManager.V[rec_stencil_list],v[:]) #set current state to be the solution iterate of the Newton Krylov solver
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling QDEIM
    ## Construct the unsteady residual for Crank Nicolson in a few steps
    R1 = np.zeros(np.size(regionManager.RHS[:]))
    R1[:] = regionManager.RHS[:]
    ## compute the residual at the sample points
    Rstar = np.dot(regionManager.V.transpose(), ( regionManager.a[:]*1. - regionManager.a0[:] ) - 0.5*regionManager.dt*(R1 + R0) )
    return Rstar.flatten()


  a0_pod = globalDot(regionManager.V.transpose(),regionManager.a.flatten(),regionManager)
  #A = create_mv(regionManager.a.flatten())
  #res_1 = least_squares(unsteadyResidual, a0_pod,ftol=1e-4,xtol=1e-8,jac=create_mv,method='dogbox',verbose=0)
  #res_1 = least_squares(unsteadyResidual, a0_pod,ftol=1e-8,xtol=1e-8,method='dogbox',verbose=2)
  res = scipy.optimize.newton_krylov(unsteadyResidual,a0_pod,verbose=4)
  a_sol_pod = res[:]
  regionManager.a_pod[:] = res[:] 
  regionManager.a[:] = np.dot(regionManager.V,a_sol_pod)
  regionManager.t += regionManager.dt
  regionManager.iteration += 1



## function to implement LSPG for crank nicolson timescheme
## uses scipys least_squares functionality
def crankNicolson_LSPG(regionManager,eqns,args):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  regionManager.a0[:] = regionManager.a[:]
  regionManager.getRHS_REGION_OUTER(regionManager,eqns)
  R0 = np.zeros(np.size(regionManager.RHS))
  R0[:] = regionManager.RHS[:]
  def unsteadyResidual(v):
    regionManager.a[:] = np.dot(regionManager.V,v[:])# np.dot(regionManager.V[rec_stencil_list],v[:]) #set current state to be the solution iterate of the Newton Krylov solver
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling QDEIM
    ## Construct the unsteady residual for Crank Nicolson in a few steps
    R1 = np.zeros(np.size(regionManager.RHS[:]))
    R1[:] = regionManager.RHS[:]
    ## compute the residual at the sample points
    Rstar = ( regionManager.a[:]*1. - regionManager.a0[:] ) - 0.5*regionManager.dt*(R1 + R0)

    region_counter = 0
    for j in regionManager.mpi_regions_owned:
      main = regionManager.region[region_counter]
      region_counter += 1#
      p = (1.4 - 1.)*(main.a.u[4] - 0.5*main.a.u[1]**2/main.a.u[0] - 0.5*main.a.u[2]**2/main.a.u[0] - 0.5*main.a.u[3]**2/main.a.u[0])
      s  = ( ( np.log(p) - 1.4*np.log(main.a.u[0]) )*main.a.u[0]/(1.4 - 1.)*-1.)


    return np.linalg.norm( Rstar.flatten() )**2 - 0.001*np.linalg.norm(s)**2


  def create_mv(v):
    def MF_Jacobian(v):
      an = regionManager.a0*1.
      #vr = np.reshape( np.dot(main.V,v) , np.shape(main.a.a) )
      ##vr = np.reshape(v,np.shape(main.a.a))
      #eps = 5.e-7
      #main.a.a[:] = an + eps*vr
      #main.getRHS(main,MZ,eqns)
      #RHS_BE = np.zeros(np.shape(main.RHS))
      #R1 = np.zeros(np.shape(main.RHS))
      #R1[:] = main.RHS[:]
      #RHS_BE[:] = main.dt*(R1 - Rn)/eps
      #main.basis.applyMassMatrix(main,RHS_BE)
      #Av = vr - RHS_BE
      return np.dot(regionManager.V,v)
      #return Av.flatten()

    def rmatvec(v):
      return globalDot(regionManager.V.transpose(),v,regionManager)

    mdim = np.size(regionManager.a)
    ndim = np.shape(regionManager.V)[1]
    return  LinearOperator((mdim,ndim),matvec=MF_Jacobian,rmatvec=rmatvec)


  a0_pod = globalDot(regionManager.V.transpose(),regionManager.a.flatten(),regionManager)
  A = create_mv(regionManager.a.flatten())
  #res_1 = least_squares(unsteadyResidual, a0_pod,ftol=1e-4,xtol=1e-8,jac=create_mv,method='dogbox',verbose=0)
  #res_1 = least_squares(unsteadyResidual, a0_pod,ftol=1e-8,xtol=1e-8,method='dogbox',verbose=2)
  res_1 = scipy.optimize.least_squares(unsteadyResidual, a0_pod)

  a_sol_pod = res_1.x
  regionManager.a[:] = np.dot(regionManager.V,a_sol_pod)
  regionManager.t += regionManager.dt
  regionManager.iteration += 1


## function to implement LSPG for CN timescheme with custom least squares solver
## uses scipys least_squares functionality
def crankNicolson_LSPG2(regionManager,eqns,args):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  regionManager.a0[:] = regionManager.a[:]
  regionManager.getRHS_REGION_OUTER(regionManager,eqns)
  R0 = np.zeros(np.size(regionManager.RHS))
  R0[:] = regionManager.RHS[:]
  def unsteadyResidual(v):
    regionManager.a[:] = v[:]# np.dot(regionManager.V[rec_stencil_list],v[:]) #set current state to be the solution iterate of the Newton Krylov solver
    regionManager.getRHS_REGION_OUTER(regionManager,eqns) # evaluate the RHS includes loop over all regions - should be calling QDEIM
    ## Construct the unsteady residual for Crank Nicolson in a few steps
    R1 = np.zeros(np.size(regionManager.RHS[:]))
    R1[:] = regionManager.RHS[:]
    ## compute the residual at the sample points
    Rstar = ( regionManager.a[:]*1. - regionManager.a0[:] ) - 0.5*regionManager.dt*(R1 + R0)
    return Rstar.flatten()

  nx,nbasis = np.shape(regionManager.V)
  JV = np.zeros((nx,nbasis) )
  regionManager.an = regionManager.a*1.
  # compute residual
  r = unsteadyResidual(regionManager.a)
  ls_iteration = 0
  alpha = 1.
  rbar = r
  da_norm = 1.
  r_norm = globalNorm(rbar,regionManager)
  if (regionManager.mpi_rank == 0):
    print('Initial Residual = ' + str(r_norm) )

  while ( r_norm > 1e-8 and da_norm >= 1e-8):
   if (ls_iteration >= 20 or alpha <= 0.01):
     print('Didnt converge, alpha = ' + str(alpha) ,'  iteration count = ' + str(ls_iteration) )
     da_norm = 0.
   else:
    r_norm_old = r_norm*1.
    rbar = r 
    eps = 1e-5
    an = regionManager.a[:]*1.
    if (ls_iteration%2 == 0 and regionManager.iteration%1 == 0):
     for i in range(0,nbasis):
      regionManager.a[:] = an[:] + eps*regionManager.V[:,i]
      Jv_column = unsteadyResidual(regionManager.a[:])
      JV[:,i] = (Jv_column - r[:]) / eps
     regionManager.JV = JV
    JV = regionManager.JV 
    ## now compute Jbar and rbar
    JVbar = JV
    Q,R = np.linalg.qr(JVbar)
    da_pod = np.linalg.solve(R,-np.dot(Q.transpose(),rbar) )
    da_norm = np.linalg.norm(da_pod)
    #print(np.linalg.norm(da_pod))
    regionManager.a[:] = regionManager.a[:] + np.dot(regionManager.V,da_pod[:])
    r = unsteadyResidual(regionManager.a)
    r_norm = globalNorm(r,regionManager)
    if (regionManager.mpi_rank == 0):
      print('Residual = ' + str(r_norm), ' Change to solution is ' + str(da_norm))
  regionManager.t += regionManager.dt
  regionManager.iteration += 1

