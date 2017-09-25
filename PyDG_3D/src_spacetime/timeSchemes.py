import numpy as np
#import sys, petsc4py
#petsc4py.init(sys.argv)
#from petsc4py import PETSc
from DG_functions import * 
import scipy.linalg
from linear_solvers import *
import matplotlib.pyplot as plt
from mpi4py import MPI
import sys
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
from pylab import *
from tensor_products import diffCoeffs
from jacobian_schemes import *
def gatherResid(Rstar,main):
  ## Create Global residual
  data = main.comm.gather(np.linalg.norm(Rstar)**2,root = 0)
  if (main.mpi_rank == 0):
    Rstar_glob = 0.
    for j in range(0,main.num_processes):
      Rstar_glob += data[j]
    Rstar_glob = np.sqrt(Rstar_glob)
    for j in range(1,main.num_processes):
      main.comm.send(Rstar_glob, dest=j)
  else:
    Rstar_glob = main.comm.recv(source=0)
  return Rstar_glob

def spaceTimeIncomp(main,MZ,eqns,args=None):
  main.a.Upx,main.a.Upy,main.a.Upz = main.basis.diffU(main.a.a,main)
  div = main.a.Upx[0] + main.a.Upy[1] + main.a.Upz[2]
  print('Div = ' + str(np.linalg.norm(div)))
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  main.a0[:] = main.a.a[:]
  ord_arr0= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arr1= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arr2= np.linspace(0,main.order[2]-1,main.order[2])
  ord_arr3= np.linspace(0,main.order[3]-1,main.order[3])
  scale =  (2.*ord_arr0[:,None,None,None] + 1.)*(2.*ord_arr1[None,:,None,None] + 1.)*(2.*ord_arr2[None,None,:,None]+1.)*(2.*ord_arr3[None,None,None,:] + 1. )/16.

  def unsteadyResidual(v):
    ord_arr0= np.linspace(0,main.order[0]-1,main.order[0])
    ord_arr1= np.linspace(0,main.order[1]-1,main.order[1])
    ord_arr2= np.linspace(0,main.order[2]-1,main.order[2])
    ord_arr3= np.linspace(0,main.order[3]-1,main.order[3])
    scale =  (2.*ord_arr0[:,None,None,None] + 1.)*(2.*ord_arr1[None,:,None,None] + 1.)*(2.*ord_arr2[None,None,:,None]+1.)*(2.*ord_arr3[None,None,None,:] + 1. )/16.
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    eqns.getRHS(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    ## now integrate the volume term ( \int ( dw/dt * u) )
    volint_t = main.basis.volIntegrateGlob(main,main.a.u,main.w0,main.w1,main.w2,main.wp3)*scale[None,:,:,:,:,None,None,None,None]*2./main.dt
    uFuture,uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)
    futureFlux = main.basis.faceIntegrateGlob(main,uFuture,main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    uPast[:,:,:,:,:,:,:,0] = main.a.uFuture[:,:,:,:,:,:,:,-1]
    if (main.Npt > 1):
      uPast[:,:,:,:,:,:,:,1::] = uFuture[:,:,:,:,:,:,:,0:-1]
    pastFlux   = main.basis.faceIntegrateGlob(main,uPast  ,main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    volint_t[-1] = 0.
    futureFlux[-1] = 0.
    pastFlux[-1] = 0.
    Rstar = volint_t - (futureFlux[:,:,:,:,None] - pastFlux[:,:,:,:,None]*main.altarray3[None,None,None,None,:,None,None,None,None])*scale[None,:,:,:,:,None,None,None,None]*2./main.dt + main.RHS[:]
    Rstar[-1] = main.RHS[-1]*1000.
    Rstar *= main.dt
    print(np.linalg.norm(Rstar[0]),np.linalg.norm(Rstar[1]),np.linalg.norm(Rstar[2]),np.linalg.norm(Rstar[3]))
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  def create_MF_Jacobian(v,args,main):
    ord_arr0= np.linspace(0,main.order[0]-1,main.order[0])
    ord_arr1= np.linspace(0,main.order[1]-1,main.order[1])
    ord_arr2= np.linspace(0,main.order[2]-1,main.order[2])
    ord_arr3= np.linspace(0,main.order[3]-1,main.order[3])
    scale =  (2.*ord_arr0[:,None,None,None] + 1.)*(2.*ord_arr1[None,:,None,None] + 1.)*(2.*ord_arr2[None,None,:,None]+1.)*(2.*ord_arr3[None,None,None,:] + 1. )/16.
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-2
    main.a.a[:] = an + eps*vr
    eqns.getRHS(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    vr_phys = main.basis.reconstructUGeneral(main,vr)
    volint_t = main.basis.volIntegrateGlob(main,vr_phys,main.w0,main.w1,main.w2,main.wp3)*scale[None,:,:,:,:,None,None,None,None]*2./main.dt
    uFuture,uPast = main.basis.reconstructEdgesGeneralTime(vr,main)
    futureFlux = main.basis.faceIntegrateGlob(main,uFuture,main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    uPast[:,:,:,:,:,:,:,0] = 0.#main.a.uFuture[:,:,:,:,:,:,:,-1]
    if (main.Npt > 1):
      uPast[:,:,:,:,:,:,:,1::] = uFuture[:,:,:,:,:,:,:,0:-1]
    pastFlux   = main.basis.faceIntegrateGlob(main,uPast  ,main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    volint_t[-1] = 0.
    futureFlux[-1] = 0.
    pastFlux[-1] = 0.
    Av = volint_t - (futureFlux[:,:,:,:,None] - pastFlux[:,:,:,:,None]*main.altarray3[None,None,None,None,:,None,None,None,None])*scale[None,:,:,:,:,None,None,None,None]*2./main.dt + \
         1./eps * (R1 - Rn) 
    Av[-1] = 1./eps * (R1[-1] - Rn[-1])*1000
    return Av.flatten()*main.dt

  nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns)
  main.t += main.dt*main.Npt
  main.iteration += 1
  main.a.uFuture[:],uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)



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
    Rstar = volint_t - (futureFlux[:,:,:,:,None])*2./main.dt + main.RHS[:]
    main.basis.applyMassMatrix(main,Rstar)
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob


  def create_MF_Jacobian_element(v,args,main):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-5
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

  def create_Dinv2(f,main,args=None):
    ff = np.reshape(f*1.,np.shape(main.a.a))
    main.basis.applyMassMatrix(main,ff)
    return ff.flatten()


  def create_Dinv2(fs,main,args=None):
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

  def create_Dinv(f,main,args):
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
    ferror = GMRes_element(create_MF_Jacobian_element2, f0r2, -np.reshape(a0,np.shape(f0r2))*0.,main,MF_Jacobian_args2,None,None,tol,1e-10,200,False)
    ff = np.reshape(ferror,np.shape(main.a.a))
    return ff.flatten()

  def unsteadyResidual(v):
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
    eps = 5.e-2
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

def spaceTimeExperimental(main,MZ,eqns,args=None):
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
    Rstar = volint_t - (futureFlux[:,:,:,:,None])*2./main.dt + main.RHS[:]
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  def unsteadyResidual_element_zeta(main,v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    getRHS_element_zeta(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    ## now integrate the volume term ( \int ( dw/dt * u) )
    Rstar = main.RHS[:]*1.
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  def unsteadyResidual_element_eta(main,v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    getRHS_element_eta(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    ## now integrate the volume term ( \int ( dw/dt * u) )
    Rstar = main.RHS[:]*1.
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  def unsteadyResidual_element_mu(main,v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    getRHS_element_mu(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    ## now integrate the volume term ( \int ( dw/dt * u) )
    Rstar = main.RHS[:]*1.
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob



  def unsteadyResidual_element_time(main,v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    ## now integrate the volume term ( \int ( dw/dt * u) )
    main.basis.reconstructU(main,main.a)
    volint_t = main.basis.volIntegrateGlob(main,main.a.u[:]*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.wp3)*2./main.dt
    uFuture,uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)
    futureFlux = main.basis.faceIntegrateGlob(main,uFuture*main.Jdet[None,:,:,:,:,:,:,None],main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    Rstar = volint_t - (futureFlux[:,:,:,:,None])*2./main.dt 
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,Rstar*1.,Rstar_glob


  def create_MF_Jacobian_element_zeta(v,args,main):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-1
#    main.a.a[:] = an + eps*vr
#    getRHS_element_zeta(main,main,eqns)
#    R1 = np.zeros(np.shape(main.RHS))
#    R1[:] = main.RHS[:]
#    Av = 1./eps * (R1 - Rn)
#    main.a.a[:] = an[:]
    main.a.a[:] = vr[:]
    getRHS_element_zeta(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Av = R1*1.
    return Av#.flatten()

  def create_MF_Jacobian_element_eta(v,args,main):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-1
    main.a.a[:] = vr[:]
    getRHS_element_eta(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Av = R1*1.
    return Av#.flatten()


  def create_MF_Jacobian_element_time(v,args,main):
    vr = np.reshape(v,np.shape(main.a.a))
    vr_phys = main.basis.reconstructUGeneral(main,vr)
    volint_t = main.basis.volIntegrateGlob(main,vr_phys*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.wp3)*2./main.dt
    uFuture,uPast = main.basis.reconstructEdgesGeneralTime(vr,main)
    futureFlux = main.basis.faceIntegrateGlob(main,uFuture*main.Jdet[None,:,:,:,:,:,:,None],main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    Av =  volint_t - (futureFlux[:,:,:,:,None])*2./main.dt 
    return Av


  def create_MF_Jacobian_element(v,args,main):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-3
#    main.a.a[:] = an[:]
#    getRHS_element(main,main,eqns)
#    Rn = np.zeros(np.shape(main.RHS))
#    Rn[:] = main.RHS[:]
#    print('MF_resid',np.linalg.norm(Rn),np.linalg.norm(Rn2))

    main.a.a[:] = an + eps*vr
    getRHS_element(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    vr_phys = main.basis.reconstructUGeneral(main,vr)
    volint_t = main.basis.volIntegrateGlob(main,vr_phys*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.wp3)*2./main.dt
    uFuture,uPast = main.basis.reconstructEdgesGeneralTime(vr,main)
    futureFlux = main.basis.faceIntegrateGlob(main,uFuture*main.Jdet[None,:,:,:,:,:,:,None],main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    Av = volint_t - (futureFlux[:,:,:,:,None])*2./main.dt + 1./eps * (R1 - Rn)
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
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob


  def create_MF_Jacobian_element2(v,args,main):
    an = args[0]
    Rn = args[1]
    #print(np.linalg.norm(JinvX))
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 1e-7

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
    Av = np.reshape(Av, (main.nvars*main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
#    main.a.a[:] = an[:]

    return Av




#  JXT02 = computeJacobianXT(main,eqns,unsteadyResidual_element)
#  JXT02 = np.reshape(JXT02,(main.nvars,main.order[0],main.order[3],main.order[0]*main.order[3],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
#  JXT02 = np.reshape(JXT02,(main.nvars,main.order[0],main.order[3],main.order[0],main.order[3],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
#  t0 = time.time()
#  J = computeBlockJacobian(main,unsteadyResidual_element) #get the Jacobian
#  J2 = np.zeros(np.shape(J))
#  J2[:] = J[:]
#  print('Jac time = ' , time.time() - t0)

#  JX = computeJacobianX(main,unsteadyResidual_element_zeta) #get the Jacobian
#  JY = computeJacobianY(main,unsteadyResidual_element_eta) #get the Jacobian
#  JT = computeJacobianT(main,unsteadyResidual_element_time) #get the Jacobian


  def create_Dinv_splitting(f,main,args=None):
    ImatX = np.eye(main.nvars*main.order[0])
    ImatY = np.eye(main.nvars*main.order[1])
    ImatZ = np.eye(main.nvars*main.order[2])
    ImatT = np.eye(main.nvars*main.order[3])
    rho = -1000.

    JX = computeJacobianX(main,unsteadyResidual_element_zeta) #get the Jacobian
    JX = np.reshape(JX, (main.nvars*main.order[0],main.nvars*main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    JX = np.rollaxis(np.rollaxis(JX ,1,9),0,8)

    JT = computeJacobianT(main,unsteadyResidual_element_time) #get the Jacobian
    JT = np.reshape(JT, (main.nvars*main.order[3],main.nvars*main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
    JT = np.rollaxis(np.rollaxis(JT ,1,9),0,8)

    f = np.reshape(f, (main.nvars*main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    f = np.rollaxis(f,0,8)
    f[:] = np.linalg.solve(JX ,f)
    f = np.rollaxis(f,7,0)
    f = np.reshape(f, (main.nvars,main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    f = np.rollaxis(f,1,3)


    f = np.rollaxis(f,4,1)
    f = np.reshape(f, (main.nvars*main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
    f = np.rollaxis(f,0,8)
    f[:] = np.linalg.solve(JT , f)
    f = np.rollaxis(f,7,0)
    f = np.reshape(f, (main.nvars,main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
    f = np.rollaxis(f,1,5)



    return f.flatten()



  def create_Dinv2(f,main,args=None):
    args[2] += 1
    if (args[2]%10 == 0):
      J2[:] = computeBlockJacobian(main,unsteadyResidual_element) #get the Jacobian
    J = np.reshape(J2[0:main.nvars,0:main.order[0],0:main.order[1],0:main.order[2],0:main.order[3],0:main.nvars,0:main.order[0],0:main.order[1],0:main.order[2],0:main.order[3] ,0:main.Npx,0:main.Npy,0:main.Npz,0:main.Npt], (main.nvars*main.order[0]*main.order[1]*main.order[2]*main.order[3],main.nvars*main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    J = np.rollaxis(np.rollaxis(J ,1,6),0,5)
    f = np.reshape(f,np.shape(main.a.a))
    f = np.reshape(f, (main.nvars*main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    f = np.rollaxis(f,0,5)
    f = np.linalg.solve(J,f)
    f = np.rollaxis(f,4,0)
    f = np.reshape(f,np.shape(main.a.a) )
    return f.flatten()


  def create_Dinv(f,main,args):
    tol = args[1]
    a0 = np.zeros(np.shape(main.a.a))
    a0[:] = main.a.a[:]
    f0 = np.zeros(np.size(f))
    f0[:] = f[:]
    f = np.reshape(f,np.shape(main.a.a))
    f0r = np.reshape(f0,np.shape(main.a.a))
    if (tol >= 1e-9):
      Rstarn_pc,Rn_pc,Rstar_glob_pc = unsteadyResidual_element(main,f)
      MF_Jacobian_args2 = [f,Rn_pc]
      f0r2 = np.reshape(f0r, (main.nvars*main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
      unsteadyResiduals = [unsteadyResidual_element,unsteadyResidual_element_zeta,unsteadyResidual_element_eta,unsteadyResidual_element_mu,unsteadyResidual_element_time]
      jacobians = None#[JX,JY,JT]
      #Rstarn_pc = np.reshape(Rstarn_pc,np.shape(f0r2))
      ferror = GMRes_element(create_MF_Jacobian_element2, f0r2, -np.reshape(a0,np.shape(f0r2))*0.,main,MF_Jacobian_args2,None,None,tol,1,50,False)
      #ferror = operatorSplitting(create_MF_Jacobian_element,f0r.flatten(),f0.flatten()*0.,main,MF_Jacobian_args2,unsteadyResiduals,jacobians,tol,maxiter_outer=1,maxiter=20,printnorm=0)
      #ferror = operatorSplitting(create_MF_Jacobian_element,-Rstarn_pc.flatten(),f0.flatten()*0.,main,MF_Jacobian_args2,unsteadyResiduals,jacobians,tol,maxiter_outer=1,maxiter=1,printnorm=0)
      ff = np.reshape(ferror,np.shape(main.a.a))
    else:
      ff = f
    return ff.flatten()

  def unsteadyResidual(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
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
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  def create_MF_Jacobian(v,args,main):
    an = args[0]
    Rn = args[1]
    #print(np.linalg.norm(JinvX))
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-2
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
    return Av.flatten()

  unsteadyResiduals = [unsteadyResidual,unsteadyResidual_element_zeta,unsteadyResidual_element_eta,unsteadyResidual_element_mu,unsteadyResidual_element_time]
  MF_Jacobians = [create_MF_Jacobian,create_MF_Jacobian_element_zeta,create_MF_Jacobian_element_eta,create_MF_Jacobian_element_time]
  #ADISolver(unsteadyResiduals,MF_Jacobians,main,linear_solver,sparse_quadrature,eqns,PC=None)
  #nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns,unsteadyResiduals)
  nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns,create_Dinv)

  main.t += main.dt*main.Npt
  main.iteration += 1
  main.a.uFuture[:],uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)





#######################################################

def spaceTimeSplitting(main,MZ,eqns,args=None):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  main.a0[:] = main.a.a[:]

  def unsteadyResidual_element(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    getRHS_element(main,main,eqns)
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
#    Rstar = main.a.a[:] + 2.*Rstar
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  def unsteadyResidual_element_X(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    getRHS_element_zeta(main,main,eqns)
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
#    Rstar = main.a.a[:] + 2.*Rstar
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  def unsteadyResidual_element_Y(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    getRHS_element_eta(main,main,eqns)
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
 #   Rstar = main.a.a[:] + 2.*Rstar
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  
  JinvX = computeJacobianX(main,eqns,unsteadyResidual_element_X)
  JinvY = computeJacobianY(main,eqns,unsteadyResidual_element_Y)

  def unsteadyResidualX(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    #JinvX[:] = computeJacobianX(main,eqns,unsteadyResidual_element_X)

    getRHS_zeta(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    Rstar = np.zeros(np.shape(main.RHS))

    R1[:] = main.RHS[:]
    ## now integrate the volume term ( \int ( dw/dt * u) )
    volint_t = main.basis.volIntegrateGlob(main,main.a.u*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.wp3)*2./main.dt
    uFuture,uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)
    futureFlux = main.basis.faceIntegrateGlob(main,uFuture*main.Jdet[None,:,:,:,:,:,:,None],main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    uPast[:,:,:,:,:,:,:,0] = main.a.uFuture[:,:,:,:,:,:,:,-1]
    if (main.Npt > 1):
      uPast[:,:,:,:,:,:,:,1::] = uFuture[:,:,:,:,:,:,:,0:-1]
    pastFlux   = main.basis.faceIntegrateGlob(main,uPast*main.Jdet[None,:,:,:,:,:,:,None] ,main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    Rstar[:] = (volint_t - (futureFlux[:,:,:,:,None] - pastFlux[:,:,:,:,None]*main.altarray3[None,None,None,None,:,None,None,None,None])*2./main.dt) + main.RHS[:]
    Rstar = np.einsum('zij...,zj...->zi...',JinvX,Rstar)
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  def create_MF_JacobianX(v,args,main):
    an = args[0]
    Rn = args[1]
    #print(np.linalg.norm(JinvX))
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-2
    main.a.a[:] = an + eps*vr
    getRHS_zeta(main,main,eqns)
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
    Av = (volint_t - (futureFlux[:,:,:,:,None] - pastFlux[:,:,:,:,None]*main.altarray3[None,None,None,None,:,None,None,None,None])*2./main.dt) + 1./eps * (R1 - Rn)
    Av = np.einsum('zij...,zj...->zi...',JinvX,Av)
    return Av.flatten()

  def unsteadyResidualY(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    JinvY[:] = computeJacobianY(main,eqns,unsteadyResidual_element_Y)

    getRHS_eta(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    Rstar = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    ## now integrate the volume term ( \int ( dw/dt * u) )
    volint_t = main.basis.volIntegrateGlob(main,main.a.u*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.wp3)*2./main.dt
    uFuture,uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)
    futureFlux = main.basis.faceIntegrateGlob(main,uFuture*main.Jdet[None,:,:,:,:,:,:,None],main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    uPast[:,:,:,:,:,:,:,0] = main.a.uFuture[:,:,:,:,:,:,:,-1]
    if (main.Npt > 1):
      uPast[:,:,:,:,:,:,:,1::] = uFuture[:,:,:,:,:,:,:,0:-1]
    pastFlux   = main.basis.faceIntegrateGlob(main,uPast*main.Jdet[None,:,:,:,:,:,:,None] ,main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    Rstar[:] = (volint_t - (futureFlux[:,:,:,:,None] - pastFlux[:,:,:,:,None]*main.altarray3[None,None,None,None,:,None,None,None,None])*2./main.dt) + main.RHS[:]
    Rstar = np.einsum('zmij...,zmj...->zmi...',JinvY,Rstar)
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  def create_MF_JacobianY(v,args,main):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-2
    main.a.a[:] = an + eps*vr
    getRHS_eta(main,main,eqns)
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
    Av = (volint_t - (futureFlux[:,:,:,:,None] - pastFlux[:,:,:,:,None]*main.altarray3[None,None,None,None,:,None,None,None,None])*2./main.dt) + 1./eps * (R1 - Rn)
    Av = np.einsum('zmij...,zmj...->zmi...',JinvY,Av)
    return Av.flatten()



#  unsteadyResidual(main.a.a)
  nonlinear_solver.solve(unsteadyResidualX, create_MF_JacobianX,main,linear_solver,sparse_quadrature,eqns)

  main.a.uFuture[:],uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)
  nonlinear_solver.solve(unsteadyResidualY, create_MF_JacobianY,main,linear_solver,sparse_quadrature,eqns)


  #sol = linear_solver.solve(create_MF_Jacobian,-Rstar_pc.flatten(), np.zeros(np.size(main.a.a)),main,MF_Jacobian_args,1e-8,linear_solver.maxiter_outer,500,True)
  main.t += main.dt*main.Npt
  main.iteration += 1
  main.a.uFuture[:],uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)




def spaceTimePC(main,MZ,eqns,args=None):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  main.a0[:] = main.a.a[:]

  def unsteadyResidual_element(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    getRHS_element(main,main,eqns)
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
#    Rstar = main.a.a[:] + 2.*Rstar
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  def unsteadyResidual_element_X(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    getRHS_element_zeta(main,main,eqns)
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
#    Rstar = main.a.a[:] + 2.*Rstar
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  def unsteadyResidual_element_Y(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    getRHS_element_eta(main,main,eqns)
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
 #   Rstar = main.a.a[:] + 2.*Rstar
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  
  JinvX = computeJacobianX(main,eqns,unsteadyResidual_element_X)
  JinvY = computeJacobianY(main,eqns,unsteadyResidual_element_Y)

  def unsteadyResidual(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    eqns.getRHS(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    Rstar = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    ## now integrate the volume term ( \int ( dw/dt * u) )
    volint_t = main.basis.volIntegrateGlob(main,main.a.u*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.wp3)*2./main.dt
    uFuture,uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)
    futureFlux = main.basis.faceIntegrateGlob(main,uFuture*main.Jdet[None,:,:,:,:,:,:,None],main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    uPast[:,:,:,:,:,:,:,0] = main.a.uFuture[:,:,:,:,:,:,:,-1]
    if (main.Npt > 1):
      uPast[:,:,:,:,:,:,:,1::] = uFuture[:,:,:,:,:,:,:,0:-1]
    pastFlux   = main.basis.faceIntegrateGlob(main,uPast*main.Jdet[None,:,:,:,:,:,:,None] ,main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    Rstar[:] = (volint_t - (futureFlux[:,:,:,:,None] - pastFlux[:,:,:,:,None]*main.altarray3[None,None,None,None,:,None,None,None,None])*2./main.dt) + main.RHS[:]
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
    vr_phys = main.basis.reconstructUGeneral(main,vr)
    volint_t = main.basis.volIntegrateGlob(main,vr_phys*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.wp3)*2./main.dt
    uFuture,uPast = main.basis.reconstructEdgesGeneralTime(vr,main)
    futureFlux = main.basis.faceIntegrateGlob(main,uFuture*main.Jdet[None,:,:,:,:,:,:,None],main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    uPast[:,:,:,:,:,:,:,0] = 0
    if (main.Npt > 1):
      uPast[:,:,:,:,:,:,:,1::] = uFuture[:,:,:,:,:,:,:,0:-1]
    pastFlux   = main.basis.faceIntegrateGlob(main,uPast*main.Jdet[None,:,:,:,:,:,:,None]  ,main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
    Av = (volint_t - (futureFlux[:,:,:,:,None] - pastFlux[:,:,:,:,None]*main.altarray3[None,None,None,None,:,None,None,None,None])*2./main.dt) + 1./eps * (R1 - Rn)
    return Av.flatten()



  def Minv(v,main):
    ai = np.zeros(np.shape(main.a.a))
    ai[:] = main.a.a[:]
    #main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    vr = np.reshape(v,np.shape(main.a.a))

    def unsteadyResidualX(v):
      main.a.a[:] = np.reshape(v,np.shape(main.a.a))
      #JinvX[:] = computeJacobianX(main,eqns,unsteadyResidual_element_X)
  
      getRHS_zeta(main,main,eqns)
      R1 = np.zeros(np.shape(main.RHS))
      Rstar = np.zeros(np.shape(main.RHS))
  
      R1[:] = main.RHS[:]
      ## now integrate the volume term ( \int ( dw/dt * u) )
      volint_t = main.basis.volIntegrateGlob(main,main.a.u*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.wp3)*2./main.dt
      uFuture,uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)
      futureFlux = main.basis.faceIntegrateGlob(main,uFuture*main.Jdet[None,:,:,:,:,:,:,None],main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
      uPast[:,:,:,:,:,:,:,0] = main.a.uFuture[:,:,:,:,:,:,:,-1]
      if (main.Npt > 1):
        uPast[:,:,:,:,:,:,:,1::] = uFuture[:,:,:,:,:,:,:,0:-1]
      pastFlux   = main.basis.faceIntegrateGlob(main,uPast*main.Jdet[None,:,:,:,:,:,:,None] ,main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
      Rstar[:] = (volint_t - (futureFlux[:,:,:,:,None] - pastFlux[:,:,:,:,None]*main.altarray3[None,None,None,None,:,None,None,None,None])*2./main.dt) + main.RHS[:]
      Rstar = np.einsum('zij...,zj...->zi...',JinvX,Rstar)
      return Rstar,R1
  
    def create_MF_JacobianX(v,args,main):
      an = args[0]
      Rn = args[1]
      #print(np.linalg.norm(JinvX))
      vr = np.reshape(v,np.shape(main.a.a))
      eps = 5.e-2
      main.a.a[:] = an + eps*vr
      getRHS_zeta(main,main,eqns)
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
      Av = (volint_t - (futureFlux[:,:,:,:,None] - pastFlux[:,:,:,:,None]*main.altarray3[None,None,None,None,:,None,None,None,None])*2./main.dt) + 1./eps * (R1 - Rn)
      Av = np.einsum('zij...,zj...->zi...',JinvX,Av)
      return Av.flatten()

    def unsteadyResidualY(v):
      main.a.a[:] = np.reshape(v,np.shape(main.a.a))
      JinvY[:] = computeJacobianY(main,eqns,unsteadyResidual_element_Y)
  
      getRHS_eta(main,main,eqns)
      R1 = np.zeros(np.shape(main.RHS))
      Rstar = np.zeros(np.shape(main.RHS))
      R1[:] = main.RHS[:]
      ## now integrate the volume term ( \int ( dw/dt * u) )
      volint_t = main.basis.volIntegrateGlob(main,main.a.u*main.Jdet[None,:,:,:,None,:,:,:,None],main.w0,main.w1,main.w2,main.wp3)*2./main.dt
      uFuture,uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)
      futureFlux = main.basis.faceIntegrateGlob(main,uFuture*main.Jdet[None,:,:,:,:,:,:,None],main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
      uPast[:,:,:,:,:,:,:,0] = main.a.uFuture[:,:,:,:,:,:,:,-1]
      if (main.Npt > 1):
        uPast[:,:,:,:,:,:,:,1::] = uFuture[:,:,:,:,:,:,:,0:-1]
      pastFlux   = main.basis.faceIntegrateGlob(main,uPast*main.Jdet[None,:,:,:,:,:,:,None] ,main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
      Rstar[:] = (volint_t - (futureFlux[:,:,:,:,None] - pastFlux[:,:,:,:,None]*main.altarray3[None,None,None,None,:,None,None,None,None])*2./main.dt) + main.RHS[:]
      Rstar = np.einsum('zmij...,zmj...->zmi...',JinvY,Rstar)
      return Rstar,R1
  
    def create_MF_JacobianY(v,args,main):
      an = args[0]
      Rn = args[1]
      vr = np.reshape(v,np.shape(main.a.a))
      eps = 5.e-2
      main.a.a[:] = an + eps*vr
      getRHS_eta(main,main,eqns)
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
      Av = (volint_t - (futureFlux[:,:,:,:,None] - pastFlux[:,:,:,:,None]*main.altarray3[None,None,None,None,:,None,None,None,None])*2./main.dt) + 1./eps * (R1 - Rn)
      Av = np.einsum('zmij...,zmj...->zmi...',JinvY,Av)
      return Av.flatten()




    Rstarn_l,Rn_l = unsteadyResidualX(main.a.a)
    MF_Jacobian_args = [ai,Rn_l]
    sol = GMRes(create_MF_JacobianX,-Rstarn_l.flatten(), np.zeros(np.size(main.a.a)),main,MF_Jacobian_args,1e-6,linear_solver.maxiter_outer,10,False)
    vr = np.reshape(sol,np.shape(main.a.a))
    main.a.a[:] = ai[:] + vr[:]
#    Rstarn_l,Rn_l = unsteadyResidualY(main.a.a + vr)
#main.a.uFuture[:],uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)
#    MF_Jacobian_args = [ai,Rn_l]
#    sol = GMRes(create_MF_JacobianY,-Rstarn_l.flatten(), np.zeros(np.size(main.a.a)),main,MF_Jacobian_args,1e-6,linear_solver.maxiter_outer,10,False)
#    vr = vr + np.reshape(sol,np.shape(main.a.a))
 #   main.a.uFuture[:],uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)
#$    nonlinear_solver.solve(unsteadyResidualY, create_MF_JacobianY,main,linear_solver,sparse_quadrature,eqns)
#    vr = main.a.a[:] - ai[:]
#    main.a.a[:] = ai[:]
#    vr = np.einsum('zij...,zj...->zi...',JinvX,vr)
#    main.a.a[:] = a0[:]
    return vr.flatten()

  newtonSolver_PC2(unsteadyResidual, create_MF_Jacobian,Minv,main,linear_solver,sparse_quadrature,eqns)
  main.a.uFuture[:],uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)
  main.t += main.dt*main.Npt
  main.iteration += 1
  main.a.uFuture[:],uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)












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
  #gamma = 1.4
  R = 8314.4621/1000.
  atmp = np.zeros(np.shape(main.a.a))
  atmp[:,0,0,0] = main.a.a[:,0,0,0]
  #atmp[:] = main.a.a[:] 
  main.basis.reconstructU(main,main.a)
  u0 = np.zeros(np.shape(main.a.u))
  u0[:] = main.a.u[:]
  Y0 = np.zeros(np.shape(main.a.u[5::]))
  Y0[:] = main.a.u[5::]/main.a.u[None,0]
  U = main.basis.reconstructUGeneral(main,atmp)
  Y_N2 = 1. - np.sum(U[5::]/U[None,0],axis=0)
  Winv =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],U[5::]/U[None,0]) + 1./main.W[-1]*Y_N2
  Cp = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],U[5::]/U[None,0]) + main.Cp[-1]*Y_N2
  Cv = Cp - R*Winv
  gamma = Cp/Cv
  
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
    indx = (sign(Cwf0[:,pindx])==sign(dcR[:,pindx-1])) & (sign(dcR[:,pindx-1])==sign(dcL[:,pindx-1] ))
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
  u2[5::] = u2[None,0]*Y0

  #print(np.linalg.norm(main.a.a))
  #main.a.a[:] = main.basis.volIntegrateGlob(main,u2,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]
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
 
def SSP_RK3(main,MZ,eqns,args=None):
  main.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
  #print(np.amax(main.a.p) - np.amin(main.a.p))

  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  main.basis.applyMassMatrix(main,main.RHS)
  a1 = main.a.a[:]  + main.dt*(main.RHS[:])
  main.a.a[:] = a1[:]
  #limiter_characteristic(main)
  #limiter_MF(main)

  main.getRHS(main,MZ,eqns)
  main.basis.applyMassMatrix(main,main.RHS)

  a1[:] = 3./4.*a0 + 1./4.*(a1 + main.dt*main.RHS[:]) #reuse a1 vector
  main.a.a[:] = a1[:]
  #limiter_characteristic(main)
  #limiter_MF(main)

  main.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
  main.basis.applyMassMatrix(main,main.RHS)
  main.a.a[:] = 1./3.*a0 + 2./3.*(a1[:] + main.dt*main.RHS[:])
  #limiter_characteristic(main)
  #limiter_MF(main)


  main.t += main.dt
  main.iteration += 1
#  plot(main.a.p[0,0,0,0,:,0,0,0]/1000.,color='green')
#  ylim([99.95,100.05]) 
#  pause(0.001)

  #limiter_characteristic(main)

def SSP_RK3_DOUBLEFLUX(main,MZ,eqns,args=None):
  R = 8314.4621/1000.
  af = np.zeros(np.shape(main.a.a))
  af[:] = main.a.a[:]
  af[:,1::] = 0.
  uf = main.basis.reconstructUGeneral(main,af)
  ## Compute gamma_star and this is frozen
  Y_last = 1. - np.sum(uf[5::]/uf[None,0],axis=0)
  Winv =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],uf[5::]/uf[None,0]) + 1./main.W[-1]*Y_last
  Cp = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],uf[5::]/uf[0]) + main.Cp[-1]*Y_last
  Cv = Cp - R*Winv 
  main.a.gamma_star[:] = Cp/Cv

  KE = 0.5*(main.a.u[1]**2 + main.a.u[2]**2 + main.a.u[3]**2)/main.a.u[0]
  main.a.p = (main.a.gamma_star - 1.)*( main.a.u[4] - KE)
  #print('Start',np.amax(main.a.p) - np.amin(main.a.p))

  # now get RHS with gamma_star managing thermo
  main.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  a1 = main.a.a[:]  + main.dt*(main.RHS[:])
  main.a.a[:] = a1[:]
  limiter_MF(main)

  main.getRHS(main,MZ,eqns)
  a1[:] = 3./4.*a0 + 1./4.*(a1 + main.dt*main.RHS[:]) #reuse a1 vector
  main.a.a[:] = a1[:]
  limiter_MF(main)

  main.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
  main.a.a[:] = 1./3.*a0 + 2./3.*(a1[:] + main.dt*main.RHS[:])
  limiter_MF(main)

#
#  # now update the thermodynamic state and relax energy
  main.basis.reconstructU(main,main.a)
  # compute pressure from new values of u but old value of gamma_star
  KE = 0.5*(main.a.u[1]**2 + main.a.u[2]**2 + main.a.u[3]**2)/main.a.u[0]
  main.a.p = (main.a.gamma_star - 1.)*( main.a.u[4] - KE)
  #print('End',np.amax(main.a.p) - np.amin(main.a.p))
  # now update gamma_star
  af[:] = main.a.a[:]
  af[:,1::] = 0.
  uf = main.basis.reconstructUGeneral(main,af)
  Y_last = 1. - np.sum(uf[5::]/uf[None,0],axis=0)
  Winv =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],uf[5::]/uf[None,0]) + 1./main.W[-1]*Y_last
  #main.a.T = main.a.p/(main.a.u[0]*R*Winv) 
  Cp = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],uf[5::]/uf[0]) + main.Cp[-1]*Y_last
  Cv = Cp - R*Winv
  main.a.gamma_star[:] = Cp/Cv

  # now update state with new gamma_star
  main.a.u[4] = main.a.p/(main.a.gamma_star - 1.) + KE

  # finally project this back to modal space
  ord_arrx= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arry= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arrz= np.linspace(0,main.order[2]-1,main.order[2])
  ord_arrt= np.linspace(0,main.order[3]-1,main.order[3])
  scale =  (2.*ord_arrx[:,None,None,None] + 1.)*(2.*ord_arry[None,:,None,None] + 1.)*(2.*ord_arrz[None,None,:,None] + 1.)*(2.*ord_arrt[None,None,None,:] + 1.)/16.
  main.a.a[:] = main.basis.volIntegrateGlob(main,main.a.u,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]



  main.t += main.dt
  main.iteration += 1
  #limiter_characteristic(main)

 
def ExplicitRK4(main,MZ,eqns,args=None):
  main.a0[:] = main.a.a[:]
  rk4const = np.array([1./4,1./3,1./2,1.])
#  main.basis.reconstructU(main,main.a)
#  main.a.p[:],main.a.T[:] = computePressure_and_Temperature(main,main.a.u)
#  Y_N2 = 1. - np.sum(main.a.u[5::]/main.a.u[None,0],axis=0)
#  gamma = np.einsum('i...,ijk...->jk...',main.gamma[0:-1],main.a.u[5::]/main.a.u[None,0]) + main.gamma[-1]*Y_N2
#  c = np.amax(np.sqrt(gamma*main.a.p/main.a.u[0]))
#  umax = np.sqrt( np.amax( (main.a.u[1]/main.a.u[0])**2 ) + np.amax(  (main.a.u[2]/main.a.u[0])**2 ) + np.amax( (main.a.u[3]/main.a.u[0])**2 ) )
#  #CFL = c*dt/dx -> dt = CFL*dx/c
#  main.dt = 0.1*main.dx/(c + umax)
#  if (main.mpi_rank == 0):
#    print(main.dt)
  for i in range(0,4):
    main.rkstage = i
    main.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
    main.basis.applyMassMatrix(main,main.RHS) 
    main.a.a[:] = main.a0 + main.dt*rk4const[i]*main.RHS
    #limiter_MF(main)
  main.t += main.dt
  main.iteration += 1


def SteadyState(main,MZ,eqns,args):
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

  nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns)
  main.t += main.dt
  main.iteration += 1


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




def CrankNicolsonIncomp(main,MZ,eqns,args):
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
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Rstar_time = ( main.a.a[:] - main.a0 )
    Rstar_time[-1] = 0.
    Rstar = ( main.a.a[:] - main.a0 ) - 0.5*main.dt*(R0 + R1)
    Rstar[-1] = R1[-1]/50.
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  def create_MF_Jacobian(v,args,main):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-3
    main.a.a[:] = an + eps*vr
    eqns.getRHS(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    vrtmp = np.zeros(np.shape(vr))
    vrtmp[:] = vr[:]
    vrtmp[-1] = 0.
    Av = vrtmp - main.dt/2.*(R1 - Rn)/eps
    Av[-1] = (R1[-1] - Rn[-1])/eps/50.
    return Av.flatten()

  nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns)
  main.t += main.dt
  main.iteration += 1

def fractionalStep(main,MZ,eqns,args):
  ord_arr0= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arr1= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arr2= np.linspace(0,main.order[2]-1,main.order[2])
  ord_arr3= np.linspace(0,main.order[3]-1,main.order[3])
  scale =  (2.*ord_arr0[:,None,None,None] + 1.)*(2.*ord_arr1[None,:,None,None] + 1.)*(2.*ord_arr2[None,None,:,None]+1.)*(2.*ord_arr3[None,None,None,:] + 1. )/16.


  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  pressure_linear_solver = args[3]
  main.a0[:] = main.a.a[:]
  eqns.getRHS(main,main,eqns)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]

  def unsteadyResidual(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    eqns.getRHS(main,main,eqns)
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
    eqns.getRHS(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Av = vr - main.dt/2.*(R1 - Rn)/eps
    return Av.flatten()

  # solve for velocity
  nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns)

  # now we need to solve for the pressure correction

  right_bc = 'neumann'
  left_bc = 'neumann'
  top_bc = 'neumann'
  bottom_bc = 'neumann'
  #right_bc = 'dirichlet'
  #left_bc = 'dirichlet'
  #top_bc = 'dirichlet'
  #bottom_bc = 'dirichlet'

  right_bc_args = [0]
  left_bc_args = [0]
  top_bc_args = [0]
  bottom_bc_args = [0]
  BCs = [right_bc,right_bc_args,top_bc,top_bc_args,left_bc,left_bc_args,bottom_bc,bottom_bc_args]


  eqnsPoisson = equations('Diffusion',('central','BR1'),'none' )
  poisson_main = variables(main.Nel,main.order,main.quadpoints,eqnsPoisson,1,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,main.turb_str,main.procx,main.procy,\
                         main.BCs,None,0,False,False)
  main.a.Upx,main.a.Upy,main.a.Upz = main.basis.diffU(main.a.a,main)
  div = main.a.Upx[0] + main.a.Upy[1] + main.a.Upz[2]
  #div = div - np.mean(div)
  print('Start div = ' ,np.linalg.norm(div),np.sum(div))
  force =(div)/main.dt
  source = main.basis.volIntegrateGlob(poisson_main, force ,poisson_main.w0,poisson_main.w1,poisson_main.w2,poisson_main.w3)*scale[None,:,:,:,:,None,None,None,None]
  print(np.shape(force))
  div_int = np.sum(source) 
  vol_int = main.basis.volIntegrateGlob(poisson_main, np.ones(np.shape(force) ) ,poisson_main.w0,poisson_main.w1,poisson_main.w2,poisson_main.w3)*scale[None,:,:,:,:,None,None,None,None]
  vol_int = np.sum(vol_int)
#  print(np.sum(source2))
#  difference = (np.sum(source) )/(2.*np.pi)**3
  source = main.basis.volIntegrateGlob(poisson_main, force - div_int/vol_int ,poisson_main.w0,poisson_main.w1,poisson_main.w2,poisson_main.w3)*scale[None,:,:,:,:,None,None,None,None]
  print(np.sum(source))
  def unsteadyResidual_poisson(v):
    poisson_main.a.a[:] = np.reshape(v[:],np.shape(poisson_main.a.a))
    eqnsPoisson.getRHS(poisson_main,poisson_main,eqnsPoisson)
    R1 = np.zeros(np.shape(poisson_main.RHS))
    R1[:] = poisson_main.RHS[:]
    Rstar = R1 - source 
    Rstar_glob = gatherResid(Rstar,poisson_main)
    Rstar = Rstar.flatten()
    return Rstar,R1,Rstar_glob

  def create_MF_Jacobian_poisson(v,args,poisson_main):
    vr = np.reshape(v[:],np.shape(poisson_main.a.a))
    poisson_main.a.a[:] = vr
    eqnsPoisson.getRHS(poisson_main,poisson_main,eqnsPoisson)
    R1 = np.zeros(np.shape(poisson_main.RHS))
    R1[:] = poisson_main.RHS[:]
    Av = ( R1 ).flatten()
    return Av.flatten()

  nonlinear_solver.solve(unsteadyResidual_poisson, create_MF_Jacobian_poisson,poisson_main,linear_solver,False,eqnsPoisson)
#  v = np.zeros(np.size(poisson_main.a.a))
#  v[:] = poisson_main.a.a.flatten()
#  Rstarn,Rn,Rstar_glob = unsteadyResidual_poisson(v)
#  old = np.zeros(np.size(poisson_main.a.a))
#  sol = pressure_linear_solver.solve(create_MF_Jacobian_poisson, -Rstarn.flatten(), old.flatten(),poisson_main,[],1e-7,linear_solver.maxiter_outer,500,True)
#  poisson_main.a.a[:] = np.reshape(sol[:],np.shape(poisson_main.a.a))
  ## Now fix velocity
  px,py,pz = poisson_main.basis.diffU(poisson_main.a.a,poisson_main)
  main.a.u[0] = main.a.u[0] - main.dt*px
  main.a.u[1] = main.a.u[1] - main.dt*py
  main.a.u[2] = main.a.u[2] - main.dt*pz
  main.a.a[:] = main.basis.volIntegrateGlob(main,main.a.u ,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]
  main.a.Upx,main.a.Upy,main.a.Upz = main.basis.diffU(main.a.a,main)
  div = main.a.Upx[0] + main.a.Upy[1] + main.a.Upz[2]
  print('End div = ' ,np.linalg.norm(div))
  main.t += main.dt
  main.iteration += 1
  main.a.p[:] = poisson_main.a.u[0]
#  contourf(poisson_main.a.a[0,0,0,0,0,:,:,0,0],100)
#  colorbar()
#  pause(.2)
#  clf() 

def CrankNicolsonEntropy(main,MZ,eqns,args):
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


  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  main.a0[:] = main.a.a[:]
  eqns.getRHS(main,main,eqns)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  U0 = entropy_to_conservative(main.a.u)
  ord_arr0= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arr1= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arr2= np.linspace(0,main.order[2]-1,main.order[2])
  ord_arr3= np.linspace(0,main.order[3]-1,main.order[3])
  scale =  (2.*ord_arr0[:,None,None,None] + 1.)*(2.*ord_arr1[None,:,None,None] + 1.)*(2.*ord_arr2[None,None,:,None]+1.)*(2.*ord_arr3[None,None,None,:] + 1. )/16.

  def unsteadyResidual(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    eqns.getRHS(main,main,eqns)
    U = entropy_to_conservative(main.a.u)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    ## compute volume integral term
    time_integral = main.basis.volIntegrateGlob(main,U - U0,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]
    Rstar = time_integral  - 0.5*main.dt*(R0 + R1)
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  def create_MF_Jacobian(v,args,main):
    an = args[0]
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-2
    main.a.a[:] = an
    Rn,dum,dum = unsteadyResidual(main.a.a)
    main.a.a[:] = an + eps*vr
    R1,dum,dum = unsteadyResidual(main.a.a)
    Av = (R1 - Rn)/eps
    return Av.flatten()

  nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns)
  #atmp = newton_krylov(unsteadyResidual, main.a.a.flatten(), iter=None, rdiff=None, method='lgmres', inner_maxiter=20, inner_M=None, outer_k=10, verbose=True)
  #main.a.a[:] = np.reshape(atmp,np.shape(main.a.a))
  main.t += main.dt
  main.iteration += 1



def CrankNicolson(main,MZ,eqns,args):
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
    R1 = np.zeros(np.shape(main.RHS))
    RHS_CN = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    RHS_CN[:] = 0.5*main.dt*(R0 + R1)
    main.basis.applyMassMatrix(main,RHS_CN)
    Rstar = ( main.a.a[:] - main.a0 ) - RHS_CN 
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  def create_MF_Jacobian(v,args,main):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-2
    main.a.a[:] = an + eps*vr
    eqns.getRHS(main,main,eqns)
    RHS_CN = np.zeros(np.shape(main.RHS))
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    RHS_CN[:] = 0.5*main.dt*(R1 - Rn)/eps
    main.basis.applyMassMatrix(main,RHS_CN)
    Av = vr - RHS_CN

    return Av.flatten()

  nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns,None)
  main.t += main.dt
  main.iteration += 1

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

  print "Number of SNES iterations = %d" % its
  print "Number of Linear iterations = %d" % lits

  sol = x[...]
  main.a.a =   np.reshape(sol,np.shape(main.a.a))
  main.a0[:] = np.reshape(sol,np.shape(main.a.a))
  main.t += main.dt
  main.iteration += 1

