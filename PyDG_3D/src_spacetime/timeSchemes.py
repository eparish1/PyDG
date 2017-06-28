import numpy as np
#import sys, petsc4py
#petsc4py.init(sys.argv)
#from petsc4py import PETSc
import matplotlib.pyplot as plt
from mpi4py import MPI
import sys
from init_Classes import variables,equations
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres,bicgstab
from scipy.sparse.linalg import lgmres
from scipy.optimize import newton_krylov
from myGMRES import GMRes
from scipy.optimize.nonlin import InverseJacobian
from scipy.optimize.nonlin import BroydenFirst, KrylovJacobian
import time
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


def spaceTime(main,MZ,eqns,args=None):
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
    Rstar = volint_t - (futureFlux[:,:,:,:,None] - pastFlux[:,:,:,:,None]*main.altarray3[None,None,None,None,:,None,None,None,None])*scale[None,:,:,:,:,None,None,None,None]*2./main.dt + main.RHS[:]
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
    Av = volint_t - (futureFlux[:,:,:,:,None] - pastFlux[:,:,:,:,None]*main.altarray3[None,None,None,None,:,None,None,None,None])*scale[None,:,:,:,:,None,None,None,None]*2./main.dt + \
         1./eps * (R1 - Rn) 
    return Av.flatten()

#  def create_MF_Jacobian_PC(v,args,main):
#    vr = np.reshape(v,np.shape(main.a.a))
#    vr_phys = main.basis.reconstructUGeneral(main,vr)
#
#    ord_arr0= np.linspace(0,main.order[0]-1,main.order[0])
#    ord_arr1= np.linspace(0,main.order[1]-1,main.order[1])
#    ord_arr2= np.linspace(0,main.order[2]-1,main.order[2])
#    ord_arr3= np.linspace(0,main.order[3]-1,main.order[3])
#    scale =  (2.*ord_arr0[:,None,None,None] + 1.)*(2.*ord_arr1[None,:,None,None] + 1.)*(2.*ord_arr2[None,None,:,None]+1.)*(2.*ord_arr3[None,None,None,:] + 1. )/16.
#    eps = 5.e-3
#    an = args[0]
#    Rn = args[1]
#
#    main.a.a[:] = an[:] + vr[:]
#    main.basis.reconstructU(main,main.a)
#    # evaluate inviscid flux
#    eqns.evalFluxX(main.a.u,main.iFlux.fx,[])
#    eqns.evalFluxY(main.a.u,main.iFlux.fy,[])
#    eqns.evalFluxZ(main.a.u,main.iFlux.fz,[])
#  
#    dxi = 2./main.dx2[None,None,None,None,:,None,None,None]*scale[:,:,:,:,None,None,None,None]*np.ones(np.shape(main.a.a[0]))
#    dyi = 2./main.dy2[None,None,None,None,None,:,None,None]*scale[:,:,:,:,None,None,None,None]*np.ones(np.shape(main.a.a[0]))
#    dzi = 2./main.dz2[None,None,None,None,None,None,:,None]*scale[:,:,:,:,None,None,None,None]*np.ones(np.shape(main.a.a[0]))
#    v1ijk = main.basis.volIntegrateGlob(main,main.iFlux.fx,main.wp0,main.w1,main.w2,main.w3)*dxi[None]
#    v2ijk = main.basis.volIntegrateGlob(main,main.iFlux.fy,main.w0,main.wp1,main.w2,main.w3)*dyi[None]
#    v3ijk = main.basis.volIntegrateGlob(main,main.iFlux.fz,main.w0,main.w1,main.wp2,main.w3)*dzi[None]
#    R1 = v1ijk + v2ijk + v3ijk
#    volint_t = main.basis.volIntegrateGlob(main,vr_phys,main.w0,main.w1,main.w2,main.wp3)*scale[None,:,:,:,:,None,None,None,None]*2./main.dt
#    uFuture,uPast = main.basis.reconstructEdgesGeneralTime(vr,main)
#    futureFlux = main.basis.faceIntegrateGlob(main,uFuture,main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
#    pastFlux   = main.basis.faceIntegrateGlob(main,main.a.uFuture  ,main.w0,main.w1,main.w2,main.weights0,main.weights1,main.weights2)
#    Av = R1#volint_t - (futureFlux[:,:,:,:,None])*scale[None,:,:,:,:,None,None,None,None]*2./main.dt 
    return Av.flatten()

#  unsteadyResidual(main.a.a)
  nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns)
  main.t += main.dt*main.Npt
  main.iteration += 1
  main.a.uFuture[:],uPast = main.basis.reconstructEdgesGeneralTime(main.a.a,main)

  

def ExplicitRK4(main,MZ,eqns,args=None):
  main.a0[:] = main.a.a[:]
  rk4const = np.array([1./4,1./3,1./2,1.])
  for i in range(0,4):
    main.rkstage = i
    main.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
    main.a.a[:] = main.a0 + main.dt*rk4const[i]*(main.RHS[:])
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
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Rstar = R1
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
    return Av.flatten()

  nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns)
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
    R1[:] = main.RHS[:]
    Rstar = ( main.a.a[:] - main.a0 ) - 0.5*main.dt*(R0 + R1)
    Rstar_glob = gatherResid(Rstar,main)
    return Rstar,R1,Rstar_glob

  def create_MF_Jacobian(v,args,main):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(main.a.a))


#    eqnsLin = equations('Linearized Navier-Stokes',('central','Inviscid'),'DNS')
#    main.a.a[:] = an[:]
#    vr_phys = main.basis.reconstructUGeneral(main,vr)
#    eqnsLin.getRHS(main,main,eqnsLin,[vr_phys])
#    R2 = np.zeros(np.shape(main.RHS))
#    R2[:] = main.RHS[:]
#    Av2 = vr - main.dt/2.*R2



    eps = 5.e-2
    main.a.a[:] = an + eps*vr
    eqns.getRHS(main,main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Av = vr - main.dt/2.*(R1 - Rn)/eps


    return Av.flatten()

  nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,main,linear_solver,sparse_quadrature,eqns)
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

