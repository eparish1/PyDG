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
from eos_functions import *
import time

from pylab import *
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
  a1 = main.a.a[:]  + main.dt*(main.RHS[:])
  main.a.a[:] = a1[:]
  #limiter_characteristic(main)
  #limiter_MF(main)

  main.getRHS(main,MZ,eqns)
  a1[:] = 3./4.*a0 + 1./4.*(a1 + main.dt*main.RHS[:]) #reuse a1 vector
  main.a.a[:] = a1[:]
  #limiter_characteristic(main)
  #limiter_MF(main)

  main.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
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
  main.basis.reconstructU(main,main.a)
  ## Compute gamma_star and this is frozen
  Y_last = 1. - np.sum(main.a.u[5::]/main.a.u[None,0],axis=0)
  Winv =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],main.a.u[5::]/main.a.u[None,0]) + 1./main.W[-1]*Y_last
  Cp = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],main.a.u[5::]/main.a.u[0]) + main.Cp[-1]*Y_last
  Cv = Cp - R*Winv 
  main.a.gamma_star[:] = Cp/Cv

  KE = 0.5*(main.a.u[1]**2 + main.a.u[2]**2 + main.a.u[3]**2)/main.a.u[0]
  main.a.p = (main.a.gamma_star - 1.)*( main.a.u[4] - KE)
  #print(np.amax(diff(main.a.p[0,0,0,0,:,0,0,0])))

  # now get RHS with gamma_star managing thermo
  main.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  a1 = main.a.a[:]  + main.dt*(main.RHS[:])
  main.a.a[:] = a1[:]
  #limiter_characteristic(main)

#  main.getRHS(main,MZ,eqns)
#  a1[:] = 3./4.*a0 + 1./4.*(a1 + main.dt*main.RHS[:]) #reuse a1 vector
#  main.a.a[:] = a1[:]
#  #limiter_characteristic(main)
#  main.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
#  main.a.a[:] = 1./3.*a0 + 2./3.*(a1[:] + main.dt*main.RHS[:])
#
#  # now update the thermodynamic state and relax energy
  main.basis.reconstructU(main,main.a)
  # compute pressure from new values of u but old value of gamma_star
  KE = 0.5*(main.a.u[1]**2 + main.a.u[2]**2 + main.a.u[3]**2)/main.a.u[0]
  main.a.p = (main.a.gamma_star - 1.)*( main.a.u[4] - KE)
  print(np.amax(diff(main.a.p[0,0,0,0,:,0,0,0])))

  # now update gamma_star
  Y_last = 1. - np.sum(main.a.u[5::]/main.a.u[None,0],axis=0)
  Winv =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],main.a.u[5::]/main.a.u[None,0]) + 1./main.W[-1]*Y_last
  #main.a.T = main.a.p/(main.a.u[0]*R*Winv) 
  Cp = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],main.a.u[5::]/main.a.u[0]) + main.Cp[-1]*Y_last
  Cv = Cp - R*Winv
  main.a.gamma_star[:] = Cp/Cv

  # now update state with new gamma_star
  #main.a.u[4] = main.a.p/(main.a.gamma_star - 1.) + KE

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
    main.a.a[:] = main.a0 + main.dt*rk4const[i]*(main.RHS[:])
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

