import matplotlib.pyplot as plt
import numpy as np
from MPI_functions import gatherSolSpectral, globalDot
from equations_class import *
from tensor_products import *
from navier_stokes import *#strongFormEulerXYZ
from fluxSchemes import generalFluxGen
from MPI_functions import sendEdgesGeneralSlab,sendEdgesGeneralSlab_Derivs,globalSum
from navier_stokes_entropy import entropy_to_conservative
#from jacobian_schemes import computeJacobian_full_pod
#from pylab import *
def orthogonalDynamics(main,MZ,eqns):
    ### EVAL RESIDUAL AND DO MZ STUFF
    filtarray = np.ones(np.shape(main.a.a))
    filtarray[:,main.rorder::] = 0.
    a0 = np.zeros(np.shape(main.a.a))
    a0[:] = main.a.a[:]
    eqns.getRHS(main,MZ,eqns)
    RHS1 = np.zeros(np.shape(main.RHS))
    RHS1[:] = main.RHS[:]
    main.a.a[:] = a0[:]*filtarray[:]
    eqns.getRHS(main,MZ,eqns)
    RHS2 = np.zeros(np.shape(main.RHS))
    RHS2[:] = main.RHS[:]
    main.RHS[:] = RHS1[:] - RHS2[:]
    main.a.a[:] = a0[:]


def DNS(main,MZ,eqns):
  eqns.getRHS(main,MZ,eqns)

def projection(main,U):
  ## First perform integration in x
  ord_arrx= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arry= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arrz= np.linspace(0,main.order[2]-1,main.order[2])
  ord_arrt= np.linspace(0,main.order[3]-1,main.order[3])
  scale =  (2.*ord_arrx[:,None,None,None] + 1.)*(2.*ord_arry[None,:,None,None] + 1.)*\
           (2.*ord_arrz[None,None,:,None] + 1.)*(2.*ord_arrt[None,None,None,:] + 1.)/16.
  a_project = volIntegrateGlob_tensordot(main,U,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]
  return a_project


def orthogonalProjectionEntropy(main,U,V):
  ## First perform integration in x
  a_project = volIntegrateGlob_tensordot(main,U,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]
  filta = np.zeros(np.shape(a_project))
  a_project = np.reshape(a_project,(main.nvars*main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt) )
  a_project = np.einsum('ij...,j...->i...',main.EMM,a_project)
  a_project = np.reshape(a_project,np.shape(main.a.a))

  V_project = main.basis.reconstructUGeneral(main,a_project)
  V_orthogonal = V - V_project
  return U_orthogonal

def testProjection(main,U):#,UR,UL,UU,UD,UF,UB):
  ## First perform integration in x
  ord_arrx= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arry= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arrz= np.linspace(0,main.order[2]-1,main.order[2])
  ord_arrt= np.linspace(0,main.order[3]-1,main.order[3])
  scale =  (2.*ord_arrx[:,None,None,None] + 1.)*(2.*ord_arry[None,:,None,None] + 1.)*\
           (2.*ord_arrz[None,None,:,None] + 1.)*(2.*ord_arrt[None,None,None,:] + 1.)/16.
  a_project = volIntegrateGlob_tensordot(main,U,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]
  filta = np.zeros(np.shape(a_project))
  filta[:,0:main.order[0]/2,0:main.order[1]/2,0:main.order[2]/2,:] = 1.
  U_project = main.basis.reconstructUGeneral(main,a_project*filta)
  return U_project


def orthogonalProjection(main,U):#,UR,UL,UU,UD,UF,UB):
  ## First perform integration in x
  ord_arrx= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arry= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arrz= np.linspace(0,main.order[2]-1,main.order[2])
  ord_arrt= np.linspace(0,main.order[3]-1,main.order[3])
  scale =  (2.*ord_arrx[:,None,None,None] + 1.)*(2.*ord_arry[None,:,None,None] + 1.)*\
           (2.*ord_arrz[None,None,:,None] + 1.)*(2.*ord_arrt[None,None,None,:] + 1.)/16.
  a_project = volIntegrateGlob_tensordot(main,U,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]
  filta = np.zeros(np.shape(a_project))
  #filta[:,0:1,0:1,0:1,:] = 1.
  #a_project *= filta
#  U_projectR,U_projectL,U_projectU,U_projectD,U_projectF,U_projectB = main.basis.reconstructEdgesGeneral(a_project,main)

  U_project = main.basis.reconstructUGeneral(main,a_project)
  U_orthogonal = U - U_project
#  U_orthogonalR = UR - U_projectR
#  U_orthogonalL = UL - U_projectL
#  U_orthogonalU = UU - U_projectU
#  U_orthogonalD = UD - U_projectD
#  U_orthogonalF = UF - U_projectF
#  U_orthogonalB = UB - U_projectB
  return U_orthogonal#,U_orthogonalR,U_orthogonalL,U_orthogonalU,U_orthogonalD,U_orthogonalF,U_orthogonalB

def orthogonalProjectionVol(main,U):
  ## First perform integration in x
  ord_arrx= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arry= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arrz= np.linspace(0,main.order[2]-1,main.order[2])
  ord_arrt= np.linspace(0,main.order[3]-1,main.order[3])
  scale =  (2.*ord_arrx[:,None,None,None] + 1.)*(2.*ord_arry[None,:,None,None] + 1.)*\
           (2.*ord_arrz[None,None,:,None] + 1.)*(2.*ord_arrt[None,None,None,:] + 1.)/16.
  a_project = volIntegrateGlob_tensordot(main,U,main.w0,main.w1,main.w2,main.w3)*scale[None,:,:,:,:,None,None,None,None]
  filta = np.zeros(np.shape(a_project))
  #filta[:,0:1,0:1,0:1,:] = 1.
  #a_project *= filta
  #U_projectR,U_projectL,U_projectU,U_projectD,U_projectF,U_projectB = main.basis.reconstructEdgesGeneral(a_project,main)
  U_project = main.basis.reconstructUGeneral(main,a_project)
  U_orthogonal = U - U_project
  return U_orthogonal

def orthogonalSubscale(main,MZ,eqns):
   eqns.getRHS(main,MZ,eqns)
   R0 = np.zeros(np.shape(main.RHS))
   R1 = np.zeros(np.shape(main.RHS))
   R0[:] = main.RHS[:]


   main.RHS[:] = 0.
#   R,RR,RL,RU,RD,RF,RB = strongFormEulerXYZ(main,main.a.a,None)
#   R_orthogonal,R_orthoR,R_orthoL,R_orthoU,R_orthoD,R_orthoF,R_orthoB = orthogonalProjection(main,R,RR,RL,RU,RD,RF,RB)
   R = eqns.strongFormResidual(main,main.a.a,None)
   R_orthogonal= orthogonalProjection(main,R)

   PLQLu2 = np.zeros(np.shape(main.RHS))
   u0 = main.a.u*1.

   main.a.u[:] = u0[:]
   eqns.evalFluxXYZLin(main,main.a.u,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,[-R_orthogonal])
   main.basis.applyVolIntegral(main,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,PLQLu2)
   main.RHS[:] = R0[:] 
   #indx0 = abs(PLQLu2[0,1,0,0,0,:,0,0,0]) >  abs(R0[0,1,0,0,0,:,0,0,0])
   #indx1 = abs(PLQLu2[1,1,0,0,0,:,0,0,0]) >  abs(R0[1,1,0,0,0,:,0,0,0])
   #indx2 = abs(PLQLu2[2,1,0,0,0,:,0,0,0]) >  abs(R0[2,1,0,0,0,:,0,0,0])
   #indx3 = abs(PLQLu2[3,1,0,0,0,:,0,0,0]) >  abs(R0[3,1,0,0,0,:,0,0,0])
   indx = 10.*abs(PLQLu2[4]) > (  abs(R0[4]) + 1e-3)
   main.a.a[:,indx] = 0.
   main.RHS[:,indx] = 0.
#   for i in range(main.order[0]-1,0,-1):
#     indx = np.ones(np.shape(PLQLu2[4,0]),dtype=bool)  #initialize an array with all trues
#     for j in range(main.order[0] - 1, i-1,-1):
#       chk = 100.*abs(PLQLu2[4,j]) > (  abs(R0[4,j]) + 1e-3)
#       indx = indx & chk 
#     main.a.a[0,i,indx] = 0.
#     main.RHS[0,i,indx] = 0.
#     main.a.a[1,i,indx] = 0.
#     main.RHS[1,i,indx] = 0.
#     main.a.a[2,i,indx] = 0.
#     main.RHS[2,i,indx] = 0.
#     main.a.a[3,i,indx] = 0.
#     main.RHS[3,i,indx] = 0.
#     main.a.a[4,i,indx] = 0.
#     main.RHS[4,i,indx] = 0.


def orthogonalSubscale2(main,MZ,eqns):
   eqns.getRHS(main,MZ,eqns)
   R0 = np.zeros(np.shape(main.RHS))
   R1 = np.zeros(np.shape(main.RHS))
   R0[:] = main.RHS[:]


   eps = 1.e-7
   main.RHS[:] = 0.
   R,RR,RL,RU,RD,RF,RB = strongFormEulerXYZ(main,main.a.a,None)
   #u0 = main.a.u*1.
   #main.a.u[:] = u0[:]
   R_orthogonal,R_orthoR,R_orthoL,R_orthoU,R_orthoD,R_orthoF,R_orthoB = orthogonalProjection(main,R,RR,RL,RU,RD,RF,RB)
   main.adum.uR[:] = -R_orthoR
   main.adum.uL[:] = -R_orthoL
   main.adum.uU[:] = -R_orthoU
   main.adum.uD[:] = -R_orthoD
   main.adum.uB[:] = -R_orthoF
   main.adum.uF[:] = -R_orthoB

#   R_orthogonal = R#orthogonalProjection(main,f)
#   main.adum.uR[:] = -RR
#   main.adum.uL[:] = -RL
#   main.adum.uU[:] = -RU
#   main.adum.uD[:] = -RD
#   main.adum.uB[:] = -RF
#   main.adum.uF[:] = -RB


   main.adum.uR_edge[:],main.adum.uL_edge[:],main.adum.uU_edge[:],main.adum.uD_edge[:],main.adum.uF_edge[:],main.adum.uB_edge[:] = sendEdgesGeneralSlab_Derivs(main.adum.uL,main.adum.uR,main.adum.uD,main.adum.uU,main.adum.uB,main.adum.uF,main)

   generalFluxGen(main,eqns,main.iFlux,main.a,eulerCentralFluxLinearized,[main.adum])
#   # now we need to integrate along the boundary 
   main.iFlux.fRLI = main.basis.faceIntegrateGlob(main,main.iFlux.fRLS*main.J_edge_det[0][None,:,:,None,:,:,:,None],MZ.w1,MZ.w2,MZ.w3,MZ.weights1,MZ.weights2,MZ.weights3)
   main.iFlux.fUDI = main.basis.faceIntegrateGlob(main,main.iFlux.fUDS*main.J_edge_det[1][None,:,:,None,:,:,:,None],MZ.w0,MZ.w2,MZ.w3,MZ.weights0,MZ.weights2,MZ.weights3)
   main.iFlux.fFBI = main.basis.faceIntegrateGlob(main,main.iFlux.fFBS*main.J_edge_det[2][None,:,:,None,:,:,:,None],MZ.w0,MZ.w1,MZ.w3,MZ.weights0,MZ.weights1,MZ.weights3)
#   # now add inviscid flux contribution to the RHS
   PLQLu2 = np.zeros(np.shape(main.RHS))

   PLQLu2[:] =  -main.iFlux.fRLI[:,None,:,:,:,1::] 
   PLQLu2[:] += main.iFlux.fRLI[:,None,:,:,:,0:-1]*main.altarray0[None,:,None,None,None,None,None,None,None]
   PLQLu2[:] -= main.iFlux.fUDI[:,:,None,:,:,:,1::]
   PLQLu2[:] += main.iFlux.fUDI[:,:,None,:,:,:,0:-1]*main.altarray1[None,None,:,None,None,None,None,None,None]
   PLQLu2[:] -= main.iFlux.fFBI[:,:,:,None,:,:,:,1::] 
   PLQLu2[:] += main.iFlux.fFBI[:,:,:,None,:,:,:,0:-1]*main.altarray2[None,None,None,:,None,None,None,None,None]

#   main.RHS[:] = 0.
   u0 = main.a.u*1.
#   main.a.u[:] = u0 - eps*R_orthogonal
#   eqns.evalFluxXYZ(main,main.a.u,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,None)
#   main.basis.applyVolIntegral(main,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,main.RHS)
#   R1  = main.RHS*1.
#   #
#   main.RHS[:] = 0.
#   main.a.u[:] = u0[:]
#   eqns.evalFluxXYZ(main,main.a.u,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,None)
#   main.basis.applyVolIntegral(main,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,main.RHS)
#   R2  = main.RHS*1.
#   #
#   #
#   main.RHS[:] = 0.
#   PLQLu = (R1 - R2)/eps

   main.a.u[:] = u0[:]
   evalFluxXYZEulerLin(main,main.a.u,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,[-R_orthogonal])
   main.basis.applyVolIntegral(main,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,PLQLu2)
  
   #print(np.amax(PLQLu2 - PLQLu) )#,np.linalg.norm(PLQLu[1]),np.linalg.norm(PLQLu2[1]))
   #print(np.linalg.norm(PLQLu),np.linalg.norm(f),np.linalg.norm(f_orthogonal),np.linalg.norm(R1),np.linalg.norm(R2 - R1))
   tau = 0.001 #tau8.0
   main.RHS[:] = R0[:] + tau*PLQLu2


def projection_pod(u,V,main):
  tmp = globalDot(V.transpose(),u.flatten(),main)
  u_proj = np.dot(V, tmp.flatten())
  #u_proj = np.reshape( np.dot(Pi,u.flatten() ), np.shape(u)) 
  return u_proj

def test_projection_pod(u,V,main):
  nk,nbasis = np.shape(V)
  V2 = np.zeros(np.shape(V))
  V2[:,0:nbasis/2] = V[:,0:nbasis/2]
  tmp = globalDot(V2.transpose(),u.flatten(),main)
  u_proj = np.dot(V2, tmp.flatten())
  return u_proj

 
def LSPG_POD(main,MZ,eqns):
  eps = 1e-5
  a0 = main.a.a*1.
  eqns.getRHS(main,MZ,eqns)
  #==================================================
  #R_proj = projection_pod(main.RHS,main.V,main)
  R_ortho = main.RHS
  ##print(np.linalg.norm(R_ortho))
  RHS0 = main.RHS*1.
  main.a.a[:] = a0[:] + eps*R_ortho
  eqns.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
  #main.basis.applyMassMatrix(main,main.RHS)
  PLQLu = (main.RHS - RHS0)/eps
  main.PLQLu[:] = PLQLu
  #tau = main.dt
  tau = main.tau
  #print(np.linalg.norm(PLQLu))
  #=====================================
  main.RHS[:] =  RHS0[:]+ tau*PLQLu


def orthogonalSubscale_POD(main,MZ,eqns):
#  if (main.iteration%50 == 0):
#    J = computeJacobian_full_pod(main,eqns)
#    e,s = np.linalg.eig(J)
#    tau = 0.45/np.amax(abs(e))
#    main.tau = tau
#    print('Spectral Radius = ',np.amax(abs(e)))
#    print('tau             = ',np.amax(abs(tau)))

  eps = 1e-5
  a0 = main.a.a*1.
  eqns.getRHS(main,MZ,eqns)
  RHS0 = main.RHS*1.

  R1 = np.zeros(np.shape(main.RHS))
  R1[:] = main.RHS[:]
  main.basis.applyMassMatrix(main,R1)

  #==================================================
  R_proj = projection_pod(R1,main.V,main)
  R_ortho = R1 - np.reshape(R_proj,np.shape(main.a.a))

  ##print(np.linalg.norm(R_ortho))
  main.a.a[:] = a0[:] + eps*R_ortho
  eqns.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
  #main.basis.applyMassMatrix(main,main.RHS)
  PLQLu = (main.RHS - RHS0)/eps
  main.PLQLu[:] = PLQLu
  tau = main.tau
  #=====================================
  main.RHS[:] =  RHS0[:]+ tau*PLQLu
  main.a.a[:] = a0[:]


def orthogonalSubscale_POD2(main,MZ,eqns):
#  if (main.iteration%50 == 0):
#    J = computeJacobian_full_pod(main,eqns)
#    e,s = np.linalg.eig(J)
#    tau = 0.45/np.amax(abs(e))
#    main.tau = tau
#    print('Spectral Radius = ',np.amax(abs(e)))
#    print('tau             = ',np.amax(abs(tau)))

  eps = 1e-5
  a0 = main.a.a*1.
  eqns.getRHS(main,MZ,eqns)
  #==================================================
  R_proj = projection_pod(main.RHS,main.V,main)
  R_ortho = main.RHS - np.reshape(R_proj,np.shape(main.a.a))
  ##print(np.linalg.norm(R_ortho))
  RHS0 = main.RHS*1.
  main.a.a[:] = a0[:] + eps*R_ortho
  eqns.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
  #main.basis.applyMassMatrix(main,main.RHS)
  PLQLu = (main.RHS - RHS0)/eps
#  sz = np.shape(main.V)[1]
#  z = np.random.normal(loc=0.0,scale=1.0,size=sz)
#  z /= np.sqrt(np.sum(z**2))
#  main.a.a[:] = a0[:] + np.reshape( eps*np.dot(main.V,z) , np.shape(main.a.a))
#  eqns.getRHS(main,MZ,eqns)
#  approx_norm = np.linalg.norm(np.dot(main.V.transpose(),1./eps*(main.RHS - RHS0).flatten()))
#  z = np.reshape( np.random.normal(loc=0.0,scale=1.0,size=np.size(main.a.a.flatten()) ), np.shape(main.a.a))
#  z /= np.sqrt(np.sum(z**2))
#  main.a.a[:] = a0[:] + eps*z
#  eqns.getRHS(main,MZ,eqns)
#  approx_norm = np.linalg.norm(1./eps*(main.RHS - RHS0))
#  approx_norm2 = np.linalg.norm(PLQLu)
#  print(approx_norm)#,approx_norm2)
  main.PLQLu[:] = PLQLu
  tau = main.tau
  #tau = 1./approx_norm*0.05
  #print(np.linalg.norm(PLQLu))
  #=====================================
  main.RHS[:] =  RHS0[:]+ tau*PLQLu
  main.a.a[:] = a0[:]

def orthogonalSubscale_POD_dtau(main,MZ,eqns):
  eps = 1e-5
  a0 = main.a.a*1.
  eqns.getRHS(main,MZ,eqns)
  #==================================================
  R_proj = projection_pod(main.RHS,main.V,main)
  R_ortho = main.RHS - np.reshape(R_proj,np.shape(main.a.a))
  RHS0 = main.RHS*1.

  main.a.a[:] = a0[:] + eps*R_ortho
  eqns.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
  RHS1 = np.zeros(np.shape(main.RHS))
  RHS1[:] = main.RHS[:]

  #main.basis.applyMassMatrix(main,main.RHS)
  PLQLu = (RHS1 - RHS0)/eps
  main.PLQLu[:] = PLQLu

  # now compute tau
  a_projf = test_projection_pod(main.a.a,main.V,main)

  main.a.a[:] = a_projf[:] 
  eqns.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
  RHS0f = np.zeros(np.shape(main.RHS))
  RHS0f[:] = main.RHS[:]

  R_projf = test_projection_pod(RHS1f,main.V,main)
  R_orthof = RHS0f[:] - np.reshape(R_projf,np.shape(main.a.a))

  main.a.a[:] = a_projf[:] + eps*R_orthof
  eqns.getRHS(main,MZ,eqns)  ## put RHS in a array since we don't need it
  RHS1f = np.zeros(np.shape(main.RHS))
  RHS1f[:] = main.RHS[:]

  tau = main.tau
  #print(np.linalg.norm(PLQLu))
  #=====================================
  main.RHS[:] =  RHS0[:]+ tau*PLQLu



def orthogonalSubscaleEntropyb(main,MZ,eqns):
   eqns.getRHS(main,MZ,eqns)
   R0 = np.zeros(np.shape(main.RHS))
   R1 = np.zeros(np.shape(main.RHS))
   R0[:] = main.RHS[:]
   PLQLu2 = np.zeros(np.shape(main.RHS))
   main.RHS[:] = 0.
   R= strongFormEulerXYZEntropy(main,main.a.a,None)
   u0 = main.a.u*1.
   R_orthogonal= orthogonalProjectionVol(main,R)
   main.a.u[:] = u0[:]
   evalFluxXYZEulerLinEntropy(main,main.a.u,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,[-R_orthogonal])
   main.basis.applyVolIntegral(main,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,PLQLu2)
#   plot(abs(PLQLu2[1,1,0,0,0,:,0,0,0]) / abs( R0[1,1,0,0,0,:,0,0,0]) )
#   plot(abs(PLQLu2[1,2,0,0,0,:,0,0,0]) / abs( R0[1,2,0,0,0,:,0,0,0]) )
#   plot(abs(PLQLu2[1,3,0,0,0,:,0,0,0]) / abs( R0[1,3,0,0,0,:,0,0,0]) )
#   yscale('log')
#   pause(0.001)
#   clf()
   main.RHS[:] = R0[:] #+ tau*PLQLu2
   indx0 = abs(PLQLu2[0,1,0,0,0,:,0,0,0]) >  abs(R0[0,1,0,0,0,:,0,0,0])
   indx1 = abs(PLQLu2[1,1,0,0,0,:,0,0,0]) >  abs(R0[1,1,0,0,0,:,0,0,0])
   indx2 = abs(PLQLu2[2,1,0,0,0,:,0,0,0]) >  abs(R0[2,1,0,0,0,:,0,0,0])
   indx3 = abs(PLQLu2[3,1,0,0,0,:,0,0,0]) >  abs(R0[3,1,0,0,0,:,0,0,0])
   indx4 = abs(PLQLu2[4,1,0,0,0,:,0,0,0]) >  abs(R0[4,1,0,0,0,:,0,0,0])
   #print(indx4)
   main.a.a[0,1,0,0,0,indx4,0,0,0] = 0.
   main.RHS[0,1,0,0,0,indx4,0,0,0] = 0.
   main.a.a[1,1,0,0,0,indx4,0,0,0] = 0.
   main.RHS[1,1,0,0,0,indx4,0,0,0] = 0.
   main.a.a[2,1,0,0,0,indx4,0,0,0] = 0.
   main.RHS[2,1,0,0,0,indx4,0,0,0] = 0.
   main.a.a[3,1,0,0,0,indx4,0,0,0] = 0.
   main.RHS[3,1,0,0,0,indx4,0,0,0] = 0.
   main.a.a[4,1,0,0,0,indx4,0,0,0] = 0.
   main.RHS[4,1,0,0,0,indx4,0,0,0] = 0.

def projectionEntropy(main,U):
  ## First perform integration in x
  f = U*main.Jdet[None,:,:,:,None,:,:,:,None]
  a_project =  main.basis.volIntegrateGlob(main,f,main.w0,main.w1,main.w2,main.w3)
  a_project = np.reshape(a_project,(main.nvars*main.order[0]*main.order[1]*main.order[2]*main.order[3],main.Npx,main.Npy,main.Npz,main.Npt) )
  a_project = np.einsum('ij...,j...->i...',main.EMM,a_project)
  a_project = np.reshape(a_project,np.shape(main.a.a))

  V_project = main.basis.reconstructUGeneral(main,a_project)
  return V_project

def dynamicTauEntropy(main,MZ,eqns):
   a0 = main.a.a*1.
   filtarray = np.zeros(np.shape(main.a.a))
   filtarray[:,0:main.order[0]/2,0:main.order[1]/2,0:main.order[2]/2] = 1.
   af = main.a.a*filtarray
   vf = main.basis.reconstructUGeneral(main,af)

   ## get standard RHS
   eqns.getRHS(main,MZ,eqns)
   u0 = main.a.u*1.
   R0 = np.zeros(np.shape(main.RHS))
   R0[:] = main.RHS[:]

   ## now compute standard PLQLu
   PLQLu = np.zeros(np.shape(main.RHS))
   main.RHS[:] = 0.
   R= strongFormEulerXYZEntropy(main,main.a.a,None)
   R_orthogonal = orthogonalProjection(main,R)
   main.a.u[:] = u0[:]
   evalFluxXYZEulerLinEntropy(main,main.a.u,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,[-R_orthogonal])
   main.basis.applyVolIntegral(main,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,PLQLu)
   ##

   if (main.linear_iteration == 0 and main.NLiter == 0):
     R0real = main.basis.reconstructUGeneral(main,R0*filtarray)
     PLQLuU = main.basis.reconstructUGeneral(main,PLQLu*filtarray)
     ## now get filtered RHS
     main.a.a[:] = af[:]
     main.RHS[:] = 0.
     eqns.getRHS(main,main,eqns)
     R0f = np.zeros(np.shape(main.RHS))
     R0f[:] = main.RHS[:]
     R0freal = main.basis.reconstructUGeneral(main,R0f)
  
  
     ## now compute filtered PLQLu
     PLQLuf = np.zeros(np.shape(main.RHS))
     main.RHS[:] = 0.
     R= strongFormEulerXYZEntropy(main,af,None)
     R_orthogonal = R - testProjection(main,R)
     evalFluxXYZEulerLinEntropy(main,vf,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,[-R_orthogonal])
     main.basis.applyVolIntegral(main,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,PLQLuf)
     PLQLufU = main.basis.reconstructUGeneral(main,PLQLuf)
  
     #entropy transfer
     num = globalSum( np.sum( np.sum( vf*(R0freal - R0real) , axis=0) ) ,main)
     den = globalSum( np.sum( np.sum( vf*(PLQLuU - 2.*PLQLufU) , axis=0) ) ,main)
     tau = num/den
     if (main.mpi_rank == 0):
       print('Computing tau, tau = ' + str(tau) ) 
     main.tau = tau 
#   U = entropy_to_conservative(main.a.u)
#   rhoi = 1./U[0]
#   h = main.dx/main.order[0]
#   tau = np.mean(  (4./h**2*rhoi**2*(U[1]**2 + U[2]**2 + U[3]**2) + 3.*np.pi*main.mus**2*(4./h**2)**2 )**-0.5 )
#   main.tau = tau
   main.RHS[:] = R0[:] + main.tau*PLQLu
   main.a.a = a0
   main.a.u = u0

def orthogonalSubscaleEntropy(main,MZ,eqns):
   eqns.getRHS(main,MZ,eqns)
   u0 = main.a.u*1.

   R0 = np.zeros(np.shape(main.RHS))
   R1 = np.zeros(np.shape(main.RHS))
   R0[:] = main.RHS[:]
   PLQLu2 = np.zeros(np.shape(main.RHS))
   main.RHS[:] = 0.
   R= strongFormEulerXYZEntropy(main,main.a.a,None)

   #dSdt = np.sum( np.sum(main.a.u*R,axis=0) )
   #print(dSdt)
   R_orthogonal = orthogonalProjection(main,R)

   main.a.u[:] = u0[:]
   evalFluxXYZEulerLinEntropy(main,main.a.u,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,[-R_orthogonal])
   main.basis.applyVolIntegral(main,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,PLQLu2)
   U = entropy_to_conservative(main.a.u)
   rhoi = 1./U[0]
   h = main.dx/main.order[0]
   tau = np.mean(  (4./h**2*rhoi**2*(U[1]**2 + U[2]**2 + U[3]**2) + 3.*np.pi*main.mus**2*(4./h**2)**2 )**-0.5 )
   main.tau = tau
   main.RHS[:] = R0[:] + main.tau*PLQLu2


def orthogonalSubscaleEntropyChain(main,MZ,eqns):
   eqns.getRHS(main,MZ,eqns)
   R0 = np.zeros(np.shape(main.RHS))
   R1 = np.zeros(np.shape(main.RHS))
   R0[:] = main.RHS[:]
   PLQLu2 = np.zeros(np.shape(main.RHS))
   main.RHS[:] = 0.
   R= strongFormEulerXYZEntropy(main,main.a.a,None)
   dudv = mydUdV(main.a.u)
   dvdu = np.rollaxis( np.rollaxis(dudv,1,10),0,9)
   dvdu = np.linalg.inv(dvdu)
   dvdu = np.rollaxis( np.rollaxis(dvdu,8,0), 9,1)
   u0 = main.a.u*1.

   R_project = projectionEntropy(main,R)
   R = np.einsum('ij...,j...->i...',dvdu,R)
#   print(np.linalg.norm(R_project),np.linalg.norm(R))
#   plot(R[0,0,0,0,0,:,0,0,0])
#   plot(R_project[0,0,0,0,0,:,0,0,0])
#   pause(0.01)
#   clf()
   R_orthogonal = R - R_project
   main.a.u[:] = u0[:]
   evalFluxXYZEulerLinEntropy(main,main.a.u,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,[-R_orthogonal,dudv])
   main.basis.applyVolIntegral(main,main.iFlux.fx,main.iFlux.fy,main.iFlux.fz,PLQLu2)
   U = entropy_to_conservative(main.a.u)
   rhoi = 1./U[0]
   h = main.dx/main.order[0]
   tau = np.mean(  (4./h**2*rhoi**2*(U[1]**2 + U[2]**2 + U[3]**2) + 3.*np.pi*main.mus**2*(4./h**2)**2 )**-0.5 )
   main.tau = tau
   main.RHS[:] = R0[:] + main.tau*PLQLu2


## Evaluate the tau model through the FD approximation. This is expensive AF
def tauModelFDEntropy(main,MZ,eqns):
    ### EVAL RESIDUAL AND DO MZ STUFF
    filtarray = np.zeros(np.shape(MZ.a.a))
    filtarray[:,0:main.order[0],0:main.order[1],0:main.order[2]] = 1.
    eps = 1.e-5
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2],:] = main.a.a[:]
    #eqns.getRHS(main,main,eqns)
    eqns.getRHS(MZ,MZ,eqns)
    RHS1 = np.zeros(np.shape(MZ.RHS))
    Rtmp = np.zeros(np.shape(MZ.RHS))
    RHS1[:] =MZ.RHS[:]
    #Rtmp[:] = RHS1[:]
    #MZ.basis.applyMassMatrix(MZ,Rtmp)
    #MZ.a.a[:] = 0.
    #MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    #MZ.a.a[:] = MZ.a.a[:] + eps*Rtmp[:]
    #eqns.getRHS(MZ,MZ,eqns)
    #RHS2 = np.zeros(np.shape(MZ.RHS))
    #RHS2[:] = MZ.RHS[:]
    #MZ.a.a[:] = 0.
    #MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    #MZ.a.a[:] = MZ.a.a[:] + eps*Rtmp[:]*filtarray
    #eqns.getRHS(MZ,MZ,eqns)
    #RHS3 = np.zeros(np.shape(MZ.RHS))
    #RHS3[:] = MZ.RHS[:]
    #PLQLU = (RHS2[:,0:main.order[0],0:main.order[1],0:main.order[2]] - RHS3[:,0:main.order[0],0:main.order[1],0:main.order[2]])/(eps + 1e-30)
    main.RHS[:] =  RHS1[:,0:main.order[0],0:main.order[1],0:main.order[2]] #+ main.dx/MZ.order[0]**2*PLQLU



## Evaluate the tau model through the FD approximation. This is expensive AF
def tauModelFD(main,MZ,eqns):
    ### EVAL RESIDUAL AND DO MZ STUFF
    filtarray = np.zeros(np.shape(MZ.a.a))
    filtarray[:,0:main.order[0],0:main.order[1],0:main.order[2]] = 1.
    eps = 1.e-5
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2],:] = main.a.a[:]
    #eqns.getRHS(main,main,eqns)
    eqns.getRHS(MZ,MZ,eqns)
    RHS1 = np.zeros(np.shape(MZ.RHS))
    Rtmp = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    Rtmp[:] = RHS1[:]
    MZ.basis.applyMassMatrix(MZ,Rtmp)
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*(Rtmp[:] - Rtmp[:]*filtarray)
    eqns.getRHS(MZ,MZ,eqns)
    RHS2 = np.zeros(np.shape(MZ.RHS))
    RHS2[:] = MZ.RHS[:]
    #MZ.a.a[:] = 0.
    #MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    #MZ.a.a[:] = MZ.a.a[:] + eps*Rtmp[:]*filtarray
    #eqns.getRHS(MZ,MZ,eqns)
    #RHS3 = np.zeros(np.shape(MZ.RHS))
    #RHS3[:] = MZ.RHS[:]
    PLQLU = (RHS2[:,0:main.order[0],0:main.order[1],0:main.order[2]] - RHS1[:,0:main.order[0],0:main.order[1],0:main.order[2]])/(eps + 1e-30)
    tau = main.tau#main.dx/MZ.order[0]**2

    #U = main.a.u 
    #rhoi = 1./U[0]
    #h = main.dx/main.order[0]
    #tau = np.mean(  (4./h**2*rhoi**2*(U[1]**2 + U[2]**2 + U[3]**2) + 3.*np.pi*main.mus**2*(4./h**2)**2 )**-0.5 )
    #main.tau = tau
    #print(main.tau)

    main.RHS[:] =  RHS1[:,0:main.order[0],0:main.order[1],0:main.order[2]] + 1.*tau*PLQLU


def FM1Linearized(main,MZ,eqns):
   filtarray = np.zeros(np.shape(MZ.a.a))
   filtarray[:,0:main.order[0],0:main.order[1],0:main.order[2]] = 1.
   MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
   MZ.a.a[0:5,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[0:5]
   eqns.getRHS(MZ,MZ,eqns) #compute the residual in an enriched space
   tau = 0.1
   RHS1 = np.zeros(np.shape(MZ.RHS))
   RHS1f = np.zeros(np.shape(MZ.RHS))
   RHS1[:] = MZ.RHS[:]
   RHS1f[:] = RHS1[:] - RHS1[:]*filtarray
   eqnsLin = equations('Linearized Navier-Stokes',('central','Inviscid') )
   # now we need to compute the linearized RHS
   MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
   MZ.a.a[0:5,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[0:5]
   eqnsLin.getRHS(MZ,main,eqnsLin,[RHS1f]) ## this is PLQLu
   PLQLU = MZ.RHS[:,0:main.order[0],0:main.order[1],0:main.order[2]]
   main.RHS[0:5] =  RHS1[0:5,0:main.order[0],0:main.order[1],0:main.order[2]] + main.a.a[5::] 
   main.RHS[5::] = -2.*main.a.a[5::]/tau + 2.*PLQLU
   main.comm.Barrier()




def tauModelLinearized(main,MZ,eqns):
   filtarray = np.zeros(np.shape(MZ.a.a))
   filtarray[:,0:main.order[0],0:main.order[1],0:main.order[2],:,:,:] = 1.
   MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
   MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
   eqns.getRHS(MZ,MZ,eqns) #compute the residual in an enriched space
   RHS1 = np.zeros(np.shape(MZ.RHS))
   RHS1f = np.zeros(np.shape(MZ.RHS))
   RHS1[:] = MZ.RHS[:]
   RHS1f[:] = RHS1[:] - RHS1[:]*filtarray
   RHS1f_phys = main.basis.reconstructUGeneral(MZ,RHS1f)
   eqnsLin = equations('Linearized Navier-Stokes',('central','Inviscid'),'DNS' )
   # now we need to compute the linearized RHS
   MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
   MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
   eqnsLin.getRHS(MZ,MZ,eqnsLin,[RHS1f],[RHS1f_phys]) ## this is PLQLu
   PLQLU = MZ.RHS[:,0:main.order[0],0:main.order[1],0:main.order[2]]
   main.RHS[:] =  RHS1[:,0:main.order[0],0:main.order[1],0:main.order[2]] + main.tau*PLQLU
   main.comm.Barrier()




def validateLinearized(main,MZ,eqns):
    # validation of the linearized equations
    # should have R(a + eps f) - R(a) / eps = Rlin(a)
    filtarray = np.zeros(np.shape(MZ.a.a))
    filtarray[:,0:main.order[0],0:main.order[1],0:main.order[2],:,:,:] = 1.
    MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    eqnsLin = equations('Linearized Navier-Stokes',('central','Inviscid') )
    eqnsLin.getRHS(MZ,MZ,eqns) #compute the residual in an enriched space
    RHSLin = np.zeros(np.shape(MZ.RHS))
    RHSLin[:] = MZ.RHS[:]
    # now we need to compute the linearized RHS via finite difference
    eps = 1.e-6
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    eqnsLin.getRHS(MZ,eqns,eqns)
    RHS1 = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*MZ.a.a[:]
    eqns.getRHS(MZ,eqns,eqns)
    RHS2 = np.zeros(np.shape(MZ.RHS))
    RHS2[:] = MZ.RHS[:]
    RHSLinFD = (RHS2 - RHS1)/eps
    print(np.linalg.norm(RHSLin),np.linalg.norm(RHSLinFD) )
    print(np.linalg.norm(RHSLin[:] -  RHSLinFD[:]) )
    main.RHS[:] =  RHS1[:,0:main.order[0],0:main.order[1],0:main.order[2]] #+ 0.0001*MZ.RHS[:,0:main.order[0],0:main.order[1],0:main.order[2]]


def tauModelValidateLinearized(main,MZ,eqns):
    filtarray = np.zeros(np.shape(MZ.a.a))
    filtarray[:,0:main.order[0],0:main.order[1],0:main.order[2],:,:,:] = 1.
    MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    eqns.getRHS(MZ,MZ,eqns) #compute the residual in an enriched space
    RHS1 = np.zeros(np.shape(MZ.RHS))
    RHS1f = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    RHS1f[:] = RHS1[:] - RHS1[:]*filtarray
    Z = reconstructUGeneral(main,main.a.a)
    eqnsLin = equations('Linearized Navier-Stokes',('central','Inviscid') )
    # now we need to compute the linearized RHS
    MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    eqnsLin.getRHS(MZ,main,eqnsLin,[RHS1f]) ## this is PLQLu
    PLQLULin = np.zeros(np.shape(MZ.RHS[:,0:main.order[0],0:main.order[1],0:main.order[2]]))
    PLQLULin[:] = MZ.RHS[:,0:main.order[0],0:main.order[1],0:main.order[2]]

#    print(np.linalg.norm(PLQLU[0]))

    eps = 1.e-5
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    eqns.getRHS(MZ,MZ,eqns)
    RHS1 = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*RHS1[:]
    eqns.getRHS(MZ,MZ,eqns)
    RHS2 = np.zeros(np.shape(MZ.RHS))
    RHS2[:] = MZ.RHS[:]
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*RHS1[:]*filtarray
    eqns.getRHS(MZ,MZ,eqns)
    RHS3 = np.zeros(np.shape(MZ.RHS))
    RHS3[:] = MZ.RHS[:]
    PLQLUFD = (RHS2[:,0:main.order[0],0:main.order[1],0:main.order[2]] - RHS3[:,0:main.order[0],0:main.order[1],0:main.order[2]])/eps

    print(np.linalg.norm(PLQLULin),np.linalg.norm(PLQLUFD) )
    print(np.linalg.norm(PLQLULin[4] - PLQLUFD[4] ))
    main.RHS[:] =  RHS1[:,0:main.order[0],0:main.order[1],0:main.order[2]] #+ 0.0001*MZ.RHS[:,0:main.order[0],0:main.order[1],0:main.order[2]]

    main.comm.Barrier()



def DtauModel(main,MZ,eqns,schemes):

  def sendScalar(scalar,main):
    if (main.mpi_rank == 0):
      for i in range(1,main.num_processes):
        loc_rank = i
        main.comm.Send(np.ones(1)*scalar,dest=loc_rank,tag=loc_rank)
      return scalar
    else:
      test = np.ones(1)*scalar
      main.comm.Recv(test,source=0,tag=main.mpi_rank)
      return test[0]

  ### EVAL RESIDUAL AND DO MZ STUFF
  filtarray = np.zeros(np.shape(MZ.a.a))
  filtarray[:,0:main.order,0:main.order,:,:] = 1.
  eps = 1.e-5
  MZ.a.a[:] = 0.
  MZ.a.a[:,0:main.order,0:main.order] = main.a.a[:]
  MZ.getRHS(MZ,eqns,schemes)
  RHS1 = np.zeros(np.shape(MZ.RHS))
  RHS1[:] = MZ.RHS[:]
  MZ.a.a[:] = 0.
  MZ.a.a[:,0:main.order,0:main.order] = main.a.a[:]
  MZ.a.a[:] = MZ.a.a[:] + eps*RHS1[:]
  MZ.getRHS(MZ,eqns,schemes)
  RHS2 = np.zeros(np.shape(MZ.RHS))
  RHS2[:] = MZ.RHS[:]
  MZ.a.a[:] = 0.
  MZ.a.a[:,0:main.order,0:main.order] = main.a.a[:]
  MZ.a.a[:] = MZ.a.a[:] + eps*RHS1[:]*filtarray
  MZ.getRHS(MZ,eqns,schemes)
  RHS3 = np.zeros(np.shape(MZ.RHS))
  RHS3[:] = MZ.RHS[:]
  PLQLU = (RHS2[:,0:main.order,0:main.order] - RHS3[:,0:main.order,0:main.order])/eps

  if (main.rkstage == 0):
    ### Now do dynamic procedure to get tau
    filtarray2 = np.zeros(np.shape(MZ.a.a))
    filtarray2[:,0:MZ.forder,0:MZ.forder,:,:] = 1.
    eps = 1.e-5
    ## Get RHS
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:MZ.forder,0:MZ.forder] = main.a.a[:,0:MZ.forder,0:MZ.forder]
    MZ.getRHS(MZ,eqns,schemes)
    RHS4 = np.zeros(np.shape(MZ.RHS))
    RHS4[:] = MZ.RHS[:]
    ## Now get RHS(a + eps*RHS)
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:MZ.forder,0:MZ.forder] = main.a.a[:,0:MZ.forder,0:MZ.forder]
    MZ.a.a[:] = MZ.a.a[:]*filtarray2 + eps*RHS4[:]
    MZ.getRHS(MZ,eqns,schemes)
    RHS5 = np.zeros(np.shape(MZ.RHS))
    RHS5[:] = MZ.RHS[:]
    ## Now get RHS(a + eps*RHSf)
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:MZ.forder,0:MZ.forder] = main.a.a[:,0:MZ.forder,0:MZ.forder]
    MZ.a.a[:] = MZ.a.a[:]*filtarray2 + eps*RHS4[:]*filtarray2
    MZ.getRHS(MZ,eqns,schemes)
    RHS6 = np.zeros(np.shape(MZ.RHS))
    RHS6[:] = MZ.RHS[:]
  
    ## Now compute PLQLUf
    PLQLUf = (RHS5[:,0:main.order,0:main.order] - RHS6[:,0:main.order,0:main.order])/eps
  
    PLQLUG = gatherSolSpectral(PLQLU,main)
    MZ.PLQLUG = PLQLUG
    PLQLUfG = gatherSolSpectral(PLQLUf[:,0:main.order,0:main.order],main)
    RHS1G = gatherSolSpectral(RHS1[:,0:main.order,0:main.order],main)
    RHS4G = gatherSolSpectral(RHS4[:,0:main.order,0:main.order],main)

    afG = gatherSolSpectral(main.a.a[:,0:main.order,0:main.order],main)

    if (main.mpi_rank == 0):
      num = 2.*np.mean(np.sum(afG[1:3,0:MZ.forder,0:MZ.forder]*(RHS4G[1:3,0:MZ.forder,0:MZ.forder] - RHS1G[1:3,0:MZ.forder,0:MZ.forder]),axis=(0,1,2)) ,axis=(0,1))
      den =  np.mean ( np.sum(afG[1:3,0:MZ.forder,0:MZ.forder]*(PLQLUG[1:3,0:MZ.forder,0:MZ.forder] - \
                                         (main.order/MZ.forder)*PLQLUfG[1:3,0:MZ.forder,0:MZ.forder]),axis=(0,1,2)) ,axis=(0,1))
      tau = num/(den + 1.e-1)
      print(tau)
    else:
      tau = 0.
    MZ.tau = np.maximum(0.,sendScalar(tau,main))
    #MZ.tau = sendScalar(tau,main)
  return 0.0002*PLQLU



def tauModel2(main,MZ,eqns,schemes):
    ### EVAL RESIDUAL AND DO MZ STUFF
    filtarray = np.ones(np.shape(main.a.a))
    filtarray[:,main.rorder::,main.rorder::,:,:] = 0.
    eps = 1.e-5
    MZ.a.a[:] = main.a.a[:]*filtarray
    main.getRHS(MZ,eqns,schemes)
    RHS1 = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    MZ.a.a[:] = main.a.a[:]*filtarray + eps*RHS1[:]
    main.getRHS(MZ,eqns,schemes)
    RHS2 = np.zeros(np.shape(MZ.RHS))
    RHS2[:] = MZ.RHS[:]
    MZ.a.a[:] = main.a.a[:] + eps*RHS1[:]*filtarray
    main.getRHS(MZ,eqns,schemes)
    RHS3 = np.zeros(np.shape(MZ.RHS))
    RHS3[:] = MZ.RHS[:]
    PLQLU = (RHS2 - RHS3)/eps
    return MZ.tau*PLQLU



def shockCapturingSetViscosity(main):
  ### Shock capturing
  filta = np.zeros(np.shape(main.a.a))
  filta[:,0:main.order[0]-1,main.order[1]-1,main.order[2]-1] = 1.
  af = main.a.a*filta    #make filtered state
  uf = reconstructUGeneral(main,af)
  udff = (main.a.u - uf)**2
  # now compute switch
  Se_num = volIntegrate(main.weights0,main.weights1,main.weights2,udff) 
  Se_den = volIntegrate(main.weights0,main.weights1,main.weights2,main.a.u**2)
  Se = (Se_num + 1e-10)/(Se_den + 1.e-30)
  eps0 = 1.*main.dx/main.order[0]
  s0 =1./main.order[0]**4
  kap = 5.
  se = np.log10(Se)
  #print(np.amax(udff))
  epse = eps0/2.*(1. + np.sin(np.pi/(2.*kap)*(se - s0) ) )
  epse[se<s0-kap] = 0.
  epse[se>s0  + kap] = eps0
  #plt.clf()
  #print(np.amax(epse),np.amin(epse))
  #plt.plot(epse[0,:,0,0])
  #plt.ylim([1e-9,0.005])
  #plt.pause(0.001)
  #print(np.shape(main.mu),np.shape(epse) )
  main.mu = main.mu0 + epse[0]
  main.muR = main.mu0R + epse[0]
  main.muL = main.mu0L + epse[0]
  main.muU = main.mu0F + epse[0]
  main.muD = main.mu0D + epse[0]
  main.muF = main.mu0U + epse[0]
  main.muB = main.mu0B + epse[0]





