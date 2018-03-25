import numpy as np
import numpy.linalg
from DG_functions import getRHS_element
from navier_stokes_entropy import *
from navier_stokes import *

def getEntropyJacobianFD(main,eqns):
  #=================
  norder = main.order[0]*main.order[1]*main.order[2]*main.order[3]
  main.J2 = np.zeros(np.shape(main.EMM))
  main.J2[:] = 0.
  a0 = main.a.a*1.
  getRHS_element(main,main,eqns)
  RHS0 = main.RHS*1.
  t0 = time.time()
  counter = 0
  eps = 1.e-7
  for z in range(0,5):
    for p in range(0,main.order[0]):
      for q in range(0,main.order[1]):
        for r in range(0,main.order[2]): 
          for s in range(0,main.order[3]):
            main.a.a[:] = a0[:]
            main.a.a[z,p,q,r,s] += eps
            getRHS_element(main,main,eqns)
            R1 = main.RHS*1.
            main.a.a[z,p,q,r,s] -= 2.*eps
            getRHS_element(main,main,eqns)
            R2 = main.RHS*1.

            main.J2[:,counter] = np.reshape( (R1 - R2)/eps , (5*norder,main.Npx,main.Npy,main.Npz,main.Npt) )
            counter += 1 
#  t1 = time.time()
  return main.J2




def getEntropyJacobian(main,eqns):
  def getInnerMassMatrix(main,g,w0,w1,w2,w3):
    norder = main.order[0]*main.order[1]*main.order[2]*main.order[3]
    M2 = np.zeros((norder,norder,\
                  main.Npx,main.Npy,main.Npz,1 ) )
    f2 = g*main.Jdet[None,:,:,:,None,:,:,:,None] 
    M2[:] = np.reshape( volIntegrateGlob_einsumMM2(main,f2,w0,w1,w2,w3) ,np.shape(M2))
    return M2
  def getInnerMassMatrixX(main,g,w0,w1,w2,w3,w0b,w1b,w2b,w3b):
    norder = main.order[0]*main.order[1]*main.order[2]*main.order[3]
    M2 = np.zeros((norder,norder,\
                  main.Npx,main.Npy,main.Npz,1 ) )
    f2 = g*main.Jinv[0,0][None,:,:,:,None,:,:,:,None]
    f2 = f2*main.Jdet[None,:,:,:,None,:,:,:,None] 
    M2[:] = np.reshape( volIntegrateGlob_einsumMM2_derivs(main,f2,w0,w1,w2,w3,w0b,w1b,w2b,w3b) ,np.shape(M2))
    return M2
  def getInnerMassMatrixY(main,g,w0,w1,w2,w3,w0b,w1b,w2b,w3b):
    norder = main.order[0]*main.order[1]*main.order[2]*main.order[3]
    M2 = np.zeros((norder,norder,\
                  main.Npx,main.Npy,main.Npz,1 ) )
    f2 = g*main.Jinv[1,1][None,:,:,:,None,:,:,:,None]
    f2 = f2*main.Jdet[None,:,:,:,None,:,:,:,None] 
    M2[:] = np.reshape( volIntegrateGlob_einsumMM2_derivs(main,f2,w0,w1,w2,w3,w0b,w1b,w2b,w3b) ,np.shape(M2))
    return M2
  def getInnerMassMatrixZ(main,g,w0,w1,w2,w3,w0b,w1b,w2b,w3b):
    norder = main.order[0]*main.order[1]*main.order[2]*main.order[3]
    M2 = np.zeros((norder,norder,\
                  main.Npx,main.Npy,main.Npz,1 ) )
    f2 = g*main.Jinv[2,2][None,:,:,:,None,:,:,:,None]
    f2 = f2*main.Jdet[None,:,:,:,None,:,:,:,None] 
    M2[:] = np.reshape( volIntegrateGlob_einsumMM2_derivs(main,f2,w0,w1,w2,w3,w0b,w1b,w2b,w3b) ,np.shape(M2))
    return M2

  #=================
  dudv = mydUdV(main.a.u)
  norder = main.order[0]*main.order[1]*main.order[2]*main.order[3]
  main.J = np.zeros(np.shape(main.EMM))
  main.J[:] = 0.
  count = 0
  I = np.eye(5)
  t0 = time.time()
  for i in range(0,5):
    for j in range(i,5):
      main.J[i*norder:(i+1)*norder,j*norder:(j+1)*norder] = getInnerMassMatrix(main,dudv[i,j],main.w0,main.w1,main.w2,main.w3)
      main.J[j*norder:(j+1)*norder,i*norder:(i+1)*norder] = main.J[i*norder:(i+1)*norder,j*norder:(j+1)*norder]*1.
 
  getEntropyJacobianFD(main,eqns) 
  #print(np.lina.
  main.J -= 0.5*main.dt*main.J2
#
#  U = entropy_to_conservative(main.a.u)
#  dFdU,dGdU,dHdU = evalJacobianXYZEulerLin(main,U)
#  dFdU_dUdV = np.einsum('ij...,jk...->ik...',dFdU,dudv)
#  dGdU_dUdV = np.einsum('ij...,jk...->ik...',dGdU,dudv)
#  dHdU_dUdV = np.einsum('ij...,jk...->ik...',dHdU,dudv)
#
#  for i in range(0,5):
#    for j in range(0,5):
#      main.J[i*norder:(i+1)*norder,j*norder:(j+1)*norder] -= getInnerMassMatrixX(main,dFdU_dUdV[i,j],main.w0,main.w1,main.w2,main.w3,main.wp0,main.w1,main.w2,main.w3)
#      #main.J[j*norder:(j+1)*norder,i*norder:(i+1)*norder] += main.J[i*norder:(i+1)*norder,j*norder:(j+1)*norder]
##
#  for i in range(0,5):
#    for j in range(0,5):
#      main.J[i*norder:(i+1)*norder,j*norder:(j+1)*norder] -= getInnerMassMatrixY(main,dGdU_dUdV[i,j],main.w0,main.w1,main.w2,main.w3,main.w0,main.wp1,main.w2,main.w3)
#      #main.J[j*norder:(j+1)*norder,i*norder:(i+1)*norder] += main.J[i*norder:(i+1)*norder,j*norder:(j+1)*norder]
#
#  for i in range(0,5):
#    for j in range(0,5):
#      main.J[i*norder:(i+1)*norder,j*norder:(j+1)*norder] -= getInnerMassMatrixZ(main,dHdU_dUdV[i,j],main.w0,main.w1,main.w2,main.w3,main.w0,main.w1,main.wp2,main.w3)
#      #main.J[j*norder:(j+1)*norder,i*norder:(i+1)*norder] += main.J[i*norder:(i+1)*norder,j*norder:(j+1)*norder]
#
#  print(np.linalg.norm(main.J))
#
#  t1 = time.time()
  main.J = np.rollaxis( np.rollaxis(main.J,1,6),0,5)
  main.J = np.linalg.inv(main.J)
  main.J = np.rollaxis( np.rollaxis(main.J,4,0),5,1)
  return main.J





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
