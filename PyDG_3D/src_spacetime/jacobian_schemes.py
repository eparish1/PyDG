import numpy as np
import numpy.linalg

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
