import numpy as np
import numpy.linalg
def computeBlockJacobian(main,eqns):
  J = np.zeros((main.nvars,main.order[0],main.order[1],main.order[2],main.order[3],\
               main.order[0],main.order[1],main.order[2],main.order[3],\
               main.Nel[0],main.Nel[1],main.Nel[2],main.Nel[3]))
  eqns.getRHS(main,main,eqns)
  RHS0 = np.zeros(np.shape(main.RHS))
  RHStmp = np.zeros(np.shape(main.RHS))
  RHS0[:] = main.RHS[:]
  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  eps = 1e-5
  for i in range(0,main.order[0]):
    for j in range(0,main.order[1]):
      for k in range(0,main.order[2]):
        for l in range(0,main.order[3]):
          main.a.a[:] = a0[:] + eps 
          eqns.getRHS(main,main,eqns)
          RHStmp[:] = main.RHS[:]
          J[:,:,:,:,:,i,j,k,l] = (RHStmp - RHS0)/eps
  return J

def computeJacobianX(main,eqns,f):
  J = np.zeros((main.nvars,main.order[0],main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  RHS0 = np.zeros(np.shape(main.RHS))
  RHStmp = np.zeros(np.shape(main.RHS))
  Rstar0,R0,Rstar_glob = f(main.a.a)
  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  eps = 1e-5
  for i in range(0,main.order[0]):
    main.a.a[:] = a0[:]
    main.a.a[:,i,:,:,:] = a0[:,i,:,:,:] + eps 
    Rstar,RHStmp,Rstar_glob = f(main.a.a)
    J[:,:,i,:,:,:] = ( (Rstar - Rstar0)/eps )[:,:,:,:,:]
  Jinv = np.rollaxis(np.rollaxis( np.linalg.inv(np.rollaxis(np.rollaxis(J,2,10),1,9)) , 8 , 1) , 9 , 2)
  main.a.a[:] = a0[:]
  return Jinv


def computeJacobianY(main,eqns,f):
  J = np.zeros((main.nvars,main.order[0],main.order[1],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  RHS0 = np.zeros(np.shape(main.RHS))
  RHStmp = np.zeros(np.shape(main.RHS))
  Rstar0,R0,Rstar_glob = f(main.a.a)
  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  eps = 1e-3
  for i in range(0,main.order[1]):
    main.a.a[:] = a0[:]
    main.a.a[:,:,i,:,:] = a0[:,:,i,:,:] + eps 
    Rstar,RHStmp,Rstar_glob = f(main.a.a)
    J[:,:,:,i,:,:] = ( (Rstar - Rstar0)/eps )[:,:,:,:,:]
  Jinv = np.rollaxis(np.rollaxis( np.linalg.inv(np.rollaxis(np.rollaxis(J,3,10),2,9)) , 8 , 2) , 9 , 3)
  main.a.a[:] = a0[:]
  return Jinv

def computeJacobianZ(main,eqns,f):
  J = np.zeros((main.nvars,main.order[0],main.order[1],main.order[2],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  RHS0 = np.zeros(np.shape(main.RHS))
  RHStmp = np.zeros(np.shape(main.RHS))
  Rstar0,R0,Rstar_glob = f(main.a.a)
  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  eps = 1e-3
  for i in range(0,main.order[2]):
    main.a.a[:] = a0[:]
    main.a.a[:,:,:,i,:] = a0[:,:,:,i,:] + eps 
    Rstar,RHStmp,Rstar_glob = f(main.a.a)
    J[:,:,:,:,i,:] = ( (Rstar - Rstar0)/eps )[:,:,:,:,:]
  Jinv = np.rollaxis(np.rollaxis( np.linalg.inv(np.rollaxis(np.rollaxis(J,4,10),3,9)) , 8 , 3) , 9 , 4)
  main.a.a[:] = a0[:]
  return Jinv


def computeJacobianT(main,eqns,f):
  J = np.zeros((main.nvars,main.order[0],main.order[1],main.order[2],main.order[3],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  RHS0 = np.zeros(np.shape(main.RHS))
  RHStmp = np.zeros(np.shape(main.RHS))
  Rstar0,R0,Rstar_glob = f(main.a.a)
  a0 = np.zeros(np.shape(main.a.a))
  a0[:] = main.a.a[:]
  eps = 1e-3
  for i in range(0,main.order[3]):
    main.a.a[:] = a0[:]
    main.a.a[:,:,:,:,i] = a0[:,:,:,:,i] + eps 
    Rstar,RHStmp,Rstar_glob = f(main.a.a)
    J[:,:,:,:,:,i] = ( (Rstar - Rstar0)/eps )[:,:,:,:,:]
  Jinv = np.rollaxis(np.rollaxis( np.linalg.inv(np.rollaxis(np.rollaxis(J,5,10),4,9)) , 8 , 4) , 9 , 5)
  main.a.a[:] = a0[:]
  return Jinv






def computeJacobianX_full(main,eqns,f):
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


def computeJacobianY_full(main,eqns,f):
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
