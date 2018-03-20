import numpy as np
import numpy.linalg
from init_Classes import *
from adolc import *
import adolc
import scipy.sparse as sp
import time
def getR(a_flat,main,eqns):
  a = np.reshape(a_flat,np.shape(main.a.a))
  main.a.a = a[:]
  main.getRHS(main,main,eqns)
  return main.RHS.flatten()

def testAdolcJacobian1(main,main_adolc,eqns):
  N = np.size(main.a.a)
  #ax = numpy.array([adouble(0) for n in range(N)])
  ax = adouble(main.a.a.flatten())
#  main_adolc = variables(main.Nel,main.order,main.quadpoints,eqns,main.mus,main.x,main.y,main.z,main.t,main.et,main.dt,main.iteration,main.save_freq,main.turb_str,main.procx,main.procy,main.BCs,main.fsource,main.source_mag,main.shock_capturing,main.mol_str,main.basis_args,ax.dtype)
  a0 = main.a.a[:]*1.
  #R = getR(main,main,eqns)
  trace_on(1)
  independent(ax)
  ay = getR(ax,main_adolc,eqns)
  dependent(ay)
  trace_off()
  #x = numpy.array([n+1 for n in range(N)])
  # compute jacobian of f at x
  #main.a.a[:] = a0[:]

  x = main.a.a.flatten()
#  y = getR(x,main_adolc,eqns)
#  y2 = adolc.function(0,x)
#  assert numpy.allclose(y,y2)

#  options = numpy.array([0,0,0,0],dtype=int)
#  pat = adolc.sparse.jac_pat(0,x,options)
#  J = colpack.sparse_jac_no_repeat(0,x,options)
  J = jacobian(1,x)#main.a.a.flatten())

  return J

def testAdolcJacobian2(main,main_adolc,eqns):
  N = np.size(main.a.a)
  #ax = numpy.array([adouble(0) for n in range(N)])
  ax = adouble(main.a.a.flatten())
#  main_adolc = variables(main.Nel,main.order,main.quadpoints,eqns,main.mus,main.x,main.y,main.z,main.t,main.et,main.dt,main.iteration,main.save_freq,main.turb_str,main.procx,main.procy,main.BCs,main.fsource,main.source_mag,main.shock_capturing,main.mol_str,main.basis_args,ax.dtype)
  a0 = main.a.a[:]*1.
  #R = getR(main,main,eqns)
  ta = time.time()
  trace_on(0)
  independent(ax)
  ay = getR(ax,main_adolc,eqns)
  dependent(ay)
  trace_off()
  print(time.time()  - ta)
  #x = numpy.array([n+1 for n in range(N)])
  # compute jacobian of f at x
  #main.a.a[:] = a0[:]

  x = main.a.a.flatten()
#  y = getR(x,main_adolc,eqns)
#  y2 = adolc.function(0,x)
#  assert numpy.allclose(y,y2)

  options = numpy.array([0,0,0,0],dtype=int)
  t0 = time.time()
  pat = adolc.sparse.jac_pat(0,x,options)
  t1 = time.time()
  print(t1 - t0)
  result = colpack.sparse_jac_no_repeat(0,x,options)
  print(time.time() - t1)
  nnz = result[0]
  ridx = result[1]
  cidx = result[2]
  values = result[3]
  J = sp.csr_matrix((values, (ridx,cidx) ),shape=(N,N) )
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
