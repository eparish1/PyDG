import numpy as np
from MPI_functions import globalNorm,globalSum
import sys
from jacobian_schemes import *
import time
def globalMax(r,main):
  ## Create Global residual
  data = main.comm.gather(np.amax(np.abs(r)),root = 0)
  if (main.mpi_rank == 0):
    rn_glob = np.zeros(main.num_processes)
    for j in range(0,main.num_processes):
      rn_glob[j] = data[j]
    rn_glob = np.amax(rn_glob)
    for j in range(1,main.num_processes):
      main.comm.send(rn_glob, dest=j)
  else:
    rn_glob = main.comm.recv(source=0)
  return rn_glob



#def globalNorm(r,main):
#  ## Create Global residual
#  data = main.comm.gather(np.linalg.norm(r)**2,root = 0)
#  if (main.mpi_rank == 0):
#    rn_glob = 0.
#    for j in range(0,main.num_processes):
#      rn_glob += data[j]
#    rn_glob = np.sqrt(rn_glob)
#    for j in range(1,main.num_processes):
#      main.comm.send(rn_glob, dest=j)
#  else:
#    rn_glob = main.comm.recv(source=0)
#  return rn_glob
#
#def globalSum(r,main):
#  ## Create Global residual
#  data = main.comm.gather(np.sum(r),root = 0)
#  if (main.mpi_rank == 0):
#    rn_glob = 0.
#    for j in range(0,main.num_processes):
#      rn_glob += data[j]
#    for j in range(1,main.num_processes):
#      main.comm.send(rn_glob, dest=j)
#  else:
#    rn_glob = main.comm.recv(source=0)
#  return rn_glob

## Jacobi linear solver for Ax = b
# Af = matrix free method to evaluate A acting on f
# b, x0
# PC = matrix free method to evaluate diagonal inverse acting on f 
def Jacobi(Af,b,x0,main,args,PC,PCargs,tol=1e-9,maxiter_outer=1,maxiter=20,printnorm=0):
  omega = PCargs[0]
  PC_tol = PCargs[1]
  x = np.zeros(np.shape(x0))
  xtmp = np.zeros(np.shape(x0))
  x[:] = x0[:]
  k = 0
  r = b - Af(x,args,main)
  x[:] = x0[:]
  rnorm = globalNorm(r,main) #same across procs
  rnorm0 = rnorm*1.
  #print(omega,maxiter,np.shape(main.a.a))
  while(rnorm/rnorm0 >= tol and  k < maxiter and rnorm > 1e-9):
    r = PC(r,main,PCargs)
    x[:] = omega*r[:] + x[:]
    r = b - Af(x,args,main)
    rnorm = globalNorm(r,main) #same across procs
    k += 1
    if (main.mpi_rank == 0 and printnorm == 1):
      sys.stdout.write(' Iteration = ' + str(k) + ' Jacobi error = ' + str(rnorm)+ ' Relative Error = ' + str(rnorm/rnorm0) + '\n')

  return x

def ADI(Af,b,f0,main,MF_args,PC,PCargs,tol=1e-9,maxiter_outer=1,maxiter=20,printnorm=0):
  b = np.reshape(b,np.shape(main.a.a))
  f0 = np.reshape(f0,np.shape(main.a.a))
  unsteadyResidual = PC[0]
  unsteadyResidual_element_zeta = PC[1]
  unsteadyResidual_element_eta = PC[2]
  unsteadyResidual_element_mu = PC[3]
  unsteadyResidual_element_time = PC[4]

  MF_Jacobian = Af 
#  MF_Jacobian_element_zeta = MF_Jacobians[1]
#  MF_Jacobian_element_eta = MF_Jacobians[2]
#  MF_Jacobian_element_time = MF_Jacobians[3]

  f = np.zeros(np.shape(main.a.a))
  f[:] = f0[:]
  k = 0
  an = np.zeros(np.shape(main.a0))
  an[:] = main.a.a[:]
  resid_hist = np.zeros(0)
  t_hist = np.zeros(0)
  tnls = time.time()
  rho = -50.
  args = MF_args
  t0 = time.time()
  #JX,JY,JT = PCargs[0],PCargs[1],PCargs[2]
  JX = computeJacobianX(main,unsteadyResidual_element_zeta) #get the Jacobian
  JX = np.reshape(JX, (main.nvars*main.order[0],main.nvars*main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  JX = np.rollaxis(np.rollaxis(JX ,1,9),0,8)

  JY = computeJacobianY(main,unsteadyResidual_element_eta) #get the Jacobian
  JY = np.reshape(JY, (main.nvars*main.order[1],main.nvars*main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  JY = np.rollaxis(np.rollaxis(JY ,1,9),0,8)


  JT = computeJacobianT(main,unsteadyResidual_element_time) #get the Jacobian
  JT = np.reshape(JT, (main.nvars*main.order[3],main.nvars*main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
  JT = np.rollaxis(np.rollaxis(JT ,1,9),0,8)
  #print(time.time() - t0)
  ImatX = np.eye(main.nvars*main.order[0])
  ImatY = np.eye(main.nvars*main.order[1])
  ImatZ = np.eye(main.nvars*main.order[2])
  ImatT = np.eye(main.nvars*main.order[3])
  ta = time.time()
  Jf =np.reshape(  MF_Jacobian(f,args,main) , np.shape(main.a.a) )
  r = b - Jf
  rnorm = globalNorm(r,main) #same across procs
  rnorm0 = rnorm*1.
  JXinv = np.linalg.inv(JX + rho*ImatX)
  JYinv = np.linalg.inv(JY + rho*ImatY)
  JTinv = np.linalg.inv(JT + rho*ImatT)
  #printnorm = 1
  while(rnorm >= 1e-8 and rnorm/rnorm0 >= tol and  k < maxiter*20):
    #Jxf = MF_Jacobian_element_zeta(f,args_el,main)
    # perform iteration in the zeta direction   
    if (np.shape(f) != np.shape(main.a.a)):
      print('shape error') 
    f = np.reshape(f, (main.nvars*main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    f = np.rollaxis(f,0,8)
    #Jxf = np.reshape(Jxf, (main.nvars*main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    #Jxf = np.rollaxis(Jxf,0,8)
    Jf = np.reshape(Jf, (main.nvars*main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    Jf = np.rollaxis(Jf,0,8)
    b = np.reshape(b, (main.nvars*main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    b = np.rollaxis(b,0,8)
    ta = time.time()
    Jxf = np.einsum('pqrijklmn...,pqrijkln...->pqrijklm...',JX,f)
    #print('MF_x',np.linalg.norm(Jxfb - Jxf))
    ta = time.time()
    f[:] = np.linalg.solve(JX + rho*ImatX,b - (Jf - Jxf ) + rho*f) 
    #f[:] = np.einsum('pqrijklmn...,pqrijkln...->pqrijklm...',JXinv,b - (Jf - Jxf ) + rho*f)
    f = np.rollaxis(f,7,0)
    f = np.reshape(f,np.shape(main.a.a) )
    Jf = np.rollaxis(Jf,7,0)
    Jf = np.reshape(Jf,np.shape(main.a.a) )
    b = np.rollaxis(b,7,0)
    b = np.reshape(b,np.shape(main.a.a))


#    # now perform iteration in the eta direction
#    if (np.shape(f) != np.shape(main.a.a)):
#      print('shape error') 
#    #Jyf2 = MF_Jacobian_element_eta(f,args_el,main)
#    Jf = np.reshape( MF_Jacobian(f,args,main) , np.shape(main.a.a) )
#    f = np.rollaxis(f,2,1)
#    f = np.reshape(f, (main.nvars*main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
#    f = np.rollaxis(f,0,8)
#    Jf = np.rollaxis(Jf,2,1)
#    Jf = np.reshape(Jf, (main.nvars*main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
#    Jf = np.rollaxis(Jf,0,8)
#    #Jyf2 = np.rollaxis(Jyf2,2,1)
#    #Jyf2 = np.reshape(Jyf2, (main.nvars*main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
#    #Jyf2 = np.rollaxis(Jyf2,0,8)
#    Jyf = np.einsum('pqrijklmn...,pqrijkln...->pqrijklm...',JY,f)
#    #print('MF_Y',np.linalg.norm(Jyf2 - Jyf))
#    b = np.rollaxis(b,2,1)
#    b = np.reshape(b, (main.nvars*main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
#    b = np.rollaxis(b,0,8)
#    f[:] = np.linalg.solve(JY + rho*ImatY,b - (Jf - Jyf) + rho*f)
#    #f[:] =  np.einsum('pqrijklmn...,pqrijkln...->pqrijklm...',JYinv,b - (Jf - Jyf)*0. + rho*f)
#    f = np.rollaxis(f,7,0)
#    f = np.reshape(f, (main.nvars,main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
#    f = np.rollaxis(f,1,3)
#    b = np.rollaxis(b,7,0)
#    b = np.reshape(b, (main.nvars,main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
#    b = np.rollaxis(b,1,3)



    # now perform iteration in the time direction
    if (np.shape(f) != np.shape(main.a.a)):
      print('shape error') 
    #Jtf = MF_Jacobian_element_time(f,args_el,main)
    Jf = np.reshape( MF_Jacobian(f,args,main) , np.shape(main.a.a) )
    f = np.rollaxis(f,4,1)
    f = np.reshape(f, (main.nvars*main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
    f = np.rollaxis(f,0,8)
    Jf = np.rollaxis(Jf,4,1)
    Jf = np.reshape(Jf, (main.nvars*main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
    Jf = np.rollaxis(Jf,0,8)
    #Jtf = np.rollaxis(Jtf,4,1)
    #Jtf = np.reshape(Jtf, (main.nvars*main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
    #Jtf = np.rollaxis(Jtf,0,8)
    Jtf = np.einsum('pqrijklmn...,pqrijkln...->pqrijklm...',JT,f)
    #print('MF_T',np.linalg.norm(Jtfb - Jtf))
    b = np.rollaxis(b,4,1)
    b = np.reshape(b, (main.nvars*main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
    b = np.rollaxis(b,0,8)
    f[:] = np.linalg.solve(JT + rho*ImatT,b - (Jf - Jtf) + rho*f)
    #f[:] = np.einsum('pqrijklmn...,pqrijkln...->pqrijklm...',JTinv,b - (Jf - Jtf) + rho*f) 
    f = np.rollaxis(f,7,0)
    f = np.reshape(f, (main.nvars,main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
    f = np.rollaxis(f,1,5)
    b = np.rollaxis(b,7,0)
    b = np.reshape(b, (main.nvars,main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
    b = np.rollaxis(b,1,5)
    k += 1
    ts = time.time()
    #an[:] = main.a.a[:]
    Jf = np.reshape( MF_Jacobian(f,args,main) , np.shape(main.a.a) )
    r = b - Jf

    rnormn = globalNorm(r,main) #same across procs
    if (rnormn <= rnorm):
      rho = rho*0.98
    else:
      rho = rho*3
    rnorm = rnormn*1.
    t_hist = np.append(t_hist,time.time() - tnls)
    if (main.mpi_rank == 0 and printnorm == 1):
      sys.stdout.write('ADI iteration = ' + str(k) + ' residual = ' + str(rnorm) + ' relative decrease = ' + str(rnorm/rnorm0) + ' Solve time = ' + str(time.time() - ts)  + '  rho = ' + str(rho) + '\n')
      sys.stdout.flush()
  return f.flatten()
  #np.savez('resid_history',resid=resid_hist,t=t_hist)

def operatorSplitting(Af,b,f0,main,MF_args,PC,PCargs,tol=1e-9,maxiter_outer=1,maxiter=20,printnorm=0):
  b = np.reshape(b,np.shape(main.a.a))
  f0 = np.reshape(f0,np.shape(main.a.a))
  unsteadyResidual = PC[0]
  unsteadyResidual_element_zeta = PC[1]
  unsteadyResidual_element_eta = PC[2]
  unsteadyResidual_element_mu = PC[3]
  unsteadyResidual_element_time = PC[4]

  MF_Jacobian = Af 
#  MF_Jacobian_element_zeta = MF_Jacobians[1]
#  MF_Jacobian_element_eta = MF_Jacobians[2]
#  MF_Jacobian_element_time = MF_Jacobians[3]

  f = np.zeros(np.shape(main.a.a))
  f[:] = f0[:]
  k = 0
  an = np.zeros(np.shape(main.a0))
  an[:] = main.a.a[:]
  resid_hist = np.zeros(0)
  t_hist = np.zeros(0)
  tnls = time.time()
  rho = -1000.
  args = MF_args
  t0 = time.time()
  #JX,JY,JT = PCargs[0],PCargs[1],PCargs[2]
  JX = computeJacobianX(main,unsteadyResidual_element_zeta) #get the Jacobian
  JX = np.reshape(JX, (main.nvars*main.order[0],main.nvars*main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  JX = np.rollaxis(np.rollaxis(JX ,1,9),0,8)

  JY = computeJacobianY(main,unsteadyResidual_element_eta) #get the Jacobian
  JY = np.reshape(JY, (main.nvars*main.order[1],main.nvars*main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  JY = np.rollaxis(np.rollaxis(JY ,1,9),0,8)


  JT = computeJacobianT(main,unsteadyResidual_element_time) #get the Jacobian
  JT = np.reshape(JT, (main.nvars*main.order[3],main.nvars*main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
  JT = np.rollaxis(np.rollaxis(JT ,1,9),0,8)
  #print(time.time() - t0)
  ImatX = np.eye(main.nvars*main.order[0])
  ImatY = np.eye(main.nvars*main.order[1])
  ImatZ = np.eye(main.nvars*main.order[2])
  ImatT = np.eye(main.nvars*main.order[3])
  ta = time.time()
  #Jf =np.reshape(  MF_Jacobian(f,args,main) , np.shape(main.a.a) )
  #r = b - Jf
  #rnorm = globalNorm(r,main) #same across procs
  #rnorm0 = rnorm*1.
  #while(rnorm >= 1e-8 and rnorm/rnorm0 >= tol and  k < maxiter):
  while(k < maxiter):
    #Jxf = MF_Jacobian_element_zeta(f,args_el,main)
    # perform iteration in the zeta direction   
    if (np.shape(f) != np.shape(main.a.a)):
      print('shape error') 
    f = np.reshape(f, (main.nvars*main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    f = np.rollaxis(f,0,8)
    b = np.reshape(b, (main.nvars*main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    b = np.rollaxis(b,0,8)
    ta = time.time()
    ta = time.time()
    f[:] = np.linalg.solve(JX + rho*ImatX,b + rho*f) 
    f = np.rollaxis(f,7,0)
    f = np.reshape(f,np.shape(main.a.a) )
    b = np.rollaxis(b,7,0)
    b = np.reshape(b,np.shape(main.a.a))


    # now perform iteration in the eta direction
    if (np.shape(f) != np.shape(main.a.a)):
      print('shape error') 
    #Jyf2 = MF_Jacobian_element_eta(f,args_el,main)
    f = np.rollaxis(f,2,1)
    f = np.reshape(f, (main.nvars*main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    f = np.rollaxis(f,0,8)
    #print('MF_Y',np.linalg.norm(Jyf2 - Jyf))
    b = np.rollaxis(b,2,1)
    b = np.reshape(b, (main.nvars*main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    b = np.rollaxis(b,0,8)
    f[:] = np.linalg.solve(JY + rho*ImatY,b + rho*f)

    f = np.rollaxis(f,7,0)
    f = np.reshape(f, (main.nvars,main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    f = np.rollaxis(f,1,3)
    b = np.rollaxis(b,7,0)
    b = np.reshape(b, (main.nvars,main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    b = np.rollaxis(b,1,3)



    # now perform iteration in the time direction
    if (np.shape(f) != np.shape(main.a.a)):
      print('shape error') 
    #Jtf = MF_Jacobian_element_time(f,args_el,main)
    f = np.rollaxis(f,4,1)
    f = np.reshape(f, (main.nvars*main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
    f = np.rollaxis(f,0,8)
    b = np.rollaxis(b,4,1)
    b = np.reshape(b, (main.nvars*main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
    b = np.rollaxis(b,0,8)
    f[:] = np.linalg.solve(JT + rho*ImatT,b + rho*f)
    f = np.rollaxis(f,7,0)
    f = np.reshape(f, (main.nvars,main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
    f = np.rollaxis(f,1,5)
    b = np.rollaxis(b,7,0)
    b = np.reshape(b, (main.nvars,main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
    b = np.rollaxis(b,1,5)
    k += 1
    ts = time.time()
    #an[:] = main.a.a[:]
    #Jf = np.reshape( MF_Jacobian(f,args,main) , np.shape(main.a.a) )
    #r = b - Jf
    #rnormn = globalNorm(r,main) #same across procs
    #if (rnormn <= rnorm):
    #  rho = rho*0.98
    #else:
    #  rho = rho*3
    #rnorm = rnormn*1.
    t_hist = np.append(t_hist,time.time() - tnls)
    if (main.mpi_rank == 0 and printnorm == 1):
      sys.stdout.write('ADI iteration = ' + str(k) + ' residual = ' + str(rnorm) + ' relative decrease = ' + str(rnorm/rnorm0) + ' Solve time = ' + str(time.time() - ts)  + '  rho = ' + str(rho) + '\n')
      sys.stdout.flush()
  return f.flatten()



def rungeKutta(Af,b,x0,main,args,PC,PCargs,tol=1e-9,maxiter_outer=1,maxiter=20,printnorm=0):
     tau = -0.015
     r = b - Af(x0,args,main)
     rnorm = globalNorm(r,main) #same across procs
     rnorm_old = rnorm*1.
     iteration = 0
     rk4const = np.array([0.15,1.0])
     x = np.zeros(np.shape(x0))
     x[:] = x0[:]
     print_freq = 500
     while( rnorm > 1e-9 and iteration <= maxiter):
       x0[:] = x[:]
       gamma1 = -1.
       gamma2 = -1./3.
       v0 = tau*r
       x1 = x0 - gamma1*v0
       v1 = -tau*Af(v0,args,main)
       x = x1 - gamma2*v1
       r = b - Af(x,args,main)
       rnorm = globalNorm(r,main) 
       #tau = tau*np.fmin(rnorm_old/rnorm,1.001)
       rnorm_old = rnorm*1.
       iteration += 1
       if (main.mpi_rank == 0 and iteration%print_freq == 0):# and printnorm == 1):
         sys.stdout.write(' Iteration = ' + str(iteration) + ' Runge Kutta error = ' + str(rnorm) + ' tau = ' + str(tau) +  '\n')
     if (main.mpi_rank == 0 and printnorm == 1):
       sys.stdout.write(' ================================== ' +  '\n')
     return x

def GMRes(Af, b, x0,main,args,PC=None,PCargs=None,tol=1e-9,maxiter_outer=1,maxiter=20,printnorm=0):
    k_outer = 0
    bnorm = globalNorm(b,main)
    error = 10.
    x = np.zeros(np.shape(x0))
    x[:] = x0[:]
    main.linear_iteration = 0
    while (k_outer < maxiter_outer and error >= tol):
      r = b - Af(x0,args,main)
      if (main.mpi_rank == 0 and printnorm==1):
        print('Outer true norm = ' + str(np.linalg.norm(r)))
      cs = np.zeros(maxiter) #should be the same on all procs
      sn = np.zeros(maxiter) #same on all procs
#      e1 = np.zeros(np.size(b)) #should vary across procs
      e1 = np.zeros(maxiter+1)
      e1[0] = 1
  
      rnorm = globalNorm(r,main) #same across procs
      Q = np.zeros((np.size(b),maxiter)) 
      #v = [0] * (nmax_iter)
      Q[:,0] = r / rnorm ## The first index of Q is across all procs
      H = np.zeros((maxiter + 1, maxiter)) ### this should be the same on all procs
      beta = rnorm*e1
      main.linear_iteration = 0
      k = 0
      while (k < maxiter - 1  and error >= tol):
  #    for k in range(0,nmax_iter-1):
          Arnoldi(Af,H,Q,k,args,main)
          apply_givens_rotation(H,cs,sn,k)
          #update the residual vector
          beta[k+1] = -sn[k]*beta[k]
          beta[k] = cs[k]*beta[k]
          error = abs(beta[k+1])/bnorm
          main.linear_iteration += 1

          ## For testing
          #y = np.linalg.solve(H[0:k,0:k],beta[0:k]) 
          #x = x0 + np.dot(Q[:,0:k],y)
          #rt = b - Af(x)
          #rtnorm = np.linalg.norm(rt)#globalNorm(rt,main)
          if (main.mpi_rank == 0 and printnorm == 1):
            sys.stdout.write('Outer iteration = ' + str(k_outer) + ' Iteration = ' + str(k) + '  GMRES error = ' + str(error) +  '\n')
            #print('Outer iteration = ' + str(k_outer) + ' Iteration = ' + str(k) + '  GMRES error = ' + str(error), ' Real norm = ' + str(rtnorm))
          k += 1
      y = np.linalg.solve(H[0:k,0:k],beta[0:k]) 
      x = x0 + np.dot(Q[:,0:k],y)
      x0[:] = x[:]
      k_outer += 1
    return x[:]


#routine for GMRES solver where the problem is local to each element
def elementNorm(f,main):
  fsum = np.sum(f**2,axis=0)
  return np.sqrt(fsum)

def elementSum(f,main):
  return np.sum(f,axis=0)

def GMRes_element(Af, b, x0,main,args,PC=None,PCargs=None,tol=1e-9,maxiter_outer=1,maxiter=20,printnorm=0):
    k_outer = 0
    bnorm = elementNorm(b,main)
    error = 10.
    x = np.zeros(np.shape(x0))
    x[:] = x0[:]
    while (k_outer < maxiter_outer and globalMax(error,main) >= tol):
      r = b - Af(x0,args,main)
      if (main.mpi_rank == 0 and printnorm==1):
        print('Outer true norm = ' + str(np.linalg.norm(r)))
      cs = np.zeros((maxiter,main.Npx,main.Npy,main.Npz,main.Npt)) #should be the same on all procs
      sn = np.zeros((maxiter,main.Npx,main.Npy,main.Npz,main.Npt)) #same on all procs
      e1 = np.zeros((maxiter+1,main.Npx,main.Npy,main.Npz,main.Npt))
      e1[0] = 1
  
      rnorm = elementNorm(r,main) #same across procs
      solve_size = np.shape(b)[0]
      Q = np.zeros((solve_size,maxiter,main.Npx,main.Npy,main.Npz,main.Npt)) 
      #v = [0] * (nmax_iter)
      Q[:,0] = r / rnorm ## The first index of Q is across all procs
      H = np.zeros((maxiter + 1, maxiter,main.Npx,main.Npy,main.Npz,main.Npt)) ### this should be the same on all procs
      beta = rnorm[None]*e1

      k = 0
      while (k < maxiter - 1  and globalMax(error,main) >= tol):
  #    for k in range(0,nmax_iter-1):
          Arnoldi_element(Af,H,Q,k,args,main)
          apply_givens_rotation_element(H,cs,sn,k)
          #update the residual vector
          beta[k+1] = -sn[k]*beta[k]
          beta[k] = cs[k]*beta[k]
          error = abs(beta[k+1])/bnorm
          if (main.mpi_rank == 0 and printnorm == 1):
            sys.stdout.write('Outer iteration = ' + str(k_outer) + ' Iteration = ' + str(k) + '  GMRES error = ' + str(np.mean(error)) +  '\n')
            #print('Outer iteration = ' + str(k_outer) + ' Iteration = ' + str(k) + '  GMRES error = ' + str(error), ' Real norm = ' + str(rtnorm))
          k += 1
      H = np.rollaxis(np.rollaxis(H,1,6),0,5)
      beta = np.rollaxis(beta,0,5)
      y = np.linalg.solve(H[:,:,:,:,0:k,0:k],beta[:,:,:,:,0:k])
      H = np.rollaxis(np.rollaxis(H,4,0),5,1)
      Q = np.rollaxis(np.rollaxis(Q,1,6),0,5)
      tmp = np.einsum('ijklmn,ijkln->ijklm',Q[:,:,:,:,:,0:k],y) 
      tmp = np.rollaxis(tmp,4,0)
      x = x0 + tmp#np.rollaxis( np.dot(Q[:,:,:,:,:,0:k],y),5,0)
      x0[:] = x[:]
      k_outer += 1
    return x[:]



def BICGSTAB(Af, b, x0,main,args,PC=None,PC_args=None,tol=1e-9,maxiter_outer=1,maxiter=50,printnorm=0):
  r0 = b - Af(x0,args,main)
  rhat0 = np.zeros(np.shape(r0))
  rhat0[:] = r0[:]
  rhat0_norm = globalNorm(rhat0,main)
  r0_norm = rhat0_norm*1.
  rho0,alpha,omega0 = 1.,1.,1.
  v0,p0 = np.zeros(np.shape(r0)),np.zeros(np.shape(r0))
  iterat = 0
  while (r0_norm/rhat0_norm  >= tol and iterat <= maxiter):
    rhoi = globalSum(rhat0*r0,main)
    beta = rhoi/rho0*alpha/omega0
    p0 = r0 + beta*(p0 - omega0*v0)
    v0 = Af(p0,args,main)
    alpha = rhoi/globalSum(rhat0*v0,main)
    h = x0 + alpha*p0
    s = r0 - alpha*v0
    t = Af(s,args,main)
    omega0 = globalSum(t*s,main)/globalSum(t*t,main)
    x0 = h + omega0*s
    r0 = s - omega0*t
    #update old values
    rho0 = rhoi*1.
    r0_norm = globalNorm(r0,main)
    if (main.mpi_rank == 0 and printnorm == 1):
      sys.stdout.write(' Iteration = ' + str(iterat) + '  BICGSTAB residual = ' + str(r0_norm/rhat0_norm) +  '\n')
    iterat += 1
  return x0


def fGMRes(Af, b, x0,main,args,PC=None,PC_args=None,tol=1e-9,maxiter_outer=1,maxiter=20,printnorm=1):
    #printnorm = 1
    k_outer = 0
    bnorm = globalNorm(b,main)
    error = 1.
    x = np.zeros(np.shape(x0))
    x[:] = x0[:]
    coarse_order = np.shape(main.a.a)[1:5]
    while (k_outer < maxiter_outer and error >= tol):
      r = b - Af(x0,args,main)
      if (main.mpi_rank == 0 and printnorm==1):
        print('Outer true norm = ' + str(np.linalg.norm(r)))
      cs = np.zeros(maxiter) #should be the same on all procs
      sn = np.zeros(maxiter) #same on all procs
      e1 = np.zeros(np.size(b)) #should vary across procs
      e1[0] = 1
  
      rnorm = globalNorm(r,main) #same across procs
      Q = np.zeros((np.size(b),maxiter)) 
      Z = np.zeros((np.size(b),maxiter)) 
      #v = [0] * (nmax_iter)
      Q[:,0] = r / rnorm ## The first index of Q is across all procs
      H = np.zeros((maxiter + 1, maxiter)) ### this should be the same on all procs
      beta = rnorm*e1
      k = 0
      while (k < maxiter - 1  and error >= tol):
          Z[:,k] = PC(Q[:,k],main,PC_args)
          Q[:,k+1] = Af(Z[:,k],args,main)
          Arnoldi_fgmres(Af,H,Q,k,args,main)
          apply_givens_rotation(H,cs,sn,k)
          #update the residual vector
          beta[k+1] = -sn[k]*beta[k]
          beta[k] = cs[k]*beta[k]
          error = abs(beta[k+1])/bnorm
          if (main.mpi_rank == 0 and printnorm == 1):
            sys.stdout.write('Outer iteration = ' + str(k_outer) + \
            ' Iteration = ' + str(k) + '  GMRES error = ' + str(error) +  '\n')
          k += 1
      y = np.linalg.solve(H[0:k,0:k],beta[0:k]) 
      Zy = np.dot(Z[:,0:k],y) 
      x = x0 + Zy 
      x0[:] = x[:]
      k_outer += 1
    return x[:]

def Arnoldi_fgmres(Af,H,Q,k,args,main):
    for i in range(0,k+1):
        H[i, k] = globalSum(Q[:,i]*Q[:,k+1],main)
        Q[:,k+1] = Q[:,k+1] - H[i, k] * Q[:,i]
    H[k + 1, k] = globalNorm(Q[:,k+1],main)
#    if (h[k + 1, k] != 0 and k != nmax_iter - 1):
    Q[:,k + 1] = Q[:,k+1] / H[k + 1, k]

def Arnoldi_element(Af,H,Q,k,args,main):
    Q[:,k+1] = Af(Q[:,k],args,main)
    for i in range(0,k+1):
        H[i, k] = elementSum(Q[:,i]*Q[:,k+1],main)
        Q[:,k+1] = Q[:,k+1] - H[i, k] * Q[:,i]
    H[k + 1, k] = elementNorm(Q[:,k+1],main)
#    if (h[k + 1, k] != 0 and k != nmax_iter - 1):
    Q[:,k + 1] = Q[:,k+1] / H[k + 1, k]
#    return h,v 



def Arnoldi(Af,H,Q,k,args,main):
    Q[:,k+1] = Af(Q[:,k],args,main)
    for i in range(0,k+1):
        H[i, k] = globalSum(Q[:,i]*Q[:,k+1],main)
        Q[:,k+1] = Q[:,k+1] - H[i, k] * Q[:,i]
    H[k + 1, k] = globalNorm(Q[:,k+1],main)
#    if (h[k + 1, k] != 0 and k != nmax_iter - 1):
    Q[:,k + 1] = Q[:,k+1] / H[k + 1, k]
#    return h,v 

def apply_givens_rotation(H, cs, sn, k):
  #apply for ith column
  for i in range(0,k):                              
    temp     =  cs[i]*H[i,k] + sn[i]*H[i+1,k]
    H[i+1,k] = -sn[i]*H[i,k] + cs[i]*H[i+1,k]
    H[i,k]   = temp
  
  #update the next sin cos values for rotation
  cs[k],sn[k] = givens_rotation( H[k,k], H[k+1,k])
  
  #eliminate H(i+1,i)
  H[k,k] = cs[k]*H[k,k] + sn[k]*H[k+1,k]
  H[k+1,k] = 0.0

#----Calculate the Given rotation matrix----%%
def givens_rotation(v1, v2):
  if (v1==0):
    cs = 0
    sn = 1
  else:
    t=np.sqrt(v1**2+v2**2)
    cs = np.abs(v1) / t
    sn = cs * v2 / v1
  return cs,sn



def apply_givens_rotation_element(H, cs, sn, k):
  #apply for ith column
  for i in range(0,k):                              
    temp     =  cs[i]*H[i,k] + sn[i]*H[i+1,k]
    H[i+1,k] = -sn[i]*H[i,k] + cs[i]*H[i+1,k]
    H[i,k]   = temp
  
  #update the next sin cos values for rotation
  cs[k],sn[k] = givens_rotation_element( H[k,k], H[k+1,k])
  
  #eliminate H(i+1,i)
  H[k,k] = cs[k]*H[k,k] + sn[k]*H[k+1,k]
  H[k+1,k] = 0.0

#----Calculate the Given rotation matrix----%%
def givens_rotation_element(v1, v2):
    t=np.sqrt(v1**2+v2**2)
    cs = np.abs(v1) / t
    sn = cs * v2 / v1
    cs[v1==0] = 0
    sn[v1==0] = 1
    return cs,sn





def GMResOrig(Af, b, x0, nmax_iter, restart=None):
    r = b - Af(x0)
    rnorm = np.linalg.norm(r)
    v = [0] * (nmax_iter)
    v[0] = r / rnorm
    h = np.zeros((nmax_iter + 1, nmax_iter))
    for k in range(0,nmax_iter):
        w = Af(v[k])
        for j in range(0,k+1):
            h[j, k] = np.dot(v[j], w)
            w = w - h[j, k] * v[j]
        h[k + 1, k] = np.linalg.norm(w)
        if (h[k + 1, k] != 0 and k != nmax_iter - 1):
            v[k + 1] = w / h[k + 1, k]
        b2 = np.zeros(nmax_iter + 1)
        b2[0] = rnorm
        result = np.linalg.lstsq(h, b2)[0]
        x = np.dot(np.asarray(v).transpose(), result) + x0
        r1 = b - Af(x)
        print(np.linalg.norm(r))
    return x[:]

