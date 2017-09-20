import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from init_Classes import variables,equations
from linear_solvers import *
from jacobian_schemes import *
def newtonSolver(unsteadyResidual,MF_Jacobian,main,linear_solver,sparse_quadrature,eqns,PC=None):
  if (sparse_quadrature):
    coarsen = 2
    quadpoints_coarsen = np.fmax(main.quadpoints/(coarsen),1)
    quadpoints_coarsen[-1] = main.quadpoints[-1]
    main_coarse = variables(main.Nel,main.order,quadpoints_coarsen,eqns,main.mus,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy,main.BCs,main.fsource,main.source_mag,main.shock_capturing,main.mol_str)
    main_coarse.basis = main.basis
    main_coarse.a.a[:] = main.a.a[:]
    def newtonHook(main_coarse,main,Rn):
      main_coarse.a.a[:] = main.a.a[:]
      main_coarse.getRHS(main_coarse,main_coarse,eqns)
      #getRHS_SOURCE(main_coarse,main_coarse,eqns)
      Rn[:] = main_coarse.RHS[:]
  else: 
    main_coarse = main
    def newtonHook(main_coarse,main,Rn):
       pass
  Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a)
  NLiter = 0
  an = np.zeros(np.shape(main.a0))
  an[:] = main.a0[:]
  Rstar_glob0 = Rstar_glob*1. 
  old = np.zeros(np.shape(main.a.a))
  resid_hist = np.zeros(0)
  t_hist = np.zeros(0)
  tnls = time.time()
  while (Rstar_glob >= 1e-8 and Rstar_glob/Rstar_glob0 > 1e-8):
    NLiter += 1
    ts = time.time()
    newtonHook(main_coarse,main,Rn)
    MF_Jacobian_args = [an,Rn]
    delta = 1
#    if (Rstar_glob/Rstar_glob0 < 1e-4):
#      delta = 2
#    if (Rstar_glob/Rstar_glob0 < 1e-5):
#      delta = 3
#    if (Rstar_glob/Rstar_glob0 < 1e-6):
#      delta = 3
    loc_tol = 0.1*Rstar_glob/Rstar_glob0
    PC_iteration = 0
    PC_args = [1,loc_tol,PC_iteration]
    sol = linear_solver.solve(MF_Jacobian, -Rstarn.flatten(), old.flatten(),main_coarse,MF_Jacobian_args,PC,PC_args,loc_tol,linear_solver.maxiter_outer,20,False)
    main.a.a[:] = an[:] + 1.0*np.reshape(sol,np.shape(main.a.a))
    an[:] = main.a.a[:]
    Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a)
    resid_hist = np.append(resid_hist,Rstar_glob)
    t_hist = np.append(t_hist,time.time() - tnls)
    if (main.mpi_rank == 0):
      sys.stdout.write('Newton iteration = ' + str(NLiter) + '  NL residual = ' + str(Rstar_glob) + ' relative decrease = ' + str(Rstar_glob/Rstar_glob0) + ' Solve time = ' + str(time.time() - ts)  + '\n')
      #print(np.linalg.norm(Rstarn[0]),np.linalg.norm(Rstarn[-1]))
      sys.stdout.flush()
  np.savez('resid_history',resid=resid_hist,t=t_hist)




def NEJSolver(unsteadyResidual,MF_Jacobian,main,linear_solver,sparse_quadrature,eqns,PC=None):
  if (sparse_quadrature):
    coarsen = 2
    quadpoints_coarsen = np.fmax(main.quadpoints/(coarsen),1)
    quadpoints_coarsen[-1] = main.quadpoints[-1]
    main_coarse = variables(main.Nel,main.order,quadpoints_coarsen,eqns,main.mus,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy,main.BCs,main.fsource,main.source_mag,main.shock_capturing,main.mol_str)
    main_coarse.basis = main.basis
    main_coarse.a.a[:] = main.a.a[:]
    def newtonHook(main_coarse,main,Rn):
      main_coarse.a.a[:] = main.a.a[:]
      main_coarse.getRHS(main_coarse,main_coarse,eqns)
      #getRHS_SOURCE(main_coarse,main_coarse,eqns)
      Rn[:] = main_coarse.RHS[:]
  else: 
    main_coarse = main
    def newtonHook(main_coarse,main,Rn):
       pass
  Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a)
  NLiter = 0
  an = np.zeros(np.shape(main.a0))
  an[:] = main.a.a[:]
  Rstar_glob0 = Rstar_glob*1. 
  old = np.zeros(np.shape(main.a.a))
  resid_hist = np.zeros(0)
  t_hist = np.zeros(0)
  tnls = time.time()
  omega = 1. 
  while (Rstar_glob >= 1e-8 and Rstar_glob/Rstar_glob0 > 1e-8):
    NLiter += 1
    ts = time.time()
    main.a.a[:] = an[:]
    loc_tol = 0.1*Rstar_glob/Rstar_glob0
    PC_iteration = 0
    PC_args = [omega,loc_tol,PC_iteration]
    r = PC(-Rstarn.flatten(),main,PC_args)
    main.a.a[:] = omega*np.reshape(r,np.shape(main.a.a)) + an[:]
    an[:] = main.a.a[:]
    rnorm = globalNorm(r,main) #same across procs
    Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a)

    resid_hist = np.append(resid_hist,Rstar_glob)
    t_hist = np.append(t_hist,time.time() - tnls)
    if (main.mpi_rank == 0):
      sys.stdout.write('NEJ iteration = ' + str(NLiter) + '  NL residual = ' + str(Rstar_glob) + ' relative decrease = ' + str(Rstar_glob/Rstar_glob0) + ' Solve time = ' + str(time.time() - ts)  + '\n')
      sys.stdout.flush()
  np.savez('resid_history',resid=resid_hist,t=t_hist)


def ADISolver(unsteadyResiduals,MF_Jacobians,main,linear_solver,sparse_quadrature,eqns,PC=None):
  #computeJacobianX = computeJacobians[0]
  #computeJacobianT = computeJacobians[1]
  unsteadyResidual = unsteadyResiduals[0]
  unsteadyResidual_element_zeta = unsteadyResiduals[1]
  unsteadyResidual_element_eta = unsteadyResiduals[2]
  unsteadyResidual_element_mu = unsteadyResiduals[3]
  unsteadyResidual_element_time = unsteadyResiduals[4]

  MF_Jacobian = MF_Jacobians[0]
  MF_Jacobian_element_zeta = MF_Jacobians[1]
  MF_Jacobian_element_eta = MF_Jacobians[2]
  MF_Jacobian_element_time = MF_Jacobians[3]

  f = np.zeros(np.shape(main.a.a))
  f0 = np.zeros(np.shape(main.a.a))
  dum,Rn_el,dum = unsteadyResidual_element_zeta(main,main.a.a)
#  print('ADI resid',np.linalg.norm(Rn_el))
  Rstarn0,Rn,Rstar_glob = unsteadyResidual(main.a.a)
  NLiter = 0
  an = np.zeros(np.shape(main.a0))
  an[:] = main.a.a[:]
  Rstar_glob0 = Rstar_glob*1. 
  resid_hist = np.zeros(0)
  t_hist = np.zeros(0)
  tnls = time.time()
  rho = -40.
  args = [an,Rn]
  args_el = [an,Rn_el]
  JX = computeJacobianX(main,unsteadyResidual_element_zeta) #get the Jacobian
  JX = np.reshape(JX, (main.nvars*main.order[0],main.nvars*main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  JX = np.rollaxis(np.rollaxis(JX ,1,9),0,8)

  JY = computeJacobianY(main,unsteadyResidual_element_eta) #get the Jacobian
  JY = np.reshape(JY, (main.nvars*main.order[1],main.nvars*main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
  JY = np.rollaxis(np.rollaxis(JY ,1,9),0,8)


  JT = computeJacobianT(main,unsteadyResidual_element_time) #get the Jacobian
  JT = np.reshape(JT, (main.nvars*main.order[3],main.nvars*main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
  JT = np.rollaxis(np.rollaxis(JT ,1,9),0,8)

  ImatX = np.eye(main.nvars*main.order[0])
  ImatY = np.eye(main.nvars*main.order[1])
  ImatZ = np.eye(main.nvars*main.order[2])
  ImatT = np.eye(main.nvars*main.order[3])
  while (Rstar_glob >= 1e-8 and Rstar_glob/Rstar_glob0 > 1e-8):
    f0[:] = f0[:]
    #Jxf = MF_Jacobian_element_zeta(f,args_el,main)
    Jf = MF_Jacobian(f,args,main)
    # perform iteration in the zeta direction   
    if (np.shape(f) != np.shape(main.a.a)):
      print('shape error') 
    f = np.reshape(f, (main.nvars*main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    f = np.rollaxis(f,0,8)
    #Jxf = np.reshape(Jxf, (main.nvars*main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    #Jxf = np.rollaxis(Jxf,0,8)
    Jf = np.reshape(Jf, (main.nvars*main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    Jf = np.rollaxis(Jf,0,8)
    Rstarn0 = np.reshape(Rstarn0, (main.nvars*main.order[0],main.order[1],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    Rstarn0 = np.rollaxis(Rstarn0,0,8)
    Jxf = np.einsum('pqrijklmn...,pqrijkln...->pqrijklm...',JX,f)
    #print('MF_x',np.linalg.norm(Jxfb - Jxf))
    f[:] = np.linalg.solve(JX + rho*ImatX,-Rstarn0 - (Jf - Jxf ) + rho*f) 
    f = np.rollaxis(f,7,0)
    f = np.reshape(f,np.shape(main.a.a) )
    Jf = np.rollaxis(Jf,7,0)
    Jf = np.reshape(Jf,np.shape(main.a.a) )
    Rstarn0 = np.rollaxis(Rstarn0,7,0)
    Rstarn0 = np.reshape(Rstarn0,np.shape(main.a.a))


    # now perform iteration in the eta direction
    if (np.shape(f) != np.shape(main.a.a)):
      print('shape error') 
    #Jyf2 = MF_Jacobian_element_eta(f,args_el,main)
    Jf = MF_Jacobian(f,args,main)
    f = np.rollaxis(f,2,1)
    f = np.reshape(f, (main.nvars*main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    f = np.rollaxis(f,0,8)
    Jf = np.rollaxis(Jf,2,1)
    Jf = np.reshape(Jf, (main.nvars*main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    Jf = np.rollaxis(Jf,0,8)
    #Jyf2 = np.rollaxis(Jyf2,2,1)
    #Jyf2 = np.reshape(Jyf2, (main.nvars*main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    #Jyf2 = np.rollaxis(Jyf2,0,8)
    Jyf = np.einsum('pqrijklmn...,pqrijkln...->pqrijklm...',JY,f)
    #print('MF_Y',np.linalg.norm(Jyf2 - Jyf))
    Rstarn0 = np.rollaxis(Rstarn0,2,1)
    Rstarn0 = np.reshape(Rstarn0, (main.nvars*main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    Rstarn0 = np.rollaxis(Rstarn0,0,8)
    #f[:] = np.linalg.solve(JY + rho*ImatY,-Rstarn0 - (Jf - Jyf) + rho*f)
    f = np.rollaxis(f,7,0)
    f = np.reshape(f, (main.nvars,main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    f = np.rollaxis(f,1,3)
    Rstarn0 = np.rollaxis(Rstarn0,7,0)
    Rstarn0 = np.reshape(Rstarn0, (main.nvars,main.order[1],main.order[0],main.order[2],main.order[3],main.Npx,main.Npy,main.Npz,main.Npt))
    Rstarn0 = np.rollaxis(Rstarn0,1,3)






    # now perform iteration in the time direction
    if (np.shape(f) != np.shape(main.a.a)):
      print('shape error') 
    #Jtf = MF_Jacobian_element_time(f,args_el,main)
    Jf = MF_Jacobian(f,args,main)
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
    Rstarn0 = np.rollaxis(Rstarn0,4,1)
    Rstarn0 = np.reshape(Rstarn0, (main.nvars*main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
    Rstarn0 = np.rollaxis(Rstarn0,0,8)
    f[:] = np.linalg.solve(JT + rho*ImatT,-Rstarn0 - (Jf - Jtf) + rho*f)
    f = np.rollaxis(f,7,0)
    f = np.reshape(f, (main.nvars,main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
    f = np.rollaxis(f,1,5)
    Rstarn0 = np.rollaxis(Rstarn0,7,0)
    Rstarn0 = np.reshape(Rstarn0, (main.nvars,main.order[3],main.order[0],main.order[1],main.order[2],main.Npx,main.Npy,main.Npz,main.Npt))
    Rstarn0 = np.rollaxis(Rstarn0,1,5)
#    print(np.shape(Rstarn0),np.shape(f),np.shape(main.a.a))
    NLiter += 1
    ts = time.time()
    #an[:] = main.a.a[:]
    Rstarn,dum,Rstar_globn = unsteadyResidual(an+f)
    if (Rstar_globn <= Rstar_glob):
      rho = rho*0.98
    else:
      rho = rho*3
    Rstar_glob = Rstar_globn*1.
    resid_hist = np.append(resid_hist,Rstar_glob)
    t_hist = np.append(t_hist,time.time() - tnls)
    if (main.mpi_rank == 0):
      sys.stdout.write('ADI iteration = ' + str(NLiter) + '  NL residual = ' + str(Rstar_glob) + ' relative decrease = ' + str(Rstar_glob/Rstar_glob0) + ' Solve time = ' + str(time.time() - ts)  + '  rho = ' + str(rho) + '\n')
      sys.stdout.flush()
  main.a.a[:] = an[:] + f[:]
  np.savez('resid_history',resid=resid_hist,t=t_hist)




def psuedoTimeSolver(unsteadyResidual,MF_Jacobian,main,linear_solver,sparse_quadrature,eqns):
  NLiter = 0
  tau = 0.0002
  Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a)
  Rstar_glob0 = Rstar_glob*1. 
#  rk4const = np.array([1./4,1./3,1./2,1.])
#  rk4const = np.array([0.15,0.4,1.0])
  rk4const = np.array([0.15,1.0])

  a0 = np.zeros(np.shape(main.a.a))
  Rstarn,Rn,Rstar_glob_old = unsteadyResidual(main.a.a) 
  save_freq = 10
  tnls = time.time()
  resid_hist = np.zeros(0)
  t_hist = np.zeros(0)
  while (Rstar_glob >= 1e-20 and Rstar_glob/Rstar_glob0 > 1e-8):
    NLiter += 1
    ts = time.time()
    a0[:] = main.a.a[:]
    tau = tau*np.fmin(Rstar_glob_old/Rstar_glob,1.005)

    for k in range(0,np.size(rk4const)):
      Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a) 
      #print('tau = ' + str(tau))
      main.a.a[:] = a0[:] + tau*Rstarn*rk4const[k]
  #    main.a.a[:] = a0[:] + tau*Rstarn
      Rstar_glob_old = Rstar_glob*1.
    resid_hist = np.append(resid_hist,Rstar_glob)
    t_hist = np.append(t_hist,time.time() - tnls)
    if (main.mpi_rank == 0 and NLiter%save_freq == 0):
      sys.stdout.write('NL iteration = ' + str(NLiter) + '  NL residual = ' + str(Rstar_glob) + ' relative decrease = ' + str(Rstar_glob/Rstar_glob0) + ' Solve time = ' + str(time.time() - tnls)  + '\n')
      sys.stdout.write('tau = ' + str(tau)  + '\n')
      sys.stdout.flush()
    np.savez('resid_history',resid=resid_hist,t=t_hist)




def newtonSolver_MG(unsteadyResidual,MF_Jacobian,main,linear_solver,sparse_quadrature,eqns,PC=None):
  n_levels =  main.mg_args[0]#int( np.log(np.amax(main.order))/np.log(2))  
  coarsen = np.int32(2**np.linspace(0,n_levels-1,n_levels))
  mg_classes = []
  mg_Rn = []
  mg_an = []
  #eqns2 = equations('Navier-Stokes',('roe','Inviscid'),'DNS')
  mg_b = []
  mg_e = []
  iterations = main.mg_args[1]
  omega = main.mg_args[2]
  def newtonHook(main,mg_classes,mg_Rn,mg_an):
    for i in range(0,n_levels):
      order_coarsen = np.fmax(main.order/coarsen[i],1)
#      order_coarsen[-1] = main.order[-1]
      mg_classes[i].a.a[:] = main.a.a[:,0:order_coarsen[0],0:order_coarsen[1],0:order_coarsen[2],0:order_coarsen[3]]
      mg_classes[i].getRHS(mg_classes[i],mg_classes[i],eqns)
      mg_Rn[i][:] = mg_classes[i].RHS[:]
      mg_an[i][:] = mg_classes[i].a.a[:]
      mg_e[i][:] = 0. 

  def mv_resid(MF_Jacobian,args,main,v,b):
    return b - MF_Jacobian(v,args,main)
  Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a)
  NLiter = 0
  an = np.zeros(np.shape(main.a0))
  an[:] = main.a0[:]
  Rstar_glob0 = Rstar_glob*1.
  old = np.zeros(np.shape(main.a.a))
  tnls = time.time()
  resid_hist = np.zeros(0)
  t_hist = np.zeros(0)

  mg_classes = main.mg_classes 
  mg_Rn = main.mg_Rn 
  mg_an =main.mg_an 
  mg_b = main.mg_b
  mg_e = main.mg_e

  while (Rstar_glob >= 1e-8 and Rstar_glob/Rstar_glob0 > 1e-8):
    NLiter += 1
    ts = time.time()
    old[:] = 0.
    newtonHook(main,mg_classes,mg_Rn,mg_an)
    mg_b[0][:] = -Rstarn.flatten()
    loc_tol = 1e-6
    loc_tol = 0.1*Rstar_glob/Rstar_glob0
    for i in range(0,1):
      for j in range(0,n_levels):
        MF_Jacobian_args = [mg_an[j],mg_Rn[j]]
        PC_iteration = 0
        PC_args = [omega[j],loc_tol,PC_iteration]
        mg_e[j][:] = linear_solver.solve(MF_Jacobian,mg_b[j].flatten(),np.zeros(np.size(mg_b[j])),mg_classes[j],MF_Jacobian_args,PC,PC_args,tol=loc_tol,maxiter_outer=1,maxiter=iterations[j],printnorm=0)
        #mg_e[j][:] = Jacobi(MF_Jacobian,mg_b[j].flatten(),np.zeros(np.size(mg_b[j])),PC,omega[j],mg_classes[j],MF_Jacobian_args,tol=1e-9,maxiter_outer=1,maxiter=iterations[j],printnorm=0)
        Resid  =  np.reshape( mv_resid(MF_Jacobian,MF_Jacobian_args,mg_classes[j],mg_e[j],mg_b[j].flatten()) , np.shape(mg_classes[j].a.a ) )
        if (j != n_levels-1):
          order_coarsen = np.int32(np.fmax(main.order/coarsen[j+1],1))
#          order_coarsen[-1] = main.order[-1]
          mg_b[j+1]= Resid[:,0:order_coarsen[0],0:order_coarsen[1],0:order_coarsen[2],0:order_coarsen[3]]
      for j in range(n_levels-2,-1,-1):
        order_coarsen = np.int32(np.fmax(main.order/coarsen[j+1],1))
#        order_coarsen[-1] = main.order[-1]
        etmp = np.reshape(mg_e[j][:],np.shape(mg_classes[j].a.a))
        etmp[:,0:order_coarsen[0],0:order_coarsen[1],0:order_coarsen[2],0:order_coarsen[3]] += np.reshape(mg_e[j+1],np.shape(mg_classes[j+1].a.a))
        MF_Jacobian_args = [mg_an[j],mg_Rn[j]]
        #mg_e[j][:] = Jacobi(MF_Jacobian,mg_b[j].flatten(),etmp.flatten(),PC,omega[j],mg_classes[j],MF_Jacobian_args,tol=1e-9,maxiter_outer=1,maxiter=iterations[j],printnorm=0)
        PC_iteration = 0
        PC_args = [omega[j],loc_tol,PC_iteration]
        mg_e[j][:] = linear_solver.solve(MF_Jacobian,mg_b[j].flatten(),etmp.flatten(),mg_classes[j],MF_Jacobian_args,PC,PC_args,tol=loc_tol,maxiter_outer=1,maxiter=iterations[j],printnorm=0)
        #mg_e[j][:] = etmp.flatten()
    alpha = 1. 
    main.a.a[:] = an[:] + alpha*np.reshape(mg_e[0],np.shape(main.a.a))
    Rstar_glob_p = Rstar_glob*1.
    Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a)
#    if (Rstar_glob/Rstar_glob_p <
    an[:] = main.a.a[:]
    Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a)
    resid_hist = np.append(resid_hist,Rstar_glob)
    t_hist = np.append(t_hist,time.time() - tnls)

    if (main.mpi_rank == 0):
      sys.stdout.write('NL iteration = ' + str(NLiter) + '  NL residual = ' + str(Rstar_glob) + ' relative decrease = ' + str(Rstar_glob/Rstar_glob0) + ' Solve time = ' + str(time.time() - ts)  + '\n')
      sys.stdout.flush()
  np.savez('resid_history',resid=resid_hist,t=t_hist)

