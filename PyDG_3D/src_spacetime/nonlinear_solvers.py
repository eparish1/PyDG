import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from init_Classes import variables,equations
def newtonSolver(unsteadyResidual,MF_Jacobian,main,linear_solver,sparse_quadrature,eqns):
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
  while (Rstar_glob >= 1e-7 and Rstar_glob/Rstar_glob0 > 1e-7):
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
    sol = linear_solver.solve(MF_Jacobian, -Rstarn.flatten(), old.flatten(),main_coarse,MF_Jacobian_args,np.fmin(Rstar_glob,0.1),linear_solver.maxiter_outer,20,False)
    main.a.a[:] = an[:] + 1.0*np.reshape(sol,np.shape(main.a.a))
    an[:] = main.a.a[:]
    Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a)
    resid_hist = np.append(resid_hist,Rstar_glob)
    t_hist = np.append(t_hist,time.time() - tnls)
    if (main.mpi_rank == 0):
      sys.stdout.write('NL iteration = ' + str(NLiter) + '  NL residual = ' + str(Rstar_glob) + ' relative decrease = ' + str(Rstar_glob/Rstar_glob0) + ' Solve time = ' + str(time.time() - ts)  + '\n')
      #print(np.linalg.norm(Rstarn[0]),np.linalg.norm(Rstarn[-1]))
    
      sys.stdout.flush()
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




def newtonSolver_MG(unsteadyResidual,MF_Jacobian,main,linear_solver,sparse_quadrature,eqns):
  n_levels =  int( np.log(np.amax(main.order))/np.log(2))  
  coarsen = np.int32(2**np.linspace(0,n_levels-1,n_levels))
  mg_classes = []
  mg_Rn = []
  mg_an = []
  #eqns2 = equations('Navier-Stokes',('roe','Inviscid'),'DNS')
  mg_b = []
  mg_e = []
  for i in range(0,n_levels):
    order_coarsen = np.int32(np.fmax(main.order/coarsen[i],1))
    quadpoints_coarsen = np.int32(np.fmax(main.quadpoints/(coarsen[i]),1))
    order_coarsen[-1] = main.order[-1]
    quadpoints_coarsen[-1] = main.quadpoints[-1]
    mg_classes.append( variables(main.Nel,order_coarsen,quadpoints_coarsen,eqns,main.mus,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy,main.BCs,main.fsource,main.source_mag,main.shock_capturing) )
    mg_classes[i].basis = main.basis
    mg_Rn.append( np.zeros(np.shape(mg_classes[i].RHS)) )
    mg_an.append( np.zeros(np.shape(mg_classes[i].a.a) ) )
    mg_b.append( np.zeros(np.size(mg_classes[i].RHS)) )
    mg_e.append(  np.zeros(np.size(mg_classes[i].RHS)) )
  def newtonHook(main,mg_classes,mg_Rn,mg_an):
    for i in range(0,n_levels):
      order_coarsen = np.fmax(main.order/coarsen[i],1)
      order_coarsen[-1] = main.order[-1]
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

  while (Rstar_glob >= 1e-9 and Rstar_glob/Rstar_glob0 > 1e-9):
    NLiter += 1
    ts = time.time()
    old[:] = 0.
    newtonHook(main,mg_classes,mg_Rn,mg_an)
    mg_b[0][:] = -Rstarn.flatten()
    for i in range(0,1):
      for j in range(0,n_levels):
        MF_Jacobian_args = [mg_an[j],mg_Rn[j]]
        mg_e[j][:] = linear_solver.solve(MF_Jacobian,mg_b[j].flatten(),np.zeros(np.size(mg_b[j])),mg_classes[j],MF_Jacobian_args,tol=1e-5,maxiter_outer=1,maxiter=10,printnorm=0)
        Resid  =  np.reshape( mv_resid(MF_Jacobian,MF_Jacobian_args,mg_classes[j],mg_e[j],mg_b[j].flatten()) , np.shape(mg_classes[j].a.a ) )
        if (j != n_levels-1):
          order_coarsen = np.int32(np.fmax(main.order/coarsen[j+1],1))
          order_coarsen[-1] = main.order[-1]
          mg_b[j+1]= Resid[:,0:order_coarsen[0],0:order_coarsen[1],0:order_coarsen[2],0:order_coarsen[3]]
      for j in range(n_levels-2,-1,-1):
        order_coarsen = np.int32(np.fmax(main.order/coarsen[j+1],1))
        order_coarsen[-1] = main.order[-1]
        etmp = np.reshape(mg_e[j][:],np.shape(mg_classes[j].a.a))
        etmp[:,0:order_coarsen[0],0:order_coarsen[1],0:order_coarsen[2],0:order_coarsen[3]] += np.reshape(mg_e[j+1],np.shape(mg_classes[j+1].a.a))
        MF_Jacobian_args = [mg_an[j],mg_Rn[j]]
        mg_e[j][:] = linear_solver.solve(MF_Jacobian,mg_b[j].flatten(),etmp.flatten(),mg_classes[j],MF_Jacobian_args,tol=1e-6,maxiter_outer=1,maxiter=10,printnorm=0)
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



def newtonSolver_PC2(unsteadyResidual,MF_Jacobian,main,linear_solver,sparse_quadrature,eqns):
  if (sparse_quadrature):
    coarsen = 2
    quadpoints_coarsen = np.fmax(main.quadpoints/(coarsen),1)
    quadpoints_coarsen[-1] = main.quadpoints[-1]
    main_coarse = variables(main.Nel,main.order,quadpoints_coarsen,eqns,main.mus,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy,main.BCs,main.fsource,main.source_mag,main.shock_capturing)
    main_coarse.basis = main.basis
    main_coarse.a.a[:] = main.a.a[:]
    def newtonHook(main_coarse,main,Rn):
      main_coarse.a.a[:] = main.a.a[:]
      main_coarse.getRHS(main_coarse,main_coarse,eqns)
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

  def Minv(v,main,MF_Jacobian,MF_Jacobian_args,k):
    sol = linear_solver.solvePC(MF_Jacobian,v.flatten()*1.,v.flatten()*0.,main,MF_Jacobian_args, 1e-6,linear_solver.maxiter_outer,5,False)
    return sol.flatten()

  while (Rstar_glob >= 1e-20 and Rstar_glob/Rstar_glob0 > 1e-9):
    NLiter += 1
    ts = time.time()
    newtonHook(main_coarse,main,Rn)
    MF_Jacobian_args = [an,Rn]
    delta = 1
    sol = linear_solver.solve(MF_Jacobian,-Rstarn.flatten(), np.zeros(np.size(main_coarse.a.a)),main_coarse,MF_Jacobian_args,Minv,linear_solver.tol,linear_solver.maxiter_outer,10,False)

    main.a.a[:] = an[:] + np.reshape(sol,np.shape(main.a.a))
    an[:] = main.a.a[:]

    Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a)
    if (main.mpi_rank == 0):
      sys.stdout.write('NL iteration = ' + str(NLiter) + '  NL residual = ' + str(Rstar_glob) + ' relative decrease = ' + str(Rstar_glob/Rstar_glob0) + ' Solve time = ' + str(time.time() - ts)  + '\n')
      sys.stdout.flush()







def newtonSolver_PC8(unsteadyResidual,MF_Jacobian,main,linear_solver,sparse_quadrature,eqns):
  order_coarsen = np.fmax(main.order/2,1)
#  order_coarsen[-1] = main.order[-1]
  quadpoints_coarsen = np.fmax(main.quadpoints/2,1)
  quadpoints_coarsen[-1] = main.quadpoints[-1]
  main_coarse = variables(main.Nel,order_coarsen,quadpoints_coarsen,eqns,main.mus,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy,main.BCs,main.fsource,main.source_mag,main.shock_capturing)
  main_coarse.basis = main.basis
  main_coarse.a.a[:] = main.a.a[:,0:main_coarse.order[0],0:main_coarse.order[1],0:main_coarse.order[2],0:main_coarse.order[3] ]
  def newtonHook(main,main_coarse,Rn,Rnc):
    eqns.getRHS(main,main,eqns)
    Rn[:] = main.RHS[:]
    eqns.getRHS(main_coarse,main_coarse,eqns)
    Rnc[:] = main_coarse.RHS[:]

  def Minv(v,main,main_coarse,MF_Jacobian_args,MF_Jacobian_args2):
    def mv_resid(MF_Jacobian,args,main,v,b):
      return b - MF_Jacobian(v,args,main)
    coarse_order = np.shape(main_coarse.a.a)[1:5]
    old = np.zeros(np.shape(main.a.a))
    for i in range(0,1):
      # solve on the fine mesh
      sol = linear_solver.solvePC(MF_Jacobian ,v.flatten()*1.,old.flatten(),main,MF_Jacobian_args2, 1e-6,linear_solver.maxiter_outer,7,False)
  
      # restrict
      R =  np.reshape( mv_resid(MF_Jacobian,MF_Jacobian_args2,main,sol,v.flatten()) , np.shape(main.a.a ) )
      R_coarse = R[:,0:order_coarsen[0],0:order_coarsen[1],0:order_coarsen[2],0:order_coarsen[3]]
      # solve for the error on the coarse mesh
      e = linear_solver.solvePC(MF_Jacobian,R_coarse.flatten(), np.zeros(np.shape(R_coarse.flatten())),main_coarse,MF_Jacobian_args,1e-6,linear_solver.maxiter_outer,1,False)
#  
      old = np.zeros(np.shape(main.a.a))
      old[:] = np.reshape(sol,np.shape(main.a.a))
      old[:,0:order_coarsen[0],0:order_coarsen[1],0:order_coarsen[2],0:order_coarsen[3]] += np.reshape( e , np.shape(main_coarse.a.a) )
  
      # Run the final iterations on the fine mesh
      sol = linear_solver.solvePC(MF_Jacobian, v.flatten(), old.flatten(),main,MF_Jacobian_args2, 1e-6,linear_solver.maxiter_outer,7,False)

#    Minv_v = np.zeros(np.shape(main.a.a))
#    Minv_v[:] = np.reshape(v,np.shape(main.a.a))
#    tmp_coarse = Minv_v[:,0:coarse_order[0],0:coarse_order[1],0:coarse_order[2],0:coarse_order[3]]
#    Minv_v2 = np.zeros(np.shape(main.a.a))
#    Minv_v2[:] = Minv_v[:]
#    tmp_coarse2 = np.zeros(np.shape(tmp_coarse))
#    tmp_coarse2[:] = tmp_coarse[:]
#    tmp_coarse = linear_solver.solvePC(MF_Jacobian, tmp_coarse2.flatten(), tmp_coarse2.flatten()*0.,main_coarse,MF_Jacobian_args, 1e-6,linear_solver.maxiter_outer,10,False)
#    Minv_v[:,0:coarse_order[0],0:coarse_order[1],0:coarse_order[2],0:coarse_order[3]] = np.reshape(tmp_coarse[:],np.shape(main_coarse.a.a))
#    sol = linear_solver.solvePC(MF_Jacobian, v2.flatten(), Minv_v.flatten()*1.,main,MF_Jacobian_args2, 1e-6,linear_solver.maxiter_outer,20,False)
#
# 
#    plt.plot( np.reshape(tmp_coarse[:],np.shape(main_coarse.a.a))[0,:,0,0,0,0,0,0,0])
#    plt.plot( np.reshape(tmp_coarse1[:],np.shape(main.a.a))[0,:,0,0,0,0,0,0,0])
#    plt.pause(0.01)
#    plt.clf()

    return sol.flatten()


  Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a)
  NLiter = 0
  an = np.zeros(np.shape(main.a0))
  an[:] = main.a0[:]
  Rnc = np.zeros(np.shape(main_coarse.a0))
  anc = np.zeros(np.shape(main_coarse.a0))
  anc[:] = main.a0[:,0:main_coarse.order[0],0:main_coarse.order[1],0:main_coarse.order[2],0:main_coarse.order[3] ]
  Rstar_glob0 = Rstar_glob*1.
  while (Rstar_glob >= 1e-20 and Rstar_glob/Rstar_glob0 > 1e-9):
    NLiter += 1
    ts = time.time()
    newtonHook(main,main_coarse,Rn,Rnc)
    MF_Jacobian_args = [an,Rn]
    MF_Jacobian_args_coarse = [anc,Rnc]
    delta = 1

    sol = linear_solver.solve(MF_Jacobian, -Rstarn.flatten(), np.zeros(np.size(main.a.a)),main,MF_Jacobian_args,main_coarse,MF_Jacobian_args_coarse,Minv,linear_solver.tol,linear_solver.maxiter_outer,10,False)
    main.a.a[:] = an[:] + np.reshape(sol,np.shape(main.a.a))
    an[:] = main.a.a[:]
    anc[:] = an[:,0:main_coarse.order[0],0:main_coarse.order[1],0:main_coarse.order[2],0:main_coarse.order[3] ]
    main_coarse.a.a[:] = main.a.a[:,0:main_coarse.order[0],0:main_coarse.order[1],0:main_coarse.order[2],0:main_coarse.order[3] ]

    Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a)
    if (main.mpi_rank == 0):
      sys.stdout.write('NL iteration = ' + str(NLiter) + '  NL residual = ' + str(Rstar_glob) + ' relative decrease = ' + str(Rstar_glob/Rstar_glob0) + ' Solve time = ' + str(time.time() - ts)  + '\n')
      sys.stdout.flush()


#### OUTDATED. USE FOR VALIDATION
def newtonSolver_MG2(unsteadyResidual,MF_Jacobian,main,linear_solver,sparse_quadrature,eqns):
  coarsen = 2
  coarsen2 = 4
  eqns2 = equations('Navier-Stokes',('roe','Inviscid'))
  main_coarse = variables(main.Nel,main.order/coarsen,main.quadpoints/(2*coarsen),eqns2,main.mu,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy)
  main_qc = variables(main.Nel,main.order,main.quadpoints/2,eqns2,main.mu,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy)
  main_coarse2 = variables(main.Nel,main.order/coarsen2,main.quadpoints/(2*coarsen2),eqns2,main.mu,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy)

  Rnc = np.zeros(np.shape(main_coarse.RHS))
  Rn_qc = np.zeros(np.shape(main_qc.RHS))
  Rnc2 = np.zeros(np.shape(main_coarse2.RHS))
  anc = np.zeros(np.shape(main_coarse.a.a))
  anc2 = np.zeros(np.shape(main_coarse2.a.a))

  def newtonHook(main,main_coarse,main_qc):
    main_coarse.a.a[:] = main.a.a[:,0:main.order[0]/coarsen,0:main.order[1]/coarsen,0:main.order[2]/coarsen]
    main_coarse.getRHS(main_coarse,eqns)
    Rnc[:] = main_coarse.RHS[:]
 
    main_qc.a.a[:] = main.a.a[:]
    main_qc.getRHS(main_qc,eqns)
    Rn_qc[:] = main_qc.RHS[:]
    anc[:] = main_coarse.a.a[:]
 
    main_coarse2.a.a[:] = main.a.a[:,0:main.order[0]/coarsen2,0:main.order[1]/coarsen2,0:main.order[2]/coarsen2]
    main_coarse2.getRHS(main_coarse2,eqns)
    Rnc2[:] = main_coarse2.RHS[:]
    anc2[:] = main_coarse2.a.a[:]


  def mv_resid(MF_Jacobian,args,main,v,b):
    return b - MF_Jacobian(v,args,main)

  Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a)
  NLiter = 0
  an = np.zeros(np.shape(main.a0))
  an[:] = main.a0[:]
  Rstar_glob0 = Rstar_glob*1.
  old = np.zeros(np.shape(main.a.a))
  while (Rstar_glob >= 1e-8 and Rstar_glob/Rstar_glob0 > 1e-8):
    NLiter += 1
    ts = time.time()
    old[:] = 0.
    newtonHook(main,main_coarse,main_qc)
    for i in range(0,1):
      #run the solution on the fine mesh (probably with sparse quadrature)
      MF_Jacobian_args = [an,Rn_qc]
      sol = linear_solver.solve(MF_Jacobian,-Rstarn.flatten(), old.flatten(),main_qc,MF_Jacobian_args,tol=1e-5,maxiter_outer=1,maxiter=10,printnorm=0)
      # restrict
      R =  np.reshape( mv_resid(MF_Jacobian,MF_Jacobian_args,main_qc,sol,-Rstarn.flatten()) , np.shape(main.a.a ) )
      R_coarse = R[:,0:main.order[0]/coarsen,0:main.order[1]/coarsen,0:main.order[2]/coarsen]
      # solve for the error on the coarse mesh
      MF_Jacobian_argsc = [anc,Rnc]
      e = linear_solver.solve(MF_Jacobian,R_coarse.flatten(), np.zeros(np.shape(R_coarse.flatten())),main_coarse,MF_Jacobian_argsc,tol=1e-6,maxiter_outer=1,maxiter=30,printnorm=0)
      R1 =  np.reshape( mv_resid(MF_Jacobian,MF_Jacobian_argsc,main_coarse,e,R_coarse.flatten()) , np.shape(main_coarse.a.a ) )
      R_coarse2 = R1[:,0:main.order[0]/coarsen2,0:main.order[1]/coarsen2,0:main.order[2]/coarsen2]
      ##
      MF_Jacobian_argsc2 = [anc2,Rnc2]
      e2 = linear_solver.solve(MF_Jacobian,R_coarse2.flatten(),np.zeros(np.size(R_coarse2)),main_coarse2,MF_Jacobian_argsc2,tol=1e-5,maxiter_outer=1,maxiter=30,printnorm=0)
      ###
      etmp = np.reshape(e,np.shape(main_coarse.a.a))
      etmp[:,0:main.order[0]/coarsen2,0:main.order[1]/coarsen2,0:main.order[2]/coarsen2] += np.reshape(e2,np.shape(main_coarse2.a.a))
      e = linear_solver.solve(MF_Jacobian,R_coarse.flatten(),etmp.flatten(),main_coarse,MF_Jacobian_argsc,tol=1e-6,maxiter_outer=1,maxiter=30,printnorm=0)

      old[:] = np.reshape(sol,np.shape(main.a.a))
      old[:,0:main.order[0]/coarsen,0:main.order[1]/coarsen,0:main.order[2]/coarsen] += np.reshape( e , np.shape(main_coarse.a.a) )

    # Run the final iterations on the fine mesh
    #sol = linear_solver.solve(MF_Jacobian, -Rstarn.flatten(), np.zeros(np.size(main.a.a)),main_qc,MF_Jacobian_args, linear_solver.tol,linear_solver.maxiter_outer,linear_solver.maxiter,linear_solver.printnorm)
    sol = linear_solver.solve(MF_Jacobian, -Rstarn.flatten(), old.flatten(),main_qc,MF_Jacobian_args, 1e-5,1,30,linear_solver.printnorm)

    main.a.a[:] = an[:] + np.reshape(sol,np.shape(main.a.a))
    an[:] = main.a.a[:]
    Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a)
    if (main.mpi_rank == 0):
      sys.stdout.write('NL iteration = ' + str(NLiter) + '  NL residual = ' + str(Rstar_glob) + ' relative decrease = ' + str(Rstar_glob/Rstar_glob0) + ' Solve time = ' + str(time.time() - ts)  + '\n')
      sys.stdout.flush()


