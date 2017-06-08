import numpy as np
import sys
import time
from init_Classes import variables,equations
def newtonSolver(unsteadyResidual,MF_Jacobian,main,linear_solver,sparse_quadrature,eqns):
  if (sparse_quadrature):
    coarsen = 2
    main_coarse = variables(main.Nel,main.order,main.quadpoints/(coarsen),eqns,main.mus,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy,main.BCs,main.source,main.source_mag)
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
  while (Rstar_glob >= 1e-10 and Rstar_glob/Rstar_glob0 > 1e-9):
    NLiter += 1
    ts = time.time()
    newtonHook(main_coarse,main,Rn)
    MF_Jacobian_args = [an,Rn]
    sol = linear_solver.solve(MF_Jacobian, -Rstarn.flatten(), np.zeros(np.size(main.a.a)),main_coarse,MF_Jacobian_args, linear_solver.tol,linear_solver.maxiter_outer,linear_solver.maxiter,linear_solver.printnorm)
    main.a.a[:] = an[:] + np.reshape(sol,np.shape(main.a.a))
    an[:] = main.a.a[:]
    Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a)
    if (main.mpi_rank == 0):
      sys.stdout.write('NL iteration = ' + str(NLiter) + '  NL residual = ' + str(Rstar_glob) + ' relative decrease = ' + str(Rstar_glob/Rstar_glob0) + ' Solve time = ' + str(time.time() - ts)  + '\n')
      sys.stdout.flush()



def newtonSolver_MG(unsteadyResidual,MF_Jacobian,main,linear_solver,sparse_quadrature,eqns):
  n_levels = int(np.log(np.amin(main.order))/np.log(2)) 
  coarsen = np.int32(2**np.linspace(0,n_levels-1,n_levels))
  mg_classes = []
  mg_Rn = []
  mg_an = []
  #eqns2 = equations('Navier-Stokes',('roe','Inviscid'),'DNS')
  mg_b = []
  mg_e = []
  for i in range(0,n_levels):
    mg_classes.append( variables(main.Nel,main.order/coarsen[i],main.quadpoints/(2*coarsen[i]),eqns,main.mus,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy,main.BCs,main.source,main.source_mag,main.shock_capturing) )
    mg_classes[i].basis = main.basis
    mg_Rn.append( np.zeros(np.shape(mg_classes[i].RHS)) )
    mg_an.append( np.zeros(np.shape(mg_classes[i].a.a) ) )
    mg_b.append( np.zeros(np.size(mg_classes[i].RHS)) )
    mg_e.append(  np.zeros(np.size(mg_classes[i].RHS)) )
  print(n_levels)
  def newtonHook(main,mg_classes,mg_Rn,mg_an):
    for i in range(0,n_levels):
      mg_classes[i].a.a[:] = main.a.a[:,0:main.order[0]/coarsen[i],0:main.order[1]/coarsen[i],0:main.order[2]/coarsen[i]]
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

  while (Rstar_glob >= 1e-8 and Rstar_glob/Rstar_glob0 > 1e-8):
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
          mg_b[j+1]= Resid[:,0:main.order[0]/coarsen[j+1],0:main.order[1]/coarsen[j+1],0:main.order[2]/coarsen[j+1]]
      for j in range(n_levels-2,-1,-1):
        etmp = np.reshape(mg_e[j][:],np.shape(mg_classes[j].a.a))
        etmp[:,0:main.order[0]/coarsen[j+1],0:main.order[1]/coarsen[j+1],0:main.order[2]/coarsen[j+1]] += np.reshape(mg_e[j+1],np.shape(mg_classes[j+1].a.a))
        MF_Jacobian_args = [mg_an[j],mg_Rn[j]]
        mg_e[j][:] = linear_solver.solve(MF_Jacobian,mg_b[j].flatten(),etmp.flatten(),mg_classes[j],MF_Jacobian_args,tol=1e-6,maxiter_outer=1,maxiter=10,printnorm=0)

    main.a.a[:] = an[:] + np.reshape(mg_e[0],np.shape(main.a.a))
    an[:] = main.a.a[:]
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


