import numpy as np
import sys
import time
from init_Classes import variables
def newtonSolver(unsteadyResidual,MF_Jacobian,main,linear_solver,sparse_quadrature,eqns):
  if (sparse_quadrature):
    coarsen = 2
    main_coarse = variables(main.Nel,main.order,main.quadpoints/(coarsen),eqns,main.mu,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy)
    main_coarse.a.a[:] = main.a.a[:]
    def newtonHook(main_coarse,main,Rn):
      main_coarse.a.a[:] = main.a.a[:]
      main_coarse.getRHS(main_coarse,eqns)
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
  while (Rstar_glob >= 1e-8 and Rstar_glob/Rstar_glob0 > 1e-8):
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
  coarsen = 2
  coarsen2 = 4
  main_coarse = variables(main.Nel,main.order/coarsen,main.quadpoints/(2*coarsen),eqns,main.mu,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy)
  main_qc = variables(main.Nel,main.order,main.quadpoints/2,eqns,main.mu,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy)
  main_coarse2 = variables(main.Nel,main.order/coarsen2,main.quadpoints/(2*coarsen2),eqns,main.mu,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy)

  Rnc = np.zeros(np.shape(main_coarse.RHS))
  Rn_qc = np.zeros(np.shape(main_qc.RHS))
  Rnc2 = np.zeros(np.shape(main_coarse2.RHS))
  anc = np.zeros(np.shape(main_coarse.a.a))
  anc2 = np.zeros(np.shape(main_coarse2.a.a))

  def newtonHook(main,main_coarse,main_qc):
    main_coarse.a.a[:] = main.a.a[:,0:main.order/coarsen,0:main.order/coarsen,0:main.order/coarsen]
    main_coarse.getRHS(main_coarse,eqns)
    Rnc[:] = main_coarse.RHS[:]
 
    main_qc.a.a[:] = main.a.a[:]
    main_qc.getRHS(main_qc,eqns)
    Rn_qc[:] = main_qc.RHS[:]
    anc[:] = main_coarse.a.a[:]
 
    main_coarse2.a.a[:] = main.a.a[:,0:main.order/coarsen2,0:main.order/coarsen2,0:main.order/coarsen2]
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
      R_coarse = R[:,0:main.order/coarsen,0:main.order/coarsen,0:main.order/coarsen]
      # solve for the error on the coarse mesh
      MF_Jacobian_argsc = [anc,Rnc]
      e = linear_solver.solve(MF_Jacobian,R_coarse.flatten(), np.zeros(np.shape(R_coarse.flatten())),main_coarse,MF_Jacobian_argsc,tol=1e-6,maxiter_outer=1,maxiter=10,printnorm=0)
      R1 =  np.reshape( mv_resid(MF_Jacobian,MF_Jacobian_argsc,main_coarse,e,R_coarse.flatten()) , np.shape(main_coarse.a.a ) )
      R_coarse2 = R1[:,0:main.order/coarsen2,0:main.order/coarsen2,0:main.order/coarsen2]
      ##
      MF_Jacobian_argsc2 = [anc2,Rnc2]
      e2 = linear_solver.solve(MF_Jacobian,R_coarse2.flatten(),np.zeros(np.size(R_coarse2)),main_coarse2,MF_Jacobian_argsc2,tol=1e-5,maxiter_outer=1,maxiter=20,printnorm=0)
      ###
      etmp = np.reshape(e,np.shape(main_coarse.a.a))
      etmp[:,0:main.order/coarsen2,0:main.order/coarsen2,0:main.order/coarsen2] += np.reshape(e2,np.shape(main_coarse2.a.a))
      e = linear_solver.solve(MF_Jacobian,R_coarse.flatten(),etmp.flatten(),main_coarse,MF_Jacobian_argsc,tol=1e-6,maxiter_outer=1,maxiter=20,printnorm=0)

      old[:] = np.reshape(sol,np.shape(main.a.a))
      old[:,0:main.order/coarsen,0:main.order/coarsen,0:main.order/coarsen] += np.reshape( e , np.shape(main_coarse.a.a) )

    # Run the final iterations on the fine mesh
    #sol = linear_solver.solve(MF_Jacobian, -Rstarn.flatten(), np.zeros(np.size(main.a.a)),main_qc,MF_Jacobian_args, linear_solver.tol,linear_solver.maxiter_outer,linear_solver.maxiter,linear_solver.printnorm)
    sol = linear_solver.solve(MF_Jacobian, -Rstarn.flatten(), old.flatten(),main_qc,MF_Jacobian_args, 1e-5,1,30,linear_solver.printnorm)

    main.a.a[:] = an[:] + np.reshape(sol,np.shape(main.a.a))
    an[:] = main.a.a[:]
    Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a)
    if (main.mpi_rank == 0):
      sys.stdout.write('NL iteration = ' + str(NLiter) + '  NL residual = ' + str(Rstar_glob) + ' relative decrease = ' + str(Rstar_glob/Rstar_glob0) + ' Solve time = ' + str(time.time() - ts)  + '\n')
      sys.stdout.flush()

