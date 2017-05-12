import numpy as np
import sys
import time
def newtonSolver(unsteadyResidual,MF_Jacobian,main,main_coarse,linear_solver,newtonHook):
  Rstarn,Rn,Rstar_glob = unsteadyResidual(main.a.a)
  NLiter = 0
  an = np.zeros(np.shape(main.a0))
  an[:] = main.a0[:]
  Rstar_glob0 = Rstar_glob*1.
  while (Rstar_glob/Rstar_glob0 >= 1e-7):
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

