import numpy as np

def advanceSol(main,eqns,schemes):
  main.a0[:] = main.a.a[:]
  rk4const = np.array([1./4,1./3,1./2,1.])
  for i in range(0,4):
    main.getRHS(main,eqns,schemes)  ## put RHS in a array since we don't need it
    main.a.a[:] = main.a0 + main.dt*rk4const[i]*main.RHS[:]
  main.t += main.dt
  main.iteration += 1
