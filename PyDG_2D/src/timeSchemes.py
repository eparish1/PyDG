import numpy as np
from turb_models import tauModel
def advanceSol(main,MZ,eqns,schemes):
  main.a0[:] = main.a.a[:]
  rk4const = np.array([1./4,1./3,1./2,1.])
  for i in range(0,4):
    main.getRHS(main,eqns,schemes)  ## put RHS in a array since we don't need it
    if (main.turb_str == 'DNS'):
      main.a.a[:] = main.a0 + main.dt*rk4const[i]*(main.RHS[:])
    else:
      w = main.turb_model(main,MZ,eqns,schemes)
      main.a.a[:] = main.a0 + main.dt*rk4const[i]*(main.RHS[:] + w)
  main.t += main.dt
  main.iteration += 1
