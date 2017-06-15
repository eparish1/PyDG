import numpy as np

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
