import numpy as np
from turb_models import tauModel
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import lgmres
def advanceSol(main,MZ,eqns,schemes):
  main.a0[:] = main.a.a[:]
  rk4const = np.array([1./4,1./3,1./2,1.])
  for i in range(0,4):
    main.rkstage = i
    if (main.turb_str == 'DNS'):
      main.getRHS(main,eqns,schemes)  ## put RHS in a array since we don't need it
      main.a.a[:] = main.a0 + main.dt*rk4const[i]*(main.RHS[:])
    else:
      main.RHS[:],w = main.turb_model(main,MZ,eqns,schemes)
      #print(np.linalg.norm(main.w))
      main.a.a[:] = main.a0 + main.dt*rk4const[i]*(main.RHS[:] + w)
  main.t += main.dt
  main.iteration += 1


def advanceSolImplicit(main,MZ,eqns,schemes):
  old = np.zeros(np.shape(main.a0))
  old[:] = main.a0[:]
  main.a0[:] = main.a.a[:]
  main.getRHS(main,eqns,schemes)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  def mv(v):
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 1.e-5
    main.a.a[:] = main.a0[:] + eps*vr
    main.getRHS(main,eqns,schemes)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Av = vr - main.dt/2.*(R1 - R0)/eps
    main.a.a[:] = main.a0[:] 
    return Av.flatten()
  def printnorm(r):
    print('GMRES norm = ' + str(np.linalg.norm(r)))
    r = 1

  A = LinearOperator( (np.size(main.a.a),np.size(main.a.a)), matvec=mv )
  sol = gmres(A, main.dt*R0.flatten(), x0=old.flatten(), tol=1e-07, restart=10, maxiter=None, xtype=None, M=None, callback=printnorm,restrt=None)
#  sol = bicgstab(A, main.dt*R0.flatten(), x0=np.zeros(np.size(main.a.a)), tol=1e-07,maxiter=None, xtype=None, M=None, callback=printnorm)
  print(str(main.mpi_rank) + 'here')
  main.comm.Barrier()
  main.a.a = main.a0 + np.reshape(sol[0],np.shape(main.a.a))
  main.a0[:] = np.reshape(sol[0],np.shape(main.a.a))
  main.t += main.dt
  main.iteration += 1


