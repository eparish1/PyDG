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



def advanceSolImplicitMG(main,MZ,eqns,schemes):
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

  A = LinearOperator( (np.size(main.a.a),np.size(main.a.a)), matvec=mv )
  sol = gmres(A, main.dt*R0.flatten(), x0=old.flatten(), tol=1e-07, restart=60, maxiter=None, xtype=None, M=None, callback=printnorm,restrt=None)
#  sol = bicgstab(A, main.dt*R0.flatten(), x0=np.zeros(np.size(main.a.a)), tol=1e-07,maxiter=None, xtype=None, M=None, callback=printnorm)
  main.comm.Barrier()
  main.a.a = main.a0 + np.reshape(sol[0],np.shape(main.a.a))
  main.a0[:] = np.reshape(sol[0],np.shape(main.a.a))
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

  A = LinearOperator( (np.size(main.a.a),np.size(main.a.a)), matvec=mv )
  sol = gmres(A, main.dt*R0.flatten(), x0=old.flatten(), tol=1e-07, restart=60, maxiter=None, xtype=None, M=None, callback=printnorm,restrt=None)
#  sol = bicgstab(A, main.dt*R0.flatten(), x0=np.zeros(np.size(main.a.a)), tol=1e-07,maxiter=None, xtype=None, M=None, callback=printnorm)
  main.comm.Barrier()
  main.a.a = main.a0 + np.reshape(sol[0],np.shape(main.a.a))
  main.a0[:] = np.reshape(sol[0],np.shape(main.a.a))
  main.t += main.dt
  main.iteration += 1


def advanceSolImplicit2(main,MZ,eqns,schemes):
  old = np.zeros(np.shape(main.a0))
  old[:] = main.a0[:]
  main.a0[:] = main.a.a[:]
  main.getRHS(main,eqns,schemes)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  alpha = (2. - np.sqrt(2.))/2.
  def mv(v):
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 1.e-5
    main.a.a[:] = main.a0[:] + eps*vr
    main.getRHS(main,eqns,schemes)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Av = vr - main.dt*alpha*(R1 - R0)/eps
    main.a.a[:] = main.a0[:] 
    return Av.flatten()
  def printnorm(r):
    print('GMRES norm = ' + str(np.linalg.norm(r)))

  A = LinearOperator( (np.size(main.a.a),np.size(main.a.a)), matvec=mv )
  #stage 1
  sol = gmres(A, main.dt*alpha*R0.flatten(), x0=np.zeros(np.size(R0)), tol=1e-07, restart=60, maxiter=None, xtype=None, M=None, callback=printnorm,restrt=None)
  main.a.a = main.a0 + np.reshape(sol[0],np.shape(main.a.a))
  main.getRHS(main,eqns,schemes)
  R2 = main.dt*( (1 - alpha)*main.RHS + alpha*R0).flatten()
  sol = gmres(A, R2, x0=np.zeros(np.size(R2)), tol=1e-07, restart=60, maxiter=None, xtype=None, M=None, callback=printnorm,restrt=None)
  main.a.a = main.a0 + np.reshape(sol[0],np.shape(main.a.a))
  main.t += main.dt
  main.iteration += 1

def advanceSolImplicit4(main,MZ,eqns,schemes):
  old = np.zeros(np.shape(main.a0))
  old[:] = main.a0[:]
  main.a0[:] = main.a.a[:]
  main.getRHS(main,eqns,schemes)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  gam = 9./40.
  c2 = 7./13.
  c3 = 11./15.
  b2 = -(-2. + 3.*c3 + 9.*gam - 12.*c3*gam - 6.*gam**2 + 6.*c3*gam**2)/(6.*(c2 - c3)*(c2 - gam))
  b3 =  (-2. + 3.*c2 + 9.*gam - 12.*c2*gam - 6.*gam**2 + 6.*c2*gam**2)/(6.*(c2 - c3)*(c3 - gam))
  a32 = -(c2 - c3)*(c3 - gam)*(-1. + 9.*gam - 18.*gam**2 + 6.*gam**3)/ \
       ( (c2 - gam)*(-2. + 3.*c2 + 9.*gam - 12.*c2*gam - 6.*gam**2 + 6.*c2*gam**2) )
  def mv(v):
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 1.e-5
    main.a.a[:] = main.a0[:] + eps*vr
    main.getRHS(main,eqns,schemes)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Av = vr - main.dt*gam*(R1 - R0)/eps
    main.a.a[:] = main.a0[:] 
    return Av.flatten()

  def printnorm(r):
    #pass
    print('GMRES norm = ' + str(np.linalg.norm(r)))
  A = LinearOperator( (np.size(main.a.a),np.size(main.a.a)), matvec=mv )
  #stage 1
  sol = gmres(A, main.dt*gam*R0.flatten(), x0=np.zeros(np.size(R0)), tol=1e-07, restart=200, maxiter=None, xtype=None, M=None, callback=printnorm,restrt=None)
  main.a.a = main.a0 + np.reshape(sol[0],np.shape(main.a.a))
  main.getRHS(main,eqns,schemes)
  R1 = np.zeros(np.shape(main.RHS))
  R1[:] = main.RHS[:]
  #stage 2
  RHS_stage2 = main.dt*( (c2 - gam)*R1 + gam*R0).flatten()
  sol = gmres(A, RHS_stage2, x0=sol[0], tol=1e-07, restart=200, maxiter=None, xtype=None, M=None, callback=printnorm,restrt=None)
  main.a.a = main.a0 + np.reshape(sol[0],np.shape(main.a.a))
  main.getRHS(main,eqns,schemes)
  R2 = np.zeros(np.shape(main.RHS))
  R2[:] = main.RHS[:]
  #stage 3
  RHS_stage3 = main.dt*( (c3 - a32 - gam)*R1 + a32*R2 + gam*R0).flatten()
  sol = gmres(A, RHS_stage3, x0=sol[0], tol=1e-07, restart=200, maxiter=None, xtype=None, M=None, callback=printnorm,restrt=None)
  main.a.a = main.a0 + np.reshape(sol[0],np.shape(main.a.a))
  main.getRHS(main,eqns,schemes)
  R3 = np.zeros(np.shape(main.RHS))
  R3[:] = main.RHS[:]
  #stage 4
  RHS_stage4 = main.dt*( (1. - b2 - b3 - gam)*R1 + b2*R2 + b3*R3 + gam*R0).flatten()
  sol = gmres(A, RHS_stage4, x0=sol[0], tol=1e-07, restart=200, maxiter=None, xtype=None, M=None, callback=printnorm,restrt=None)
  main.a.a = main.a0 + np.reshape(sol[0],np.shape(main.a.a))
  main.t += main.dt
  main.iteration += 1

