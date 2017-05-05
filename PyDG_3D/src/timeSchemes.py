import numpy as np
from turb_models import tauModel
from init_Classes import variables
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres,bicgstab
from scipy.sparse.linalg import lgmres
from myGMRES import GMRes
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
  main_coarse = variables(main.Nel,main.order/2,main.quadpoints,eqns,main.mu,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,schemes,'DNS',main.procx,main.procy)
  old = np.zeros(np.shape(main.a0))
  old[:] = main.a0[:]
  main.a0[:] = main.a.a[:]
  main_coarse.a.a[:] = main.a.a[:,0:main.order/2,0:main.order/2,0:main.order/2]
  main_coarse.a0[:] = main.a.a[:,0:main.order/2,0:main.order/2,0:main.order/2]
  main.getRHS(main,eqns,schemes)
  main_coarse.getRHS(main_coarse,eqns,schemes)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  R0_coarse = np.zeros(np.shape(main_coarse.RHS))
  R0_coarse[:] = main_coarse.RHS[:]
  print(np.linalg.norm(main_coarse.a.u))
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

  def mv_resid(v,b):
    return mv(v) - b

  def mv_coarse(v):
    vr = np.reshape(v,np.shape(main_coarse.a.a))
    eps = 1.e-5
    main_coarse.a.a[:] = main_coarse.a0[:] + eps*vr
    main_coarse.getRHS(main_coarse,eqns,schemes)
    R1 = np.zeros(np.shape(main_coarse.RHS))
    R1[:] = main_coarse.RHS[:]
    Av = vr - main_coarse.dt/2.*(R1 - R0_coarse)/eps
    main_coarse.a.a[:] = main_coarse.a0[:] 
    return Av.flatten()


  def printnorm(r):
    print('GMRES norm = ' + str(np.linalg.norm(r)))

  A = LinearOperator( (np.size(main.a.a),np.size(main.a.a)), matvec=mv )
  A_coarse = LinearOperator( (np.size(main_coarse.a.a),np.size(main_coarse.a.a)), matvec=mv_coarse )
  ## preconditioner type thing
  print('Running Coarse Mesh')
  print(np.linalg.norm(R0[:,0:main.order/2,0:main.order/2,0:main.order/2] - R0_coarse) )
  e = gmres(A_coarse,main.dt*R0_coarse.flatten(), np.zeros(np.shape(R0_coarse.flatten())), tol=1e-05, restart=60, maxiter=60, xtype=None, M=None, callback=printnorm,restrt=None)
  x0 = np.zeros(np.shape(main.a.a))
  x0[:,0:main.order/2,0:main.order/2,0:main.order/2] += np.reshape( e[0] , np.shape(main_coarse.a.a) )
  sol = gmres(A, main.dt*R0.flatten(), x0=x0.flatten(), tol=1e-04, restart=60, maxiter=None, xtype=None, M=None, callback=printnorm,restrt=None)
  ## multigrid
#  sol = gmres(A, main.dt*R0.flatten(), x0=old.flatten(), tol=2e-02, restart=60, maxiter=10, xtype=None, M=None, callback=printnorm,restrt=None)[0]
#  #sol = GMRes(A, main.dt*R0.flatten(), old.flatten(),5)
#
#  R =  np.reshape( mv_resid(sol,main.dt*R0.flatten()) , np.shape(main.a.a))
#  R_coarse = R[:,0:main.order/2,0:main.order/2,0:main.order/2]
#  print('Running Coarse Mesh')
#  e = gmres(A_coarse,-R_coarse.flatten(), np.zeros(np.shape(R_coarse.flatten())), tol=2e-04, restart=60, maxiter=60, xtype=None, M=None, callback=printnorm,restrt=None)[0]
##  e = GMRes(A_coarse,-R_coarse.flatten(), np.zeros(np.shape(R_coarse.flatten())),30)
#
#  x0 = np.zeros(np.shape(main.a.a))
#  x0[:] = np.reshape(sol,np.shape(main.a.a))
#  x0[:,0:main.order/2,0:main.order/2,0:main.order/2] = x0[:,0:main.order/2,0:main.order/2,0:main.order/2] + np.reshape( e , np.shape(main_coarse.a.a) )
#
#  print(np.linalg.norm(sol),np.linalg.norm(x0)) 
#
#  print('Running Fine Mesh')
#  sol = gmres(A, main.dt*R0.flatten(), x0=x0.flatten(), tol=3.5e-04, restart=60, maxiter=None, xtype=None, M=None, callback=printnorm,restrt=None)[0]
##  sol = GMRes(A, main.dt*R0.flatten(), x0.flatten(),5)
#
#  R =  np.reshape( mv_resid(sol,main.dt*R0.flatten()) , np.shape(main.a.a))
#  R_coarse = R[:,0:main.order/2,0:main.order/2,0:main.order/2]
#  print('Running Coarse Mesh')
#  e = gmres(A_coarse,-R_coarse.flatten(), np.zeros(np.shape(R_coarse.flatten())), tol=1e-05, restart=60, maxiter=60, xtype=None, M=None, callback=printnorm,restrt=None)[0]
##  e = GMRes(A_coarse,-R_coarse.flatten(), np.zeros(np.shape(R_coarse.flatten())),20)
#
#  x0[:] = np.reshape(sol,np.shape(main.a.a))
#  x0[:,0:main.order/2,0:main.order/2,0:main.order/2] = x0[:,0:main.order/2,0:main.order/2,0:main.order/2] + np.reshape( e , np.shape(main_coarse.a.a) )
#  print(np.linalg.norm(sol),np.linalg.norm(e)) 
#  sol = gmres(A, main.dt*R0.flatten(), x0=x0.flatten(), tol=1e-06, restart=60, maxiter=None, xtype=None, M=None, callback=printnorm,restrt=None)

  main.comm.Barrier()
  main.a.a = main.a0 + np.reshape(sol[0],np.shape(main.a.a))
  main.a0[:] = np.reshape(sol[0],np.shape(main.a.a))
  main.t += main.dt
  main.iteration += 1



def advanceSolImplicitPC(main,MZ,eqns,schemes):
  coarsen = 4
  main_coarse = variables(main.Nel,main.order/coarsen,main.quadpoints/(2*coarsen),eqns,main.mu,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,schemes,'DNS',main.procx,main.procy)
  old = np.zeros(np.shape(main.a0))
  old[:] = main.a0[:]
  main.a0[:] = main.a.a[:]
  main_coarse.a.a[:] = main.a.a[:,0:main.order/coarsen,0:main.order/coarsen,0:main.order/coarsen]
  main_coarse.a0[:] = main.a.a[:,0:main.order/coarsen,0:main.order/coarsen,0:main.order/coarsen]
  main.getRHS(main,eqns,schemes)
  main_coarse.getRHS(main_coarse,eqns,schemes)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  R0_coarse = np.zeros(np.shape(main_coarse.RHS))
  R0_coarse[:] = main_coarse.RHS[:]
  old_coarse = np.zeros(np.size(main_coarse.a.a))
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
  def printnormPC(r):
    print('PC GMRES norm = ' + str(np.linalg.norm(r)))

  def mv_coarse(v):
    vr = np.reshape(v,np.shape(main_coarse.a.a))
    eps = 1.e-5
    main_coarse.a.a[:] = main_coarse.a0[:] + eps*vr
    main_coarse.getRHS(main_coarse,eqns,schemes)
    R1 = np.zeros(np.shape(main_coarse.RHS))
    R1[:] = main_coarse.RHS[:]
    Av = vr - main_coarse.dt/2.*(R1 - R0_coarse)/eps
    main_coarse.a.a[:] = main_coarse.a0[:] 
    return Av.flatten()

  A_coarse = LinearOperator( (np.size(main_coarse.a.a),np.size(main_coarse.a.a)), matvec=mv_coarse )
  def coarse_solve(v):
    vf = np.reshape(v,np.shape(main.a.a))
    vr = vf[:,0:main.order/coarsen,0:main.order/coarsen,0:main.order/coarsen]
    print(np.linalg.norm(vr))
#    sol2 = lgmres(A_coarse, vr.flatten(),x0=np.zeros(np.size(main_coarse.a.a)), tol=1e-02, maxiter=10, M=None, callback=printnormPC, inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True)[0]
    sol2 = lgmres(A_coarse, vr.flatten(),old_coarse, tol=1e-02, maxiter=10, M=None, callback=printnormPC, inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True)[0]
    old_coarse[:] = sol2[:]
#   sol2 = bicgstab(A_coarse, vr.flatten(), x0=np.zeros(np.size(main_coarse.a.a)), tol=1e-02,maxiter=3, xtype=None, M=None, callback=printnormPC)[0]

#    sol2 = GMRes(mv_coarse,vr.flatten(),x0=np.zeros(np.size(main_coarse.a.a)),nmax_iter = 10,restart=None)
    solf = np.zeros(np.shape(main.a.a))
    solf[:,0:main.order/coarsen,0:main.order/coarsen,0:main.order/coarsen] = np.reshape(sol2,np.shape(main_coarse.a.a) )
    return solf.flatten()
#    return v

  A = LinearOperator( (np.size(main.a.a),np.size(main.a.a)), matvec=mv )
  APC = LinearOperator( (np.size(main.a.a),np.size(main.a.a)), matvec=coarse_solve )

  sol = gmres(A, main.dt*R0.flatten(), x0=old.flatten(), tol=1e-07, restart=60, maxiter=None, xtype=None, M=APC, callback=printnorm,restrt=None)
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

