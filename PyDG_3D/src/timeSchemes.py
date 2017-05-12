import numpy as np
#import sys, petsc4py
#petsc4py.init(sys.argv)
#from petsc4py import PETSc
from mpi4py import MPI
import sys
from turb_models import tauModel
from init_Classes import variables
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres,bicgstab
from scipy.sparse.linalg import lgmres
from scipy.optimize import newton_krylov
from myGMRES import GMRes
from scipy.optimize.nonlin import InverseJacobian
from scipy.optimize.nonlin import BroydenFirst, KrylovJacobian
import time
def ExplicitRK4(main,MZ,eqns,args=None):
  main.a0[:] = main.a.a[:]
  rk4const = np.array([1./4,1./3,1./2,1.])
  for i in range(0,4):
    main.rkstage = i
    if (main.turb_str == 'DNS'):
      main.getRHS(main,eqns)  ## put RHS in a array since we don't need it
      main.a.a[:] = main.a0 + main.dt*rk4const[i]*(main.RHS[:])
    else:
      main.RHS[:],w = main.turb_model(main,MZ,eqns)
      #print(np.linalg.norm(main.w))
      main.a.a[:] = main.a0 + main.dt*rk4const[i]*(main.RHS[:] + w)
  main.t += main.dt
  main.iteration += 1

def CrankNicolson(main,MZ,eqns,args):
  nonlinear_solver = args[0]
  linear_solver = args[1]
  sparse_quadrature = args[2]
  main.a0[:] = main.a.a[:]
  main.getRHS(main,eqns)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  def unsteadyResidual(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    main.getRHS(main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    #Rstar = np.reshape( (v.flatten() - main.a0.flatten() ) - 0.5*main.dt*(R0 + R1).flatten() , np.shape(main.a.a))
    Rstar = ( main.a.a[:] - main.a0 ) - 0.5*main.dt*(R0 + R1)
    ## Create Global residual
    data = main.comm.gather(np.linalg.norm(Rstar)**2,root = 0)
    if (main.mpi_rank == 0):
      Rstar_glob = 0.
      for j in range(0,main.num_processes):
        Rstar_glob += data[j]
      Rstar_glob = np.sqrt(Rstar_glob)
      for j in range(1,main.num_processes):
        main.comm.send(Rstar_glob, dest=j)
    else:
      Rstar_glob = main.comm.recv(source=0)
    return Rstar,R1,Rstar_glob

  def create_MF_Jacobian(v,args,main):
    an = args[0]
    Rn = args[1]
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 5.e-2
    main.a.a[:] = an + eps*vr
    main.getRHS(main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Av = vr - main.dt/2.*(R1 - Rn)/eps
    return Av.flatten()

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
    def newtonHook(main1,main2,Rn):
      pass

  nonlinear_solver.solve(unsteadyResidual, create_MF_Jacobian,main,main,linear_solver,newtonHook)
  main.t += main.dt
  main.iteration += 1


def advanceSolImplicit_MG(main,MZ,eqns):
  main.a0[:] = main.a.a[:]
  main.getRHS(main,eqns)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  def unsteadyResid(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    main.getRHS(main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    #Rstar = np.reshape( (v.flatten() - main.a0.flatten() ) - 0.5*main.dt*(R0 + R1).flatten() , np.shape(main.a.a))
    Rstar = ( main.a.a[:] - main.a0 ) - 0.5*main.dt*(R0 + R1)
    ## Create Global residual
    data = main.comm.gather(np.linalg.norm(Rstar)**2,root = 0)
    if (main.mpi_rank == 0):
      Rstar_glob = 0.
      for j in range(0,main.num_processes):
        Rstar_glob += data[j]
      Rstar_glob = np.sqrt(Rstar_glob)
      for j in range(1,main.num_processes):
        main.comm.send(Rstar_glob, dest=j)
    else:
      Rstar_glob = main.comm.recv(source=0)
    return Rstar,R1,Rstar_glob

  Rstarn,Rn,Rstar_glob = unsteadyResid(main.a.a)
  NLiter = 0
  Rstar_glob0 = Rstar_glob*1.
  if (main.mpi_rank == 0):
    print('NL iteration = ' + str(NLiter) + '  NL residual = ' + str(Rstar_glob) ,' relative decrease = ' + str(Rstar_glob/Rstar_glob0)) 

  coarsen = 2
  coarsen2 = 4
  old = np.zeros(np.shape(main.a.a))
  Rstar_glob0 = Rstar_glob*1.
  main_coarse = variables(main.Nel,main.order/coarsen,main.quadpoints/(2*coarsen),eqns,main.mu,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy)
  main_coarse.a.a[:] = main.a.a[:,0:main.order/coarsen,0:main.order/coarsen,0:main.order/coarsen]
  main_coarse.getRHS(main_coarse,eqns)
  Rnc = np.zeros(np.shape(main_coarse.RHS))
  Rnc[:] = main_coarse.RHS[:]

  main_qc = variables(main.Nel,main.order,main.quadpoints/2,eqns,main.mu,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy)
  main_qc.a.a[:] = main.a.a[:] 
  main_qc.getRHS(main_qc,eqns)
  Rn_qc = np.zeros(np.shape(main_qc.RHS))
  Rn_qc[:] = main_qc.RHS[:]


  main_coarse2 = variables(main.Nel,main.order/coarsen2,main.quadpoints/(2*coarsen2),eqns,main.mu,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy)
  main_coarse2.a.a[:] = main.a.a[:,0:main.order/coarsen2,0:main.order/coarsen2,0:main.order/coarsen2]
  main_coarse2.getRHS(main_coarse2,eqns)
  Rnc2 = np.zeros(np.shape(main_coarse2.RHS))
  Rnc2[:] = main_coarse2.RHS[:]


  an = np.zeros(np.shape(main.a.a))
  an[:] = main.a.a[:]
  anc = np.zeros(np.shape(main_coarse.a.a))
  anc[:] = main_coarse.a.a[:]
  anc2 = np.zeros(np.shape(main_coarse2.a.a))
  anc2[:] = main_coarse2.a.a[:]

  NLiter = 0
  while (Rstar_glob >= 1e-8 and Rstar_glob/Rstar_glob0 > 1e-8):
    NLiter += 1

    def mv_coarse2(v1):
      vr = np.reshape(v1,np.shape(main_coarse2.a.a))
      eps = 5.e-2
      main_coarse2.a.a[:] = anc2[:] + eps*vr#*filtarray
      main_coarse2.getRHS(main_coarse2,eqns)
      R1 = np.zeros(np.shape(main_coarse2.RHS))
      R1[:] = main_coarse2.RHS[:]
      Av = vr - main_coarse2.dt/2.*(R1 - Rnc2)/eps
      return Av.flatten()


    def mv_coarse(v1):
      vr = np.reshape(v1,np.shape(main_coarse.a.a))
      eps = 5.e-2
      main_coarse.a.a[:] = anc[:] + eps*vr#*filtarray
      main_coarse.getRHS(main_coarse,eqns)
      R1 = np.zeros(np.shape(main_coarse.RHS))
      R1[:] = main_coarse.RHS[:]
      Av = vr - main_coarse.dt/2.*(R1 - Rnc)/eps
      return Av.flatten()

    def mv_qc(v):
      vr = np.reshape(v,np.shape(main_qc.a.a))
      eps = 5.e-2
      main_qc.a.a[:] = an[:] + eps*vr
      main_qc.getRHS(main_qc,eqns)
      R1 = np.zeros(np.shape(main.RHS))
      R1[:] = main_qc.RHS[:]
      Av = vr - main.dt/2.*(R1 - Rn_qc)/eps
      return Av.flatten()


    def mv(v):
      vr = np.reshape(v,np.shape(main.a.a))
      eps = 5.e-2
      main.a.a[:] = an[:] + eps*vr
      main.getRHS(main,eqns)
      R1 = np.zeros(np.shape(main.RHS))
      R1[:] = main.RHS[:]
      Av = vr - main.dt/2.*(R1 - Rn)/eps
      return Av.flatten()

    def mv_resid(mvf,v,b):
      return b - mvf(v) 

    ts = time.time()
#    sol = GMRes(mv, -Rstarn.flatten(), np.zeros(np.size(main.a.a)),main, tol=1e-6,maxiter_outer=1,maxiter=20, restart=None,printnorm=0)
    old[:] = 0.
    for i in range(0,1):
#      if (main.mpi_rank == 0): print('Fine Mesh')
      sol = GMRes(mv_qc,-Rstarn.flatten(), old.flatten(),main,tol=1e-5,maxiter_outer=1,maxiter=10, restart=None,printnorm=0)
      R =  np.reshape( mv_resid(mv,sol,-Rstarn.flatten()) , np.shape(main.a.a ) )
      R_coarse = R[:,0:main.order/coarsen,0:main.order/coarsen,0:main.order/coarsen]
#      if (main.mpi_rank == 0):
#        print('Running Coarse Mesh, time = ' + str(time.time() - ts))
      e = GMRes(mv_coarse,R_coarse.flatten(), np.zeros(np.shape(R_coarse.flatten())),main,tol=1e-6,maxiter_outer=1,maxiter=10, restart=None,printnorm=0)

      R1 =  np.reshape( mv_resid(mv_coarse,e,R_coarse.flatten()) , np.shape(main_coarse.a.a ) )
      R_coarse2 = R1[:,0:main.order/coarsen2,0:main.order/coarsen2,0:main.order/coarsen2]
      #
      e2 = GMRes(mv_coarse2,R_coarse2.flatten(), np.zeros(np.shape(R_coarse2.flatten())),main,tol=1e-6,maxiter_outer=1,maxiter=20, restart=None,printnorm=0)
      ##
      etmp = np.reshape(e,np.shape(main_coarse.a.a))
      etmp[:,0:main.order/coarsen2,0:main.order/coarsen2,0:main.order/coarsen2] += np.reshape(e2,np.shape(main_coarse2.a.a))
      e = GMRes(mv_coarse,R_coarse.flatten(),etmp.flatten(),main,tol=1e-6,maxiter_outer=1,maxiter=20, restart=None,printnorm=0)
 
      x0 = np.zeros(np.shape(main.a.a))
      x0[:] = np.reshape(sol,np.shape(main.a.a))
      x0[:,0:main.order/coarsen,0:main.order/coarsen,0:main.order/coarsen] = x0[:,0:main.order/coarsen,0:main.order/coarsen,0:main.order/coarsen] + np.reshape( e , np.shape(main_coarse.a.a) )
      old[:] = x0[:]

    sol = GMRes(mv_qc,-Rstarn.flatten(), old.flatten(),main,tol=1e-5,maxiter_outer=1,maxiter=30, restart=None,printnorm=0)

    main.a.a[:] = an[:] + np.reshape(sol,np.shape(main.a.a))#*1.01
    Rstarn,Rn,Rstar_glob = unsteadyResid(main.a.a)
    an[:] = main.a.a[:]
    
    main_qc.a.a[:] = main.a.a[:]
    main_qc.getRHS(main_qc,eqns)
    Rn_qc[:] = main_qc.RHS[:]



    main_coarse.a.a[:] = main.a.a[:,0:main.order/coarsen,0:main.order/coarsen,0:main.order/coarsen]
    main_coarse.getRHS(main_coarse,eqns)
    Rnc[:] = main_coarse.RHS[:]
    anc[:] = main_coarse.a.a[:]

    main_coarse2.a.a[:] = main.a.a[:,0:main.order/coarsen2,0:main.order/coarsen2,0:main.order/coarsen2]
    main_coarse2.getRHS(main_coarse2,eqns)
    Rnc2[:] = main_coarse2.RHS[:]
    anc2[:] = main_coarse2.a.a[:]

    if (main.mpi_rank == 0):
      print('NL iteration = ' + str(NLiter) + '  NL residual = ' + str(Rstar_glob) ,' relative decrease = ' + str(Rstar_glob/Rstar_glob0), ' Solve time = ' + str(time.time() - ts) ) 

  main.t += main.dt
  main.iteration += 1




def advanceSolImplicit_NK(main,MZ,eqns):
  old = np.zeros(np.shape(main.a0))
  old[:] = main.a0[:]
  main.a0[:] = main.a.a[:]
  main.getRHS(main,eqns)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  def unsteadyResid(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    main.getRHS(main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Resid = (v - main.a0.flatten() ) - 0.5*main.dt*(R0 + R1).flatten()
    return Resid

  def printnorm(x,r):
    ## Create Global residual
    data = main.comm.gather(np.linalg.norm(r)**2,root = 0)
    if (main.mpi_rank == 0):
      Rstar_glob = 0.
      for j in range(0,main.num_processes):
        Rstar_glob += data[j]
      Rstar_glob = np.sqrt(Rstar_glob)
      for j in range(1,main.num_processes):
        main.comm.send(Rstar_glob, dest=j)
    else:
      Rstar_glob = main.comm.recv(source=0)
    if (Rstar_glob < 1e-9):
      if (main.mpi_rank == 0):
        print('Convergence Achieved, NL norm = ' + str(np.linalg.norm(Rstar_glob)))
      r[:] = 1e-100
    else:
      if (main.mpi_rank == 0):
        print('NL norm = ' + str(np.linalg.norm(Rstar_glob)))

  #jac = BroydenFirst()
  #kjac = KrylovJacobian(inner_M=InverseJacobian(jac))
  sol = newton_krylov(unsteadyResid, main.a.a.flatten(), iter=None, rdiff=None, method='gmres', inner_maxiter=40, inner_M=None, outer_k=10, verbose=4, maxiter=None, f_tol=1e-8, f_rtol=1e-8, x_tol=None, x_rtol=None, tol_norm=None, line_search='armijo', callback=printnorm)
  main.a.a =   np.reshape(sol,np.shape(main.a.a))
  main.a0[:] = np.reshape(sol,np.shape(main.a.a))
  main.t += main.dt
  main.iteration += 1


def printnorm(r):
  print('GMRES norm = ' + str(np.linalg.norm(r)))

def printnormPC(r):
  print('PC GMRES norm = ' + str(np.linalg.norm(r)))


def advanceSolImplicit_MYNKPC(main,MZ,eqns):
  main.a0[:] = main.a.a[:]
  main.getRHS(main,eqns)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  def unsteadyResid(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    main.getRHS(main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    #Rstar = np.reshape( (v.flatten() - main.a0.flatten() ) - 0.5*main.dt*(R0 + R1).flatten() , np.shape(main.a.a))
    Rstar = ( main.a.a[:] - main.a0 ) - 0.5*main.dt*(R0 + R1)
    ## Create Global residual
    data = main.comm.gather(np.linalg.norm(Rstar)**2,root = 0)
    if (main.mpi_rank == 0):
      Rstar_glob = 0.
      for j in range(0,main.num_processes):
        Rstar_glob += data[j]
      Rstar_glob = np.sqrt(Rstar_glob)
      for j in range(1,main.num_processes):
        main.comm.send(Rstar_glob, dest=j)
    else:
      Rstar_glob = main.comm.recv(source=0)
    return Rstar,R1,Rstar_glob

  Rstarn,Rn,Rstar_glob = unsteadyResid(main.a.a)
  NLiter = 0
  an = np.zeros(np.shape(main.a0))
  an[:] = main.a0[:]
  Rstar_glob0 = Rstar_glob*1.
  if (main.mpi_rank == 0):
    print('NL iteration = ' + str(NLiter) + '  NL residual = ' + str(Rstar_glob) ,' relative decrease = ' + str(Rstar_glob/Rstar_glob0)) 

  coarsen = 2
  coarsen2 = 1
  old = np.zeros(np.size(main.a.a))
  coarsen_f = 1
  Rstar_glob0 = Rstar_glob*1.
  main_coarse = variables(main.Nel,main.order,main.quadpoints/(coarsen),eqns,main.mu,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy)
  filtarray = np.zeros(np.shape(main.a.a))
  filtarray[:,0:main.order/coarsen_f,0:main.order/coarsen_f,0:main.order/coarsen_f] = 1.
  main_coarse.a.a[:] = main.a.a[:]
  main_coarse.getRHS(main_coarse,eqns)
  Rnc = np.zeros(np.shape(main_coarse.RHS))
  Rnc[:] = main_coarse.RHS[:]

  #main_coarse2 = variables(main.Nel,main.order,main.quadpoints/(coarsen2),eqns,main.mu*0,main.xG,main.yG,main.zG,main.t,main.et,main.dt,main.iteration,main.save_freq,'DNS',main.procx,main.procy)
  #main_coarse2.a.a[:] = main.a.a[:]

  NLiter = 0
  an = np.zeros(np.shape(main.a0))
  an[:] = main.a0[:]
  while (Rstar_glob >= 1e-8 and Rstar_glob/Rstar_glob0 > 1e-8):
    NLiter += 1


    def mv_coarse2(v1):
      vr = np.reshape(v1,np.shape(main_coarse.a.a))
      eps = 5e-2
      main_coarse2.a.a[:] = an[:]*filtarray
      main_coarse2.getRHS(main_coarse2,eqns)

      Rn = np.zeros(np.shape(main_coarse2.RHS))
      Rn[:] = main_coarse2.RHS[:]
      main_coarse2.a.a[:] = an[:]*filtarray + eps*vr#*filtarray
      main_coarse2.getRHS(main_coarse2,eqns)
      R1 = np.zeros(np.shape(main_coarse2.RHS))
      R1[:] = main_coarse2.RHS[:]
      Av = vr - main_coarse.dt/2.*(R1 - Rn)/eps
      return Av.flatten()



    def mv_coarse(v1):
      #y = GMRes(mv_coarse2,v1, 0*np.ones(np.size(main.a.a)),main, tol=1e-5,maxiter_outer=1,maxiter=10, restart=None,printnorm=0)
      #vr = np.reshape(y,np.shape(main_coarse.a.a))

      vr = np.reshape(v1,np.shape(main.a.a))
      eps = 5.e-2
      #main_coarse.a.a[:] = an[:]*filtarray
      #main_coarse.getRHS(main_coarse,eqns)
      #Rn = np.zeros(np.shape(main_coarse.RHS))
      #Rn[:] = main_coarse.RHS[:]

      main_coarse.a.a[:] = an[:]*filtarray + eps*vr#*filtarray
      main_coarse.getRHS(main_coarse,eqns)
      R1 = np.zeros(np.shape(main_coarse.RHS))
      R1[:] = main_coarse.RHS[:]
      Av = vr - main_coarse.dt/2.*(R1 - Rnc)/eps
      return Av.flatten()


    def mv2(v):
      vr = np.reshape(v,np.shape(main.a.a))
      eps = 5.e-2
      main.a.a[:] = an[:] + eps*vr
      main.getRHS(main,eqns)
      R1 = np.zeros(np.shape(main.RHS))
      R1[:] = main.RHS[:]
      Av = vr - main.dt/2.*(R1 - Rn)/eps
      return Av.flatten()


    def mv(v):
      y = GMRes(mv_coarse,v, 0.*np.ones(np.size(main.a.a)),main, tol=1e-7,maxiter_outer=1,maxiter=120, restart=None,printnorm=0)
      #w = GMRes(mv_coarse,v, 1.e-20*np.ones(np.size(main.a.a)),main, tol=1e-5,maxiter_outer=1,maxiter=40, restart=None,printnorm=1)
      #y = GMRes(mv_coarse2,w, 1.e-20*np.ones(np.size(main.a.a)),main, tol=1e-5,maxiter_outer=1,maxiter=40, restart=None,printnorm=1)
      vr = np.reshape(y,np.shape(main.a.a))
      eps = 5.e-2
      main.a.a[:] = an[:] + eps*vr
      main.getRHS(main,eqns)
      R1 = np.zeros(np.shape(main.RHS))
      R1[:] = main.RHS[:]
      Av = vr - main.dt/2.*(R1 - Rn)/eps
      return Av.flatten()

    ts = time.time()
    #w = GMRes(mv, -Rstarn.flatten(), np.ones(np.size(main.a.a))*1.e-10,main, tol=1e-8,maxiter_outer=1,maxiter=5, restart=None,printnorm=1)
    #sol = GMRes(mv_coarse,w, np.ones(np.size(main.a.a))*0.,main, tol=1e-7,maxiter_outer=1,maxiter=120, restart=None,printnorm=1)
    sol = GMRes(mv_coarse, -Rstarn.flatten(), old*0.,main, tol=1e-6,maxiter_outer=1,maxiter=40, restart=None,printnorm=0)
    old[:] = sol[:]
    #sol = GMRes(mv_coarse
    main.a.a[:] = an[:] + np.reshape(sol,np.shape(main.a.a))#*1.01
    Rstarn,Rn,Rstar_glob = unsteadyResid(main.a.a)
    an[:] = main.a.a[:]
    #main_coarse.a.a[:] = main.a.a[:]

    main_coarse.a.a[:] = main.a.a[:]
    main_coarse.getRHS(main_coarse,eqns)
    Rnc[:] = main_coarse.RHS[:]
    if (main.mpi_rank == 0):
      print('NL iteration = ' + str(NLiter) + '  NL residual = ' + str(Rstar_glob) ,' relative decrease = ' + str(Rstar_glob/Rstar_glob0), ' Solve time = ' + str(time.time() - ts) ) 

  main.t += main.dt
  main.iteration += 1


def advanceSolImplicit2(main,MZ,eqns):
  old = np.zeros(np.shape(main.a0))
  old[:] = main.a0[:]
  main.a0[:] = main.a.a[:]
  main.getRHS(main,eqns)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  alpha = (2. - np.sqrt(2.))/2.
  def mv(v):
    vr = np.reshape(v,np.shape(main.a.a))
    eps = 1.e-5
    main.a.a[:] = main.a0[:] + eps*vr
    main.getRHS(main,eqns)
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
  main.getRHS(main,eqns)
  R2 = main.dt*( (1 - alpha)*main.RHS + alpha*R0).flatten()
  sol = gmres(A, R2, x0=np.zeros(np.size(R2)), tol=1e-07, restart=60, maxiter=None, xtype=None, M=None, callback=printnorm,restrt=None)
  main.a.a = main.a0 + np.reshape(sol[0],np.shape(main.a.a))
  main.t += main.dt
  main.iteration += 1


def advanceSolImplicit4_NK(main,MZ,eqns):
  main.a0[:] = main.a.a[:]
  main.getRHS(main,eqns)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  gam = 9./40.
  c2 = 7./13.
  c3 = 11./15.
  b2 = -(-2. + 3.*c3 + 9.*gam - 12.*c3*gam - 6.*gam**2 + 6.*c3*gam**2)/(6.*(c2 - c3)*(c2 - gam))
  b3 =  (-2. + 3.*c2 + 9.*gam - 12.*c2*gam - 6.*gam**2 + 6.*c2*gam**2)/(6.*(c2 - c3)*(c3 - gam))
  a32 = -(c2 - c3)*(c3 - gam)*(-1. + 9.*gam - 18.*gam**2 + 6.*gam**3)/ \
       ( (c2 - gam)*(-2. + 3.*c2 + 9.*gam - 12.*c2*gam - 6.*gam**2 + 6.*c2*gam**2) )

  def unsteadyResid(v):
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    main.getRHS(main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    #Rstar = np.reshape( (v.flatten() - main.a0.flatten() ) - 0.5*main.dt*(R0 + R1).flatten() , np.shape(main.a.a))
    Rstar = ( main.a.a[:] - main.a0 ) - 0.5*main.dt*(R0 + R1)
    ## Create Global residual
    data = main.comm.gather(np.linalg.norm(Rstar)**2,root = 0)
    if (main.mpi_rank == 0):
      Rstar_glob = 0.
      for j in range(0,main.num_processes):
        Rstar_glob += data[j]
      Rstar_glob = np.sqrt(Rstar_glob)
      for j in range(1,main.num_processes):
        main.comm.send(Rstar_glob, dest=j)
    else:
      Rstar_glob = main.comm.recv(source=0)
    return Rstar,R1,Rstar_glob
  Rstarn,Rn,Rstar_glob = unsteadyResid(main.a.a)
  NLiter = 0
  an = np.zeros(np.shape(main.a0))
  an[:] = main.a0[:]
  Rstar_glob0 = Rstar_glob*1.
  if (main.mpi_rank == 0):
    print('NL iteration = ' + str(NLiter) + '  NL residual = ' + str(Rstar_glob) ,' relative decrease = ' + str(Rstar_glob/Rstar_glob0)) 

  while (Rstar_glob/Rstar_glob0 >= 1e-7):
    NLiter += 1
    def mv(v):
      vr = np.reshape(v,np.shape(main.a.a))
      eps = 1.e-4
      main.a.a[:] = an[:] + eps*vr
      main.getRHS(main,eqns)
      R1 = np.zeros(np.shape(main.RHS))
      R1[:] = main.RHS[:]
      Av = vr - main.dt/2.*(R1 - Rn)/eps
      return Av.flatten()
    sys.stdout.flush()
    ts = time.time()
    main.a.a[:] = an[:] + np.reshape(sol,np.shape(main.a.a))
    an[:] = main.a.a[:]
    Rstarn,Rn,Rstar_glob = unsteadyResid(main.a.a)
    if (main.mpi_rank == 0):
      print('NL iteration = ' + str(NLiter) + '  NL residual = ' + str(Rstar_glob) ,' relative decrease = ' + str(Rstar_glob/Rstar_glob0), ' Solve time = ' + str(time.time() - ts) ) 
  main.t += main.dt
  main.iteration += 1


def advanceSolImplicit4(main,MZ,eqns):
  old = np.zeros(np.shape(main.a0))
  old[:] = main.a0[:]
  main.a0[:] = main.a.a[:]
  main.getRHS(main,eqns)
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
    main.getRHS(main,eqns)
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
  main.getRHS(main,eqns)
  R1 = np.zeros(np.shape(main.RHS))
  R1[:] = main.RHS[:]
  #stage 2
  RHS_stage2 = main.dt*( (c2 - gam)*R1 + gam*R0).flatten()
  sol = gmres(A, RHS_stage2, x0=sol[0], tol=1e-07, restart=200, maxiter=None, xtype=None, M=None, callback=printnorm,restrt=None)
  main.a.a = main.a0 + np.reshape(sol[0],np.shape(main.a.a))
  main.getRHS(main,eqns)
  R2 = np.zeros(np.shape(main.RHS))
  R2[:] = main.RHS[:]
  #stage 3
  RHS_stage3 = main.dt*( (c3 - a32 - gam)*R1 + a32*R2 + gam*R0).flatten()
  sol = gmres(A, RHS_stage3, x0=sol[0], tol=1e-07, restart=200, maxiter=None, xtype=None, M=None, callback=printnorm,restrt=None)
  main.a.a = main.a0 + np.reshape(sol[0],np.shape(main.a.a))
  main.getRHS(main,eqns)
  R3 = np.zeros(np.shape(main.RHS))
  R3[:] = main.RHS[:]
  #stage 4
  RHS_stage4 = main.dt*( (1. - b2 - b3 - gam)*R1 + b2*R2 + b3*R3 + gam*R0).flatten()
  sol = gmres(A, RHS_stage4, x0=sol[0], tol=1e-07, restart=200, maxiter=None, xtype=None, M=None, callback=printnorm,restrt=None)
  main.a.a = main.a0 + np.reshape(sol[0],np.shape(main.a.a))
  main.t += main.dt
  main.iteration += 1










def advanceSolImplicit_PETsc(main,MZ,eqns):
  old = np.zeros(np.shape(main.a0))
  old[:] = main.a0[:]
  main.a0[:] = main.a.a[:]
  main.getRHS(main,eqns)
  R0 = np.zeros(np.shape(main.RHS))
  R0[:] = main.RHS[:]
  def unsteadyResid( snes, V, R):
    v = V[...] #+ main.a0.flatten()
    Resid = R[...]
    main.a.a[:] = np.reshape(v,np.shape(main.a.a))
    main.getRHS(main,eqns)
    R1 = np.zeros(np.shape(main.RHS))
    R1[:] = main.RHS[:]
    Resid[:] = (v - main.a0.flatten() ) - 0.5*main.dt*(R0 + R1).flatten()


  # create PETSc nonlinear solver
  snes = PETSc.SNES().create()
  # register the function in charge of
  # computing the nonlinear residual
  f = PETSc.Vec().createSeq(np.size(main.a.a))
  snes.setFunction(unsteadyResid, f)
  snes.setType('newtonls')
  # configure the nonlinear solver
  # to use a matrix-free Jacobian
  snes.setUseMF(True)
  snes.getKSP().setType('gmres')
  snes.ksp.setGMRESRestart(50)
  opts = PETSc.Options()
  opts['snes_ngmres_restart_it'] = 80
  opts['snes_ngmres_m'] = 80
  opts['snes_ngmres_gammaC'] = 1e-10
  opts['snes_ngmres_epsilonB'] = 1e-10
  opts['snes_ngmres_deltaB'] = 1e-10
  opts['snes_monitor']  = ''
  opts['ksp_monitor']   = ''
  snes.setFromOptions()
  def monitor(snes, its, fgnorm):
    print(its,fgnorm)

  snes.setMonitor(monitor)
  snes.setTolerances(1e-7,1e-8,1e-7)
  snes.getKSP().setTolerances(1e-3,1e-50,10000.0,10000)
  tols2 = snes.getKSP().getTolerances()
#  snes_ngmres_restart_it
  tols = snes.getTolerances()
  # solve the nonlinear problem
  b, x = None, f.duplicate()
  x[...] = main.a.a[:]
#  x.set(0) # zero inital guess
  snes.solve(b, x)
  its = snes.getIterationNumber()
  lits = snes.getLinearSolveIterations()

  print "Number of SNES iterations = %d" % its
  print "Number of Linear iterations = %d" % lits

  sol = x[...]
  main.a.a =   np.reshape(sol,np.shape(main.a.a))
  main.a0[:] = np.reshape(sol,np.shape(main.a.a))
  main.t += main.dt
  main.iteration += 1

