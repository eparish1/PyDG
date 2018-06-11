import numpy as np
from mpi4py import MPI
from timeSchemes import *
from linear_solvers import *
from nonlinear_solvers import *
class timeschemes:
  def __init__(self,time_str='ExplicitRK4',lsolver_str='GMRes',nlsolver_str='Newton'):
    comm = MPI.COMM_WORLD
    check_t = 0
    if (comm.Get_rank() == 0): print('Time marching is set to ' + time_str)
    if (time_str == 'ExplicitRK4'):
      check_t = 0
      self.advanceSol = ExplicitRK4
      self.args = None
    if (time_str == 'ExplicitRK2'):
      check_t = 0
      self.advanceSol = ExplicitRK2
      self.args = None
    if (time_str == 'SSP_RK3'):
      check_t = 0
      self.advanceSol = SSP_RK3
      self.args = None
    if (time_str == 'SSP_RK3_Entropy'):
      check_t = 0
      self.advanceSol = SSP_RK3_Entropy
      self.args = None
    if (time_str == 'SSP_RK3_DOUBLEFLUX'):
      check_t = 0
      self.advanceSol = SSP_RK3_DOUBLEFLUX
      self.args = None

    if (time_str == 'SpaceTime'):
      check_t = 0
      self.advanceSol = spaceTime 
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]

    if (time_str == 'SpaceTimeExperimental'):
      check_t = 0
      self.advanceSol = spaceTimeExperimental 
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]


    if (time_str == 'SpaceTimeSplitting'):
      check_t = 0
      self.advanceSol = spaceTimeSplitting 
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]
    if (time_str == 'SpaceTimePC'):
      check_t = 0
      self.advanceSol = spaceTimePC
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]


    if (time_str == 'fractionalStep'):
      check_t = 0
      self.advanceSol = fractionalStep
      self.linear_solver = linearSolver(lsolver_str)
      self.pressure_linear_solver = linearSolver('BICGSTAB')
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature,self.pressure_linear_solver]


    if (time_str == 'SpaceTimeIncomp'):
      check_t = 0
      self.advanceSol = spaceTimeIncomp 
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]


    if (time_str == 'SteadyState'):
      check_t = 0
      self.advanceSol = SteadyState
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]

    if (time_str == 'SteadyStateExperimental'):
      check_t = 0
      self.advanceSol = SteadyStateExperimental
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]

    if (time_str == 'BackwardEuler'):
      check_t = 0
      self.advanceSol = backwardEuler
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]

    if (time_str == 'CrankNicolson'):
      check_t = 0
      self.advanceSol = CrankNicolson
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]

    if (time_str == 'CrankNicolsonEntropy'):
      check_t = 0
      self.advanceSol = CrankNicolsonEntropy
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]

    if (time_str == 'CrankNicolsonEntropyMZ'):
      check_t = 0
      self.advanceSol = CrankNicolsonEntropyMZ
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]


    if (time_str == 'CrankNicolsonIncomp'):
      check_t = 0
      self.advanceSol = CrankNicolsonIncomp
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]

    if (time_str == 'StrangSplitting'):
      check_t = 0
      self.advanceSol = StrangSplitting
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = True
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]


    if (time_str == 'SDIRK2'):
      check_t = 0
      self.advanceSol = SDIRK2 
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = True 
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]
    if (time_str == 'SDIRK4'):
      check_t = 0
      self.advanceSol = SDIRK4 
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = False 
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]

class nonlinearSolver:
  def __init__(self,SolverType='Newton',rtol=1e-8,printnorm=1):
    comm = MPI.COMM_WORLD
    if (comm.Get_rank() == 0): print('NL solver set to ' + SolverType)
    if (SolverType == 'Newton'):
      self.solve = newtonSolver
      self.rtol=rtol
      self.printnorm = printnorm
    if (SolverType == 'NEJ'):
      self.solve = NEJSolver
      self.rtol=rtol
      self.printnorm = printnorm
    if (SolverType == 'Newton_MG'):
      self.solve = newtonSolver_MG
      self.rtol=rtol
      self.printnorm = printnorm
    if (SolverType == 'Newton_PC'):
      self.solve = newtonSolver_PC2
      self.rtol=rtol
      self.printnorm = printnorm
    if (SolverType == 'pseudoTime'):
      self.solve = pseudoTimeSolver
      self.rtol=rtol
      self.printnorm = printnorm
    if (SolverType == 'pseudoTime_MG'):
      self.solve = pseudoTimeSolver_MG
      self.rtol=rtol
      self.printnorm = printnorm


class linearSolver:
  def __init__(self,SolverType='GMRes',tol=1e-8,maxiter_outer=1,maxiter=15,printnorm=0):
    comm = MPI.COMM_WORLD
    if (SolverType == 'GMRes'):
      if (comm.Get_rank() == 0): print('Linear solver set to ' + SolverType)
      self.solve = GMRes
      self.tol=tol
      self.maxiter_outer = maxiter_outer
      self.maxiter = maxiter
      self.printnorm = printnorm
    if (SolverType == 'Jacobi'):
      if (comm.Get_rank() == 0): print('Linear solver set to ' + SolverType)
      self.solve = Jacobi
      self.tol=tol
      self.maxiter_outer = maxiter_outer
      self.maxiter = maxiter
      self.printnorm = printnorm
    if (SolverType == 'ADI'):
      if (comm.Get_rank() == 0): print('Linear solver set to ' + SolverType)
      self.solve = ADI
      self.tol=tol
      self.maxiter_outer = maxiter_outer
      self.maxiter = maxiter
      self.printnorm = printnorm
    if (SolverType == 'fGMRes'):
      if (comm.Get_rank() == 0): print('Linear solver set to ' + SolverType)
      self.solve = fGMRes
      self.solvePC = GMRes 
      self.tol=tol
      self.maxiter_outer = maxiter_outer
      self.maxiter = maxiter
      self.printnorm = printnorm
    if (SolverType == 'RungeKutta'):
      if (comm.Get_rank() == 0): print('Linear solver set to ' + SolverType)
      self.solve = rungeKutta
      self.solvePC = rungeKutta
      self.tol=tol
      self.maxiter_outer = maxiter_outer
      self.maxiter = maxiter
    if (SolverType == 'BICGSTAB'):
      if (comm.Get_rank() == 0): print('Linear solver set to ' + SolverType)
      self.solve = BICGSTAB
      self.tol=tol
      self.maxiter_outer = maxiter_outer
      self.maxiter = 50
      self.printnorm = printnorm

