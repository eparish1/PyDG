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

    if (time_str == 'SpaceTime'):
      check_t = 0
      self.advanceSol = spaceTime 
      self.linear_solver = linearSolver(lsolver_str)
      self.nonlinear_solver = nonlinearSolver(nlsolver_str)
      self.sparse_quadrature = True
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]

    if (time_str == 'CrankNicolson'):
      check_t = 0
      self.advanceSol = CrankNicolson
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
    if (SolverType == 'Newton_MG'):
      self.solve = newtonSolver_MG
      self.rtol=rtol
      self.printnorm = printnorm


class linearSolver:
  def __init__(self,SolverType='GMRes',tol=1e-8,maxiter_outer=1,maxiter=40,printnorm=0):
    comm = MPI.COMM_WORLD
    if (SolverType == 'GMRes'):
      if (comm.Get_rank() == 0): print('Linear solver set to ' + SolverType)
      self.solve = GMRes
      self.tol=tol
      self.maxiter_outer = maxiter_outer
      self.maxiter = maxiter
      self.printnorm = printnorm

