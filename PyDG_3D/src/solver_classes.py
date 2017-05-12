import numpy as np
from timeSchemes import *
from linear_solvers import *
from nonlinear_solvers import *
class timeschemes:
  def __init__(self,time_str='ExplicitRK4'):
    check_t = 0
    if (time_str == 'ExplicitRK4'):
      check_t = 0
      self.advanceSol = ExplicitRK4
      self.args = None
    if (time_str == 'CrankNicolson'):
      check_t = 0
      self.advanceSol = CrankNicolson
      self.linear_solver = linearSolver()
      self.nonlinear_solver = nonlinearSolver()
      self.sparse_quadrature = False
      self.args = [self.nonlinear_solver,self.linear_solver,self.sparse_quadrature]

class nonlinearSolver:
  def __init__(self,SolverType='Newton',rtol=1e-8,printnorm=1):
    if (SolverType == 'Newton'):
      self.solve = newtonSolver
      self.rtol=rtol
      self.printnorm = printnorm


class linearSolver:
  def __init__(self,SolverType='GMRes',tol=1e-6,maxiter_outer=1,maxiter=40,printnorm=0):
    if (SolverType == 'GMRes'):
      self.solve = GMRes
      self.tol=tol
      self.maxiter_outer = maxiter_outer
      self.maxiter = maxiter
      self.printnorm = printnorm

