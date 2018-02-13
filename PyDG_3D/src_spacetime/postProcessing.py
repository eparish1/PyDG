import numpy as np
import sys
from evtk.hl import gridToVTK
import numpy as np
import os.path
from navier_stokes_entropy import entropy_to_conservative
class postProcessorEntropy:
  def __init__(self):
    def writeAllToParaview(self,start=0,end=100000,skip=1):
      grid = np.load('../DGgrid.npz')
      x,y,z = grid['x'],grid['y'],grid['z']
      for i in range(0,end,skip):
        sol_str = 'npsol' + str(i) + '.npz'
        if (os.path.isfile(sol_str)):
          print('found ' + sol_str)
          sol = np.load('npsol' + str(i) + '.npz')
          string = 'PVsol' + str(i)
          U = entropy_to_conservative(sol['U'])
          gridToVTK(string, x,y,z, pointData = {"rho" : U[0] , \
            "u" : U[1]/U[0] , "v" : U[2]/U[0], "w" : U[3]/U[0], \
            "rhoE" : U[4]} )
    self.writeAllToParaview = writeAllToParaview

class postProcessorScalar:
  def __init__(self):
    def writeAllToParaview(self,start=0,end=100000,skip=1):
      grid = np.load('../DGgrid.npz')
      x,y,z = grid['x'],grid['y'],grid['z']
      for i in range(0,end,skip):
        sol_str = 'npsol' + str(i) + '.npz'
        if (os.path.isfile(sol_str)):
          print('found ' + sol_str)
          sol = np.load('npsol' + str(i) + '.npz')
          string = 'PVsol' + str(i)
          gridToVTK(string, x,y,z, pointData = {"u" : sol['U'][0]} )
    self.writeAllToParaview = writeAllToParaview


class postProcessor:
  def __init__(self):
    def writeAllToParaview(self,start=0,end=100000,skip=1):
      grid = np.load('../DGgrid.npz')
      x,y,z = grid['x'],grid['y'],grid['z']
      for i in range(0,end,skip):
        sol_str = 'npsol' + str(i) + '.npz'
        if (os.path.isfile(sol_str)):
          print('found ' + sol_str)
          sol = np.load('npsol' + str(i) + '.npz')
          string = 'PVsol' + str(i)
          gridToVTK(string, x,y,z, pointData = {"rho" : sol['U'][0] , \
            "u" : sol['U'][1]/sol['U'][0] , "v" : sol['U'][2], "w" : sol['U'][3]/sol['U'][0], \
            "rhoE" : sol['U'][4]} )
    self.writeAllToParaview = writeAllToParaview

class postProcessor_subscale:
  def __init__(self):
    def writeAllToParaview(self,start=0,end=100000,skip=1):
      grid = np.load('../DGgrid.npz')
      x,y,z = grid['x'],grid['y'],grid['z']
      for i in range(0,end,skip):
        sol_str = 'npsol_sgs' + str(i) + '.npz'
        if (os.path.isfile(sol_str)):
          print('found ' + sol_str)
          sol = np.load('npsol_sgs' + str(i) + '.npz')
          string = 'PVsol_sgs' + str(i)
          gridToVTK(string, x,y,z, pointData = {"PLQLu_rho" : sol['PLQLu'][0] , \
            "PLQLu_rhoU" : sol['PLQLu'][1] , "PLQLu_rhoV" : sol['PLQLu'][2], "PLQLu_rhoW" : sol['PLQLu'][3], \
            "PLQLu_rhoE" : sol['PLQLu'][4]} )
    self.writeAllToParaview = writeAllToParaview


eqn_type = sys.argv[1:][:]
check = 0
if (eqn_type[0] == 'Entropy'):
  print('Post Processing for Entropy Variables')
  postProcess = postProcessorEntropy()
  check = 1
if (eqn_type[0] == 'Scalar'):
  print('Post Processing for Scalar Equation')
  postProcess = postProcessorScalar()
  check = 1
if (eqn_type[0] == 'Navier-Stokes'):
  print('Post Processing for Navier-Stokes Equations')
  postProcess = postProcessor()
  check = 1
if (eqn_type[0] == 'PLQLu'):
  print('Post Processing for PLQLu subscales')
  postProcess = postProcessor_subscale()
  check = 1
if (check == 0):
  print('Error, ' + eqn_type[0] + ' not found')
postProcess.writeAllToParaview(postProcess)
