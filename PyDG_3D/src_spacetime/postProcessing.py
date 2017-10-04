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

eqn_type = sys.argv[1:][:]

if (eqn_type[0] == 'Entropy'):
  print('Post Processing for Entropy Variables')
  postProcess = postProcessorEntropy()
else:
  postProcess = postProcessor()

postProcess.writeAllToParaview(postProcess)
