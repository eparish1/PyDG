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
      n_blocks = 0
      check = 0
      x = []
      y = []
      z = []
      
      while (check == 0):
        grid_str = '../DGgrid_block' + str(n_blocks) + '.npz'
        if (os.path.isfile(grid_str)):
          grid = np.load('../DGgrid_block' + str(n_blocks) + '.npz')
          x.append(grid['x'])
          y.append(grid['y'])
          z.append(grid['z'])
          n_blocks += 1

        else:
          check = 1
      
      for i in range(0,end,skip):
        
        for j in range(1,n_blocks):
          
          sol_str = 'npsol_block' + str(j) + '_'  + str(i) + '.npz'
          
          if (os.path.isfile(sol_str)):
            
            print('found ' + sol_str)
            sol = np.load('npsol_block' + str(j) + '_' + str(i) + '.npz')
            string = 'PVsol' + str(j) + '_' + str(i)
            p      = (1.4 - 1.)*(sol['U'][4] - 0.5*sol['U'][1]**2/sol['U'][0] - 0.5*sol['U'][2]**2/sol['U'][0] - 0.5*sol['U'][3]**2/sol['U'][0])
            rho    = sol['U'][0]
            u      = sol['U'][1]/sol['U'][0]
            v      = sol['U'][2]/sol['U'][0]
            w      = sol['U'][3]/sol['U'][0]
            rhoE   = sol['U'][4]

            gridToVTK(string, x[j], y[j], z[j], pointData = {"rho" : rho,  \
                                                               "u" : u,    \
                                                               "v" : v,    \
                                                               "w" : w,    \
                                                            "rhoE" : rhoE, \
                                                               "p" : p} )
    
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
if (check == 0):
  print('Error, ' + eqn_type[0] + ' not found')
postProcess.writeAllToParaview(postProcess)
