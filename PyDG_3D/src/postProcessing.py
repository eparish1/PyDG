import numpy as np
from evtk.hl import gridToVTK
import numpy as np
import os.path


class postProcessor:
  def __init__(self):
    def writeAllToParaview(self,start=0,end=10000,skip=1):
      grid = np.load('../DGgrid.npz')
      x,y,z = np.meshgrid(grid['x'],grid['y'],grid['z'],indexing='ij')
      for i in range(0,10000,skip):
        sol_str = 'npsol' + str(i) + '.npz'
        if (os.path.isfile(sol_str)):
          print('found ' + sol_str)
          sol = np.load('npsol' + str(i) + '.npz')
          string = 'PVsol' + str(i)
          gridToVTK(string, x,y,z, pointData = {"rho" : sol['U'][0] , \
            "u" : sol['U'][1]/sol['U'][0] , "v" : sol['U'][2], "w" : sol['U'][3]/sol['U'][0], \
            "rhoE" : sol['U'][4]} )
    self.writeAllToParaview = writeAllToParaview

postProcess = postProcessor()
postProcess.writeAllToParaview(postProcess)
