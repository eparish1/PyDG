import numpy as np

def conservedScalarX(u,Z,f,args): 
  #f = np.zeros(np.shape(u))
  f[:] = u[1]*Z/(u[0]) 


def conservedScalarY(u,Z,f,args):
  f[:] = u[2]*Z/u[0] 


def conservedScalarZ(u,Z,f,args):
  f[:] = u[3]*Z/u[0] 

