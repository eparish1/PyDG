import numpy as np
def TGVIC(x,y,z):
  rho = 1.
  rho0 = 1.
  nq,nq,nq,Nelx,Nely,Nelz = np.shape(x)
  u = np.sin(x)*np.cos(y)*np.cos(z)
  v = -np.cos(x)*np.sin(y)*np.cos(z)
  w = 0
  speed_of_sound = 10.
  gamma = 1.4
  R = 287.
  p0 = speed_of_sound**2*gamma/rho0
  Cv = 5./2.*R
  p = p0 + rho0/16.*(np.cos(2.*x) + np.cos(2.*y) )*(np.cos(2.*z) + 2.)
  T = p/(rho*R)
  E = Cv*T + 0.5*(u**2 + v**2 + w**2)
  q = np.zeros((5,nq,nq,nq,Nelx,Nely,Nelz))
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*w
  q[4] = rho*E
  return q
