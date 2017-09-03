import numpy as np
def TGVIC_W(x,y,z,main):
  Lx = x[-1] - x[0]
  Ly = y[-1] - y[0]
  Lz = z[-1] - z[0]
  Minf = 0.2
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  gamma = main.gas.gamma
  T0 = 1./gamma
  R = main.gas.R #1
  rho = 1.
  p0 = rho*R*T0
  a = np.sqrt(gamma*R*T0) 
  V0 = Minf*a
  Cv = 5./2.*R
  u = V0*np.sin(x*2.*np.pi/Lx)*np.cos(z*2.*np.pi/Lz)*np.sin(y*2.*np.pi/Ly)
  w = -V0*np.cos(x*2.*np.pi/Lx)*np.sin(z*2.*np.pi/Lz)*np.sin(y*2.*np.pi/Ly)
  v = 0
  p = p0 + rho*V0**2/16.*(np.sin(2.*x*2.*np.pi/Lx) + np.sin(2.*z*2.*np.pi/Lz) )*(np.sin(2.*y*2.*np.pi/Ly) + 2.)
  T = p/(rho*R)
  E = Cv*T + 0.5*(u**2 + v**2 + w**2)
  q = np.zeros((5,nqx,nqy,nqz,Nelx,Nely,Nelz))
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*w
  q[4] = rho*E
  return q

def TGVIC_incompressible(x,y,z,main):
  Lx = 2.*np.pi 
  Ly = 2.*np.pi
  Lz = 2.*np.pi
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  q = np.zeros((4,nqx,nqy,nqz,Nelx,Nely,Nelz))
  u = np.sin(x*2.*np.pi/Lx)*np.cos(y*2.*np.pi/Ly)*np.cos(z*2.*np.pi/Lz)
  v = -np.cos(x*2.*np.pi/Lx)*np.sin(y*2.*np.pi/Ly)*np.cos(z*2.*np.pi/Lz)
  w = 0
  q[0] = u 
  q[1] = v
  q[2] = w
  q[3] = 1./16.*(np.sin(2.*x*2.*np.pi/Lx) + np.sin(2.*z*2.*np.pi/Lz) )*(np.sin(2.*y*2.*np.pi/Ly) + 2.)

  return q

def TGVIC(x,y,z,main):
  Lx = 2.*np.pi 
  Ly = 2.*np.pi
  Lz = 2.*np.pi
  Minf = 0.2
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  gamma = main.gas.gamma
  T0 = 1./gamma
  R = main.gas.R #1
  rho = 1.
  p0 = rho*R*T0
  a = np.sqrt(gamma*R*T0) 
  V0 = Minf*a
  Cv = 5./2.*R
  u = V0*np.sin(x*2.*np.pi/Lx)*np.cos(y*2.*np.pi/Ly)*np.cos(z*2.*np.pi/Lz)
  v = -V0*np.cos(x*2.*np.pi/Lx)*np.sin(y*2.*np.pi/Ly)*np.cos(z*2.*np.pi/Lz)
  w = 0
  p = p0 + rho*V0**2/16.*(np.cos(2.*x*2.*np.pi/Lx) + np.cos(2.*y*2.*np.pi/Ly) )*(np.cos(2.*z*2.*np.pi/Lz) + 2.)
  T = p/(rho*R)
  E = Cv*T + 0.5*(u**2 + v**2 + w**2)
  q = np.zeros((5,nqx,nqy,nqz,Nelx,Nely,Nelz))
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*w
  q[4] = rho*E
  return q


def shocktubeIC(x,y,z,main):
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  q = np.zeros((5,nqx,nqy,nqz,Nelx,Nely,Nelz))
  gamma = main.gas.gamma 
  Cv = main.gas.Cv
  Cp = Cv*gamma
  R = Cp - Cv
  p = np.zeros((nqx,nqy,nqz,Nelx,Nely,Nelz))
  rho = np.zeros(np.shape(p))
  u = np.zeros(np.shape(rho))
  v = np.zeros(np.shape(u))
  w = np.zeros(np.shape(u))
  E = np.zeros(np.shape(u))
  p[:] = 1.
  #p[:,:,:,0:Nelx/2,:,:] = 1.
  p[x>0.5] = 0.1
  rho[:] = 1
  rho[x>0.5] = 0.125
#  rho[:,:,:,0:Nelx/2,:,:] = 1.
#  rho[:,:,:,Nelx/2::,:,:] = 0.8
  T = p/(rho*R)
  E[:] = Cv*T + 0.5*(u**2 + v**2)
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*w
  q[4] = rho*E
  return q

def zeroFSIC_incomp(x,y,z,main):
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  q = np.zeros((3,nqx,nqy,nqz,Nelx,Nely,Nelz))
  return q


def zeroFSIC(x,y,z,main):
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  q = np.zeros((5,nqx,nqy,nqz,Nelx,Nely,Nelz))
  gamma = main.gas.gamma 
  Cv = main.gas.Cv
  Cp = Cv*gamma
  R = Cp - Cv
  T = np.zeros((nqx,nqy,nqz,Nelx,Nely,Nelz))
  rho = np.zeros(np.shape(T))
  u = np.zeros(np.shape(rho))
  v = np.zeros(np.shape(u))
  w = np.zeros(np.shape(u))
  E = np.zeros(np.shape(u))
  T[:] = 1./gamma
  rho[:] = T**(1./(gamma - 1.))
  E[:] = Cv*T + 0.5*(u**2 + v**2)
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*w
  q[4] = rho*E
  return q

def vortexICS(x,y,z,main):
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  q = np.zeros((5,nqx,nqy,nqz,Nelx,Nely,Nelz))
  gamma = main.gas.gamma
  y0 = 5.
  x0 = 5.
  Cv = main.gas.Cv
  Cp = Cv*gamma
  R = Cp - Cv
  T = np.zeros((nqx,nqy,nqz,Nelx,Nely,Nelz))
  rho = np.zeros(np.shape(T))
  u = np.zeros(np.shape(rho))
  v = np.zeros(np.shape(u))
  w = np.zeros(np.shape(u))
  E = np.zeros(np.shape(u))
  r = ( (x - x0)**2 + (y - y0)**2 )**0.5
  beta = 5.
  pi = np.pi
  T[:] = 1. - (gamma - 1.)*beta**2/(8.*gamma*pi**2)*np.exp(1. - r**2)

  rho[:] = T**(1./(gamma - 1.))
  u[:] = 1. + beta/(2.*pi)*np.exp( (1. - r**2)/2.)*-(y - y0)
  v[:] = 1. +  beta/(2.*pi)*np.exp( (1. - r**2)/2.)*(x - x0)
  E[:] = Cv*T + 0.5*(u**2 + v**2)
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*w
  q[4] = rho*E
  return q

def vortexICS_entropy(x,y,z,main):
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  q = np.zeros((5,nqx,nqy,nqz,Nelx,Nely,Nelz))
  gamma = main.gas.gamma
  y0 = 5.
  x0 = 5. 
  gamma = 1.4
  Cv = main.gas.Cv
  Cp = Cv*gamma
  R = Cp - Cv
  T = np.zeros((nqx,nqy,nqz,Nelx,Nely,Nelz))
  rho = np.zeros(np.shape(T))
  u = np.zeros(np.shape(rho))
  v = np.zeros(np.shape(u))
  w = np.zeros(np.shape(u))
  s = np.zeros(np.shape(u))
  r = ( (x - x0)**2 + (y - y0)**2 )**0.5
  beta = 5.
  pi = np.pi
  T[:] = 1. - (gamma - 1.)*beta**2/(8.*gamma*pi**2)*np.exp(1. - r**2)

  rho[:] = T**(1./(gamma - 1.))
  p = rho*R*T
  u[:] = 1. + beta/(2.*pi)*np.exp( (1. - r**2)/2.)*-(y - y0)
  v[:] = 1. +  beta/(2.*pi)*np.exp( (1. - r**2)/2.)*(x - x0)
  s[:] = np.log(p) - gamma*np.log(rho)
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*w
  q[4] = rho*s
  return q



def TGVIC_2D(x,y,z,main):
  Lx = 2.*np.pi 
  Ly = 2.*np.pi
  Lz = 2.*np.pi
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  u = np.cos(x)*np.sin(y)
  v = -np.sin(x)*np.cos(y)
  w = 0
  q = np.zeros((4,nqx,nqy,nqz,Nelx,Nely,Nelz))
  q[0] = u 
  q[1] = v
  q[2] = w
  q[3] = 0.
  return q


def shocktubeIC(x,y,z,main):
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  q = np.zeros((5,nqx,nqy,nqz,Nelx,Nely,Nelz))
  gamma = main.gas.gamma 
  Cv = main.gas.Cv
  Cp = Cv*gamma
  R = Cp - Cv
  p = np.zeros((nqx,nqy,nqz,Nelx,Nely,Nelz))
  rho = np.zeros(np.shape(p))
  u = np.zeros(np.shape(rho))
  v = np.zeros(np.shape(u))
  w = np.zeros(np.shape(u))
  E = np.zeros(np.shape(u))
  p[:] = 1.
  #p[:,:,:,0:Nelx/2,:,:] = 1.
  p[x>0.5] = 0.1
  rho[:] = 1
  rho[x>0.5] = 0.125
#  rho[:,:,:,0:Nelx/2,:,:] = 1.
#  rho[:,:,:,Nelx/2::,:,:] = 0.8
  T = p/(rho*R)
  E[:] = Cv*T + 0.5*(u**2 + v**2)
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*w
  q[4] = rho*E
  return q

