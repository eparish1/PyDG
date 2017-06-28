import numpy as np

def computeEnergy(main,T,Y,u,v,w):
  R = 8.314
  T0 = 298.15
  n_reacting = np.size(main.delta_h0)
  Cv = np.einsum('i...,ijk...->jk...',main.Cv,Y)
  Winv =  np.einsum('i...,ijk...->jk...',1./main.W,Y)
  e = Cv*(T - T0) - R*T0*Winv
  for i in range(0,np.size(main.delta_h0)):
    e += main.delta_h0[i]*Y[i]
  e += 0.5*(u**2 + v**2 + w**2)
  return e 

def computeTemperature(main,u):
  R = 8.314
  T0 = 298.15
  n_reacting = np.size(main.delta_h0)
  #Cv = 0
  #Winv = 0
  #for i in range(0,n_reacting):
  #  Cv += main.Cv[i]*u[5+i] #Cv of the mixture
  #  Winv += u[5+i]/main.W[i] #mean molecular weight
  Cv = np.einsum('i...,ijk...->jk...',main.Cv,u[5::]/u[0])
  Winv =  np.einsum('i...,ijk...->jk...',1./main.W,u[5::]/u[0])
  # sensible + chemical
  T = u[4]/u[0] - 0.5/u[0]**2*( u[1]**2 + u[2]**2 + u[3]**2 )
  # subtract formation of enthalpy
  for i in range(0,np.size(main.delta_h0)):
    T -= main.delta_h0[i]*u[5+i]/u[0]
  T += R * T0 * Winv 
  T /= Cv
  T += T0

  return T


def computePressure(main,u,T):
  R = 8.314 
  Winv =  np.einsum('i...,ijk...->jk...',1./main.W,u[5::])
  p = u[0]*R*Winv*T
  return p


def computePressure_and_Temperature(main,u):
  R = 8.314
  T0 = 298.15
  n_reacting = np.size(main.delta_h0)
  #Cv = 0
  #Winv = 0
  #for i in range(0,n_reacting):
  #  Cv += main.Cv[i]*u[5+i] #Cv of the mixture
  #  Winv += u[5+i]/main.W[i] #mean molecular weight
  Cv = np.einsum('i...,ijk...->jk...',main.Cv,u[5::]/u[0])
  Winv =  np.einsum('i...,ijk...->jk...',1./main.W,u[5::]/u[0])
  # sensible + chemical
  T = u[4]/u[0] - 0.5/u[0]**2*( u[1]**2 + u[2]**2 + u[3]**2 )
  # subtract formation of enthalpy
  for i in range(0,np.size(main.delta_h0)):
    T -= main.delta_h0[i]*u[5+i]
  T += R * T0 * Winv 
  T /= Cv
  T += T0
  p = u[0]*R*Winv*T
  return p,T

