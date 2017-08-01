import numpy as np
def dirichlet_bc(Ue,UBC,args,main):
  UBC[0] = args[0] 
  return UBC

def periodic_bc(Ue,UBC,args,main):
   return UBC 

def isothermalwall_bc(Ue,UBC,args,main):
  uw = args[0]
  vw = args[1]
  ww = args[2]
  Tw = args[3]
  gamma = main.gas.gamma
  Cv = main.gas.Cv
  Cp = main.gas.Cp
  R = main.gas.R

  UBC[:] = 0.
  p = (gamma - 1.)*(Ue[4] - 0.5*Ue[1]**2/Ue[0] - 0.5*Ue[2]**2/Ue[0] - 0.5*Ue[3]**2/Ue[0]) #extraploate pressure
  #print(p/(Ue[0]*R))
  T = Tw
  E = Cv*T  + 0.5*(uw**2 + vw**2 + ww**2) #wall velocity is zero
  #p = rho R T
  UBC[0] = p/(R*T)
  print(np.mean(UBC[0]))

  UBC[1] = UBC[0]*uw
  UBC[2] = UBC[0]*vw
  UBC[3] = UBC[0]*ww
  UBC[4] = UBC[0]*E
  return UBC




def reflectingwall_bc(Ue,UBC,args,main):
  uw = args[0]
  vw = args[1]
  ww = args[2]
  gamma = main.gas.gamma
  Cv = main.gas.Cv
  Cp = main.gas.Cp
  R = main.gas.R

  UBC[0] = Ue[0] 
  UBC[1] = -Ue[1]
  UBC[2] = -Ue[2]
  UBC[3] = -Ue[3]
  UBC[4] = Ue[4]
  return UBC




def adiabaticwall_bc(Ue,UBC,args,main):
  uw = args[0]
  vw = args[1]
  ww = args[2]
  gamma = main.gas.gamma
  Cv = main.gas.Cv
  Cp = main.gas.Cp
  R = main.gas.R

  UBC[:] = 0.
  #p = (gamma - 1.)*(Ue[4] - 0.5*Ue[1]**2/Ue[0] - 0.5*Ue[2]**2/Ue[0] - 0.5*Ue[3]**2/Ue[0]) #extraploate pressure
  #print(p/(Ue[0]*R))
  #T = p/(Ue[0]*main.gas.R) #extraplate temperature (zero heat flux)
  T = 1./Cv*(Ue[4]/Ue[0] - 0.5/Ue[0]**2*( Ue[1]**2 + Ue[2]**2 + Ue[3]**2 ) ) #extraplate temperature (zero heat flux)
  E = Cv*T  + 0.5*(uw**2 + vw**2 + ww**2) #wall velocity is zero
  p = (gamma - 1.)*E  #compute pressure at wall using extrapolate temp 
  #p = rho R T
  UBC[0] = p/(R*T)
  UBC[1] = UBC[0]*uw
  UBC[2] = UBC[0]*vw
  UBC[3] = UBC[0]*ww
  UBC[4] = UBC[0]*E
  return UBC


def neumann_bc(Ue,UBC,args,main):
  return Ue

def subsonic_outflow(Ue,UBC,args,main):
  p,T = computePressure_and_Temperature(main,Ue)
  pB = p
  S_plus = p/Ue[0]**main.gamma

