import numpy as np

def periodic_bc(Ue,UBC,args,gas):
   return UBC 

def isothermalwall_bc(Ue,UBC,args,gas):
  uw = args[0]
  vw = args[1]
  ww = args[2]
  Tw = args[3]
  gamma = gas.gamma
  y0 = 5.
  x0 = 5. 
  Cv = gas.Cv
  Cp = gas.Cp
  R = gas.R

  UBC[:] = 0.
  p = (gamma - 1.)*(Ue[4] - 0.5*Ue[1]**2/Ue[0] - 0.5*Ue[2]**2/Ue[0] - 0.5*Ue[3]**2/Ue[0]) #extraploate pressure
  #print(p/(Ue[0]*R))
  T = Tw
  E = Cv*T  + 0.5*(uw**2 + vw**2 + ww**2) #wall velocity is zero
  #p = rho R T
  UBC[0] = p/(R*T)
  UBC[1] = UBC[0]*uw
  UBC[2] = UBC[0]*vw
  UBC[3] = UBC[0]*ww
  UBC[4] = UBC[0]*E
  return UBC

def adiabaticwall_bc(Ue,UBC,args,gas):
  uw = args[0]
  vw = args[1]
  ww = args[2]
  gamma = gas.gamma
  y0 = 5.
  x0 = 5. 
  Cv = gas.Cv
  Cp = gas.Cp
  R = gas.R

  UBC[:] = 0.
  p = (gamma - 1.)*(Ue[4] - 0.5*Ue[1]**2/Ue[0] - 0.5*Ue[2]**2/Ue[0] - 0.5*Ue[3]**2/Ue[0]) #extraploate pressure
  #print(p/(Ue[0]*R))
  T = p/(Ue[0]*gas.R) #extraplate temperature (zero heat flux)
  E = Cv*T  + 0.5*(uw**2 + vw**2 + ww**2) #wall velocity is zero
  #p = rho R T
  UBC[0] = p/(R*T)
  UBC[1] = UBC[0]*uw
  UBC[2] = UBC[0]*vw
  UBC[3] = UBC[0]*ww
  UBC[4] = UBC[0]*E
  return UBC
