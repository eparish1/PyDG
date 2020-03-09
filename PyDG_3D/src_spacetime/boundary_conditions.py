import numpy as np

def viscouswall_bc(Ue,UBC,args,main,normals):
  cut = 30
#  cut = 35
  gamma = 1.4
  #uw = args[0]
  #vw = args[1]
  #ww = args[2]
  #Tw = args[3]
  #cut = args[4]
  gamma = main.gas.gamma
  Cv = main.gas.Cv
  Cp = main.gas.Cp
  R = main.gas.R

  u_plus = Ue[1]/Ue[0]
  v_plus = Ue[2]/Ue[0]
  w_plus = Ue[3]/Ue[0]
  Un = u_plus*normals[0,None,None,None,:,:,None] + v_plus*normals[1,None,None,None,:,:,None] + w_plus*normals[2,None,None,None,:,:,None]
  u_norm = Un*normals[0,None,None,None,:,:,None]
  v_norm = Un*normals[1,None,None,None,:,:,None]
  w_norm = Un*normals[2,None,None,None,:,:,None]

  u_tang = u_plus - Un*normals[0,None,None,None,:,:,None]
  v_tang = v_plus - Un*normals[1,None,None,None,:,:,None]
  w_tang = w_plus - Un*normals[2,None,None,None,:,:,None]

  pb = (gamma - 1.)*(Ue[4])
  rhoE = pb/(gamma - 1.)

  UBC[:] = 0.
  UBC[0] = Ue[0]
  UBC[1] = Ue[0]*0.#(u_tang - u_norm)
  UBC[2] = Ue[0]*0.#(v_tang - v_norm)
  UBC[3] = Ue[0]*0.#(w_tang - w_norm)
  UBC[4] = rhoE
  return UBC

def inviscidwall_swe_bc(Ue,UBC,args,main,normals):
  gamma = 1.4

  u_plus = Ue[1]/(Ue[0] + 1e-30)
  v_plus = Ue[2]/(Ue[0] + 1e-30)
  Un = u_plus*normals[0,None,None,None,:,:,None] + v_plus*normals[1,None,None,None,:,:,None] 
  u_norm = Un*normals[0,None,None,None,:,:,None]
  v_norm = Un*normals[1,None,None,None,:,:,None]

  u_tang = u_plus - Un*normals[0,None,None,None,:,:,None]
  v_tang = v_plus - Un*normals[1,None,None,None,:,:,None]


  UBC[:] = 0.
  UBC[0] = Ue[0]
  UBC[1] = Ue[0]*(u_tang - u_norm)
  UBC[2] = Ue[0]*(v_tang - v_norm)
  #UBC[3] = Ue[3]
  #UBC[3] = Ue[0]*(w_tang - w_norm)
  return UBC


def inviscidwall_bc(Ue,UBC,args,main,normals):
  cut = 30
#  cut = 35
  gamma = 1.4
  #uw = args[0]
  #vw = args[1]
  #ww = args[2]
  #Tw = args[3]
  #cut = args[4]
  gamma = main.gas.gamma
  Cv = main.gas.Cv
  Cp = main.gas.Cp
  R = main.gas.R

  u_plus = Ue[1]/Ue[0]
  v_plus = Ue[2]/Ue[0]
  w_plus = Ue[3]/Ue[0]
  Un = u_plus*normals[0,None,None,None,:,:,None] + v_plus*normals[1,None,None,None,:,:,None] + w_plus*normals[2,None,None,None,:,:,None]
  u_norm = Un*normals[0,None,None,None,:,:,None]
  v_norm = Un*normals[1,None,None,None,:,:,None]
  w_norm = Un*normals[2,None,None,None,:,:,None]

  u_tang = u_plus - Un*normals[0,None,None,None,:,:,None]
  v_tang = v_plus - Un*normals[1,None,None,None,:,:,None]
  w_tang = w_plus - Un*normals[2,None,None,None,:,:,None]
  pb = (gamma - 1.)*(Ue[4] - 0.5*Ue[0]*(u_tang**2 + v_tang**2 + w_tang**2))
  rhoE = pb/(gamma - 1.) + 0.5*Ue[0]*(u_tang**2 + v_tang**2 + w_tang**2)

  UBC[:] = 0.
  UBC[0] = Ue[0]
  UBC[1] = Ue[0]*(u_tang - u_norm*0.)
  UBC[2] = Ue[0]*(v_tang - v_norm*0.)
  UBC[3] = Ue[0]*(w_tang - w_norm*0.)
  UBC[4] = rhoE[:]

  ## overwrite UE
  Ue[1] = Ue[0]*(u_tang)
  Ue[2] = Ue[0]*(v_tang)
  Ue[3] = Ue[0]*(w_tang)
  Ue[4] = rhoE[:]
  return UBC




def shuOscherBC(Ue,UBC,args,main,normals):
  UBC[0] = 3.8571430000000211
  UBC[1] = 10.141852232767054
  UBC[2] = 0.
  UBC[3] = 0.
  UBC[4] = 39.166669265042707
  return UBC


def nonreflecting_bc(Ue,UBC,args,main,normals):
  ## Compute right and left eigenvectors
  # A = R Lam L (I do like CFD vol2  pg 77,78)
  gamma = 1.4
  p = (gamma - 1.)*(U[4] - 0.5*U[1]**2/U[0] - 0.5*U[2]**2/U[0] - 0.5*U[3]**2/U[0])
  c = np.sqrt(gamma*p/U[0])
  u = U[1] / U[0]
  v = U[2] / U[0]
  w = U[3] / U[0]
  nx = 1.
  ny = 0.
  nz = 0.
  mx = 1.
  my = 0.
  mz = 0.
  lx = 1.
  ly = 0.
  lz = 0.
  K = gamma - 1. 
  ql = u*1.
  qm = u*1.
  qn = u*1.

  sizeu = np.array([5,5])#np.shape(main.a.u)[0]
  sizeu = np.append(sizeu,np.shape(U[0]))
  L = np.zeros(sizeu)
  R = np.zeros(sizeu)

  q = np.sqrt(u**2 + v**2 + w**2)
  L[0,0] = K*q**2/(4.*c**2) + qn/(2.*c)
  L[0,1] = -(K/(2.*c**2)*u + nx/(2.*c))
  L[0,2] = -(K/(2.*c**2)*v + ny/(2.*c))
  L[0,3] = -(K/(2.*c**2)*w + nz/(2.*c))
  L[0,4] = K/(2.*c**2)
  L[1,0] = 1. - K*q**2/(2.*c**2)
  L[1,1] = K*u/c**2
  L[1,2] = K*v/c**2
  L[1,3] = K*w/c**2
  L[1,4] = -K/c**2
  L[2,0] = K*q**2/(4.*c**2) - qn/(2.*c)
  L[2,1] = -(K/(2.*c**2)*u - nx/(2.*c) )
  L[2,2] = -(K/(2.*c**2)*v - ny/(2.*c) )
  L[2,3] = -(K/(2.*c**2)*w - nz/(2.*c) )
  L[2,4] = K/(2.*c**2)
  L[3,0] = -ql
  L[3,1] = lx
  L[3,2] = ly
  L[3,3] = lz
  L[4,0] = -qm
  L[4,1] = mx
  L[4,2] = my
  L[4,3] = mz

  # compute H in three steps (H = E + p/rho)
  H = (gamma - 1.)*(U[4] - 0.5*U[0]*q**2) #compute pressure
  H += U[4]
  H /= U[0]

  R[0,0] = 1.
  R[0,1] = 1.
  R[0,2] = 1.
  R[1,0] = u - c*nx
  R[1,1] = u
  R[1,2] = u + c*nx
  R[1,3] = lx
  R[1,4] = mx
  R[2,0] = v - c*ny
  R[2,1] = v
  R[2,2] = v + c*ny
  R[2,3] = ly
  R[2,4] = my
  R[3,0] = w - c*nz
  R[3,1] = w
  R[3,2] = w + c*nz
  R[3,3] = lz
  R[3,4] = mz
  R[4,0] = H - qn*c 
  R[4,1] = q**2/2.
  R[4,2] = H + qn*c
  R[4,3] = ql
  R[4,4] = qm

  ###====== To get non-reflecting bcs, cast the flux in chacteristic form
  # L u_t +  L R Lam L U_x = 0
  # L u_t + Lam L U_x = 0
  # set Lam L U_x \approx F_x = 0
  wc  =  np.einsum('ij...,j...->i...',L,F)
 

def dirichlet_bc(Ue,UBC,args,main,normals):
  for i in range(0,np.shape(UBC)[0]):
    UBC[i] = args[i] 
#  UBC[0] = np.sin(main.xG[:,0,:,None,:,0,:,None])
  return UBC

def periodic_bc(Ue,UBC,args,main,normals):
  if (args[2]==1):
      UBC = UBC[:, ::-1, :   , :, ::-1, :   ]

  if (args[3]==1):
      UBC = UBC[:, :   , ::-1, :, :   , ::-1]

  return UBC 

def freestream_temp_bc(Ue,UBC,args,main,normals):
  gamma = 1.4
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
  #E = Cv*T  + 0.5*(uw**2 + vw**2 + ww**2) #wall velocity is zero
  rhoE = p/(gamma - 1.) + 0.5*(uw**2 + vw**2 + ww**2)
  #p = rho R T
  UBC[0] = p/(R*T)
  UBC[1] = UBC[0]*uw
  UBC[2] = UBC[0]*vw
  UBC[3] = UBC[0]*ww
  UBC[4] = rhoE
  return UBC

def freestream_pressure_bc(Ue,UBC,args,main,normals):
  gamma = 1.4
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
  #E = Cv*T  + 0.5*(uw**2 + vw**2 + ww**2) #wall velocity is zero
  rhoE = p/(gamma - 1.) + 0.5*(uw**2 + vw**2 + ww**2)
  #p = rho R T
  UBC[0] = p/(R*T)
  UBC[1] = UBC[0]*uw
  UBC[2] = UBC[0]*vw
  UBC[3] = UBC[0]*ww
  UBC[4] = rhoE
  return UBC

def isothermalwall_bc_kOmega(Ue,UBC,args,main,normals):
  gamma = 1.4
  uw = args[0]
  vw = args[1]
  ww = args[2]
  Tw = args[3]
  gamma = main.gas.gamma
  Cv = main.gas.Cv
  Cp = main.gas.Cp
  R = main.gas.R

  UBC[:] = 0.
  es = 1e-30
  p = (gamma - 1.)*(Ue[4] - 0.5*Ue[1]**2/(Ue[0]+es) - 0.5*Ue[2]**2/(Ue[0]+es) - 0.5*Ue[3]**2/(Ue[0] + es)) #extraploate pressure
  #print(p/(Ue[0]*R))
  T = Tw
  #E = Cv*T  + 0.5*(uw**2 + vw**2 + ww**2) #wall velocity is zero
  rhoE = p/(gamma - 1.) + 0.5*(uw**2 + vw**2 + ww**2)
  #p = rho R T
  UBC[0] = p/(R*T)
  UBC[1] = UBC[0]*uw
  UBC[2] = UBC[0]*vw
  UBC[3] = UBC[0]*ww
  UBC[4] = rhoE
  UBC[5] = 0.
  UBC[6] = 10.
  return UBC



def isothermalwall_bc(Ue,UBC,args,main,normals):
  gamma = 1.4
  uw = args[0]
  vw = args[1]
  ww = args[2]
  Tw = args[3]
  gamma = main.gas.gamma
  Cv = main.gas.Cv
  Cp = main.gas.Cp
  R = main.gas.R

  UBC[:] = 0.
  es = 1e-30
  p = (gamma - 1.)*(Ue[4] - 0.5*Ue[1]**2/(Ue[0]+es) - 0.5*Ue[2]**2/(Ue[0]+es) - 0.5*Ue[3]**2/(Ue[0] + es)) #extraploate pressure
  #print(p/(Ue[0]*R))
  T = Tw
  #E = Cv*T  + 0.5*(uw**2 + vw**2 + ww**2) #wall velocity is zero
  rhoE = p/(gamma - 1.) + 0.5*(uw**2 + vw**2 + ww**2)
  #p = rho R T
  UBC[0] = p/(R*T)
  UBC[1] = UBC[0]*uw
  UBC[2] = UBC[0]*vw
  UBC[3] = UBC[0]*ww
  UBC[4] = rhoE
  return UBC

def incompwall_bc(Ue,UBC,args,main,normals):
  uw = args[0]
  vw = args[1]
  ww = args[2]
  UBC[:] = 0.
  UBC[0] = uw
  UBC[1] = vw
  UBC[2] = ww
  #UBC[3] = Ue[3]
  return UBC

def reflectingwally_bc(Ue,UBC,args,main,normals):
  uw = args[0]
  vw = args[1]
  ww = args[2]
  gamma = main.gas.gamma
  Cv = main.gas.Cv
  Cp = main.gas.Cp
  R = main.gas.R

  UBC[0] = Ue[0] 
  UBC[1] = Ue[1]
  UBC[2] = -Ue[2]
  UBC[3] = Ue[3]
  UBC[4] = Ue[4]
  return UBC


def reflectingwall_bc(Ue,UBC,args,main,normals):
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




def adiabaticwall_bc(Ue,UBC,args,main,normals):
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


def neumann_bc(Ue,UBC,args,main,normals):
  return Ue

def subsonic_outflow(Ue,UBC,args,main,normals):
  p,T = computePressure_and_Temperature(main,Ue)
  pB = p
  S_plus = p/Ue[0]**main.gamma

