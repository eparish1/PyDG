def HLLCFlux_reacting(F,main,UL,UR,n,args=None):
# Calculates HLLC for variable species navier-stokes equations
# implementation based on paper:
#   Discontinuous Galerkin method for multicomponent chemically reacting flows and combustion
#   Journal of Computational Physics 270 (2014) 105-137
  #process left state
  rhoL = UL[0]
  uL = UL[1]/rhoL
  vL = UL[2]/rhoL
  wL = UL[3]/rhoL
  pL = computePressure_CPG(main,UL)
  rHL = UL[4] + pL
  HL = rHL/rhoL
  unL = uL*n[0] + vL*n[1] + wL*n[2]
  # process right state
  rhoR = UR[0]
  uR = UR[1]/rhoR
  vR = UR[2]/rhoR
  wR = UR[3]/rhoR
  pR = computePressure_CPG(main,UR)
  rHR = UR[4] + pR
  HR = rHR/rhoR
  unR = uR*n[0] + vR*n[1] + wR*n[2]
  # compute speed of sounds 
  R = 8314.4621/1000.
  Y_N2_R = 1. - np.sum(UR[5::]/UR[None,0],axis=0)
  WinvR =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],UR[5::]/UR[None,0]) + 1./main.W[-1]*Y_N2_R
  CpR = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],UR[5::]/UR[None,0]) + main.Cp[-1]*Y_N2_R
  CvR = CpR - R*WinvR
  gammaR = CpR/CvR
  cR = np.sqrt(gammaR*pR/UR[0])

  Y_N2_L = 1. - np.sum(UL[5::]/UL[None,0],axis=0)
  WinvL =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],UL[5::]/UL[None,0]) + 1./main.W[-1]*Y_N2_L
  CpL = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],UL[5::]/UL[None,0]) + main.Cp[-1]*Y_N2_L
  CvL = CpL - R*WinvL
  gammaL = CpL/CvL

  cL = np.sqrt(gammaL*pL/UL[0])

  # make computations for HLLC
  rho_bar = 0.5*(UL[0] + UR[0])
  c_bar = 0.5*(cL + cR)

  p_pvrs = 0.5*(pL + pR) - 0.5*(unR - unL)*rho_bar*c_bar
  p_star = np.fmax(0,p_pvrs)

  qkR = np.ones(np.shape(pR))
  #qkR[p_star > pR] = ((1. + (gamma_star + 1.)/(2.*gamma_star) * (p_star/pR - 1.))[p_star > pR])**0.5 
  qkR[p_star > pR] = ((1. + (gammaR + 1.)/(2.*gammaR) * (p_star/pR - 1.))[p_star > pR])**0.5 
  qkL = np.ones(np.shape(pL))
  #qkL[p_star > pL] = ((1. + (gamma_star + 1.)/(2.*gamma_star) * (p_star/pL - 1.))[p_star > pL])**0.5 
  qkL[p_star > pL] = ((1. + (gammaL + 1.)/(2.*gammaL) * (p_star/pL - 1.))[p_star > pL])**0.5 

  SL = unL - cL*qkL
  SR = unR + cR*qkR
  S_star = ( pR - pL + rhoL*unL*(SL - unL) - rhoR*unR*(SR - unR) ) / ( rhoL*(SL - unL) - rhoR*(SR - unR) ) 
  # Compute UStar state for HLLC
  #left state
  srat = (SL - unL)/(SL - S_star)
  UstarL = np.zeros(np.shape(UL))
  UstarL[0] = srat*( UL[0] )
  UstarL[1] = srat*( UL[1] + rhoL*(S_star - unL)*n[0] )
  UstarL[2] = srat*( UL[2] + rhoL*(S_star - unL)*n[1] )
  UstarL[3] = srat*( UL[3] + rhoL*(S_star - unL)*n[2] )
  UstarL[4] = srat*( UL[4] + (S_star - unL)*(rhoL*S_star + pL/(SL - unL) ) )
  UstarL[5::] = srat[None,:]*( UL[5::] ) 
  #right state
  srat[:] = (SR - unR)/(SR - S_star)
  UstarR = np.zeros(np.shape(UR))
  UstarR[0] = srat*( UR[0] )
  UstarR[1] = srat*( UR[1] + rhoR*(S_star - unR)*n[0] )
  UstarR[2] = srat*( UR[2] + rhoR*(S_star - unR)*n[1] )
  UstarR[3] = srat*( UR[3] + rhoR*(S_star - unR)*n[2] )
  UstarR[4] = srat*( UR[4] + (S_star - unR)*(rhoR*S_star + pR/(SR - unR) ) )
  UstarR[5::] = srat[None,:]*( UR[5::] )

  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = rhoL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]
  FL[3] = UL[3]*unL + pL*n[2]
  FL[4] = rHL*unL
  FL[5::] = UL[5::]*unL[None,:]
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = rhoR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = UR[3]*unR + pR*n[2]
  FR[4] = rHR*unR
  FR[5::] = UR[5::]*unR[None,:]

  # Assemble final HLLC Flux
  indx0 = 0<SL
  indx1 = (SL <= 0) & (0 < S_star)
  indx2 = (S_star <= 0) & (0 < SR)
  indx3 = SR <= 0
  F[:] = 0. 
  F[:,indx0] = FL[:,indx0]
  F[:,indx1] = (FL + SL[None,:]*(UstarL - UL) )[:,indx1]
  F[:,indx2] = (FR + SR[None,:]*(UstarR - UR) )[:,indx2]
  F[:,indx3] = FR[:,indx3]
  return F

