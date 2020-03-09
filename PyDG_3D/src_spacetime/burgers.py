import numpy as np
import numexpr as ne 

##### =========== Contains all the fluxes and physics neccesary to solve the Navier-Stokes equations within a DG framework #### ============


###### ====== Inviscid Fluxes Fluxes and Eigen Values (Eigenvalues currently not in use) ==== ############
def evalFluxXYZ_burgers(eqns,main,u,fx,fy,fz,args):
  #print(eqns.params)
  es = 1.e-30
  #u = rho_eta/rho eta U
  fx[0] = u[1]
  fx[1] = hU*hU/h + 0.5*h**2*g
  fx[2] = hU*hV/h

  fy[0] = u[2]
  fy[1] = hU*hV/(h)
  fy[2] = hV*hV/(h) + 0.5*h**2*g 

def evalFluxXYZ_shallowWaterLin(eqns,main,u,fx,fy,fz,args):
  up = args[0]
  es = 1.e-30
  g = eqns.params[0] 
  h = u[0]
  hU = u[1]
  hV = u[2]
  #u = rho_eta/rho eta U
  fx[0] = up[1]
  fx[1] = 2.*hU/h*up[1] + (g*h - hU**2/h**2)*up[0]
  fx[2] = hU/h*up[2] + hV/h*up[1] - hU*hV*up[0]/h**2

  fy[0] = up[2]
  fy[1] = hU/h*up[2] + hV/h*up[1] - hU*hV*up[0]/h**2
  fy[2] = 2.*hV/h*up[2] + (g*h - hV**2/h**2)*up[0]

#==================== Numerical Fluxes for the Faces =====================
#== rusanov flux
#== Roe flux

def swe_rusanovFlux(eqns,F,main,UL,UR,n,args=None):

# PURPOSE: This function calculates the flux for the SWE equations
# using the rusanov flux function
#
# INPUTS:
#    UL: conservative state vector in left cell
#    UR: conservative state vector in right cell
#    n: normal pointing from the left cell to the right cell
#
# OUTPUTS:
#  F   : the flux out of the left cell (into the right cell)
#  smag: the maximum propagation speed of disturbance
#
  g = eqns.params[0] 
  #process left state
  hL = UL[0]
  uL = UL[1]/hL
  vL = UL[2]/hL

  unL = uL*n[0] + vL*n[1]

  pL = 0.5*g*hL**2
  FL = np.zeros(np.shape(UL),dtype=UL.dtype)
  FL[0] = hL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]

  # process right state
  hR = UR[0]
  uR = UR[1]/hR
  vR = UR[2]/hR
  unR = uR*n[0] + vR*n[1] 
  pR = 0.5*g*hR**2
  # right flux
  FR = np.zeros(np.shape(UR),dtype=UR.dtype)
  FR[0] = hR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]

  # difference in states
  du = UR - UL
  # rho average
  hm = 0.5*(hL + hR)
  um = (unL*np.sqrt(hL) + unR*np.sqrt(hR) )/(np.sqrt(hL) + np.sqrt(hR) ) 
  #% eigenvalues
  smax = np.abs(um) + np.abs(np.sqrt(g*hm))
  #smax = np.maximum(np.abs(l[0]),np.abs(l[1]))
  # flux assembly
  #F = np.zeros(np.shape(FL))  # for allocation
  F[0]    = 0.5*(FL[0]+FR[0])-0.5*smax*(UR[0] - UL[0])
  F[1]    = 0.5*(FL[1]+FR[1])-0.5*smax*(UR[1] - UL[1])
  F[2]    = 0.5*(FL[2]+FR[2])-0.5*smax*(UR[2] - UL[2])
  return F


def swe_rusanovFlux_lin(eqns,F,main,UL,UR,n,args):
  upL = args[0]
  upR = args[1]
# PURPOSE: This function calculates the flux for the SWE equations
# using the rusanov flux function
#
# INPUTS:
#    UL: conservative state vector in left cell
#    UR: conservative state vector in right cell
#    n: normal pointing from the left cell to the right cell
#
# OUTPUTS:
#  F   : the flux out of the left cell (into the right cell)
#  smag: the maximum propagation speed of disturbance
#
  g = eqns.params[0] 
  #process left state
  hL = UL[0]
  uL = UL[1]/hL
  vL = UL[2]/hL

  unL = uL*n[0] + vL*n[1]

  pL = 0.5*g*hL**2
  FL = np.zeros(np.shape(UL),dtype=UL.dtype)
  FL[0] = n[0]*upL[1] + n[1]*upL[2]
  FL[1] = UL[1]/UL[0]*upL[2]*n[1] + 1./UL[0]*(UL[2]*n[1] + 2.*UL[1]*n[0])*upL[1] + \
         (UL[1]/UL[0]*-unL + g*UL[0]*n[0])*upL[0]
  FL[2] = UL[2]/UL[0]*upL[1]*n[0] + 1./UL[0]*(2.*UL[2]*n[1] + UL[0]*n[0])*upL[2] + \
         (UL[2]/UL[0]*-unL + g*UL[0]*n[1])*upL[0]


  # process right state
  hR = UR[0]
  uR = UR[1]/hR
  vR = UR[2]/hR
  unR = uR*n[0] + vR*n[1] 
  pR = 0.5*g*hR**2
  # right flux
  FR = np.zeros(np.shape(UR),dtype=UR.dtype)
  FR[0] = n[0]*upR[1] + n[1]*upR[2]
  FR[1] = UR[1]/UR[0]*upR[2]*n[1] + 1./UR[0]*(UR[2]*n[1] + 2.*UR[1]*n[0])*upR[1] + \
         (UR[1]/UR[0]*-unR + g*UR[0]*n[0])*upR[0]
  FR[2] = UR[2]/UR[0]*upR[1]*n[0] + 1./UR[0]*(2.*UR[2]*n[1] + UR[0]*n[0])*upR[2] + \
         (UR[2]/UR[0]*-unR + g*UR[0]*n[1])*upR[0]

  # difference in states
  # rho average
  hm = 0.5*(hL + hR)
  um = (unL*np.sqrt(hL) + unR*np.sqrt(hR) )/(np.sqrt(hL) + np.sqrt(hR) ) 
  unL_h0 = -1./UL**2*(UL[1]*n[0] + UL[2]*n[1] )
  unR_h0 = -1./UR**2*(UR[1]*n[0] + UR[2]*n[1] )

  ## Grads of roe average
  #hm_q0L = 0.5


  um_q0L = ( -hL*unL_h0/(2.*np.sqrt(UL[0]) ) + np.sqrt(hL[0])*unL_h0 ) / (np.sqrt(UR[0]) + np.sqrt(UR[1]) ) - ( UR[0]*np.sqrt(UR[0])*-unR_h0 + UL[0]*np.sqrt(UL[0])*-unL_h0 )/(2.*np.sqrt(UL[0])*(np.sqrt(UR[0]) + np.sqrt(UL[0]) )**2 )
  um_q1L = n[0]/(np.sqrt(UL[0])*(np.sqrt(UR[0]) + np.sqrt(UL[0]) ) )
  um_q2L = n[1]/(np.sqrt(UL[0])*(np.sqrt(UR[0]) + np.sqrt(UL[0]) ) )

  um_q0R = ( -hR*unR_h0/(2.*np.sqrt(UR[0]) ) + np.sqrt(hR[0])*unR_h0 ) / (np.sqrt(UR[0]) + np.sqrt(UR[1]) ) - ( UR[0]*np.sqrt(UR[0])*-unR_h0 + UL[0]*np.sqrt(UL[0])*-unL_h0 )/(2.*np.sqrt(UR[0])*(np.sqrt(UR[0]) + np.sqrt(UL[0]) )**2 )
  um_q1R = n[0]/(np.sqrt(UR[0])*(np.sqrt(UR[0]) + np.sqrt(UL[0]) ) )
  um_q2R = n[1]/(np.sqrt(UR[0])*(np.sqrt(UR[0]) + np.sqrt(UL[0]) ) )


  #% eigenvalues
  smax = np.abs(um) + np.abs(np.sqrt(g*hm))
  dsmax = um/np.abs(um)*(um_q0L*upL[0] + um_q1L*upL[1] + um_q2L*upL[2] + \
                         um_q0R*upR[0] + um_q1R*upR[1] + um_q2R*upR[2] ) +  \
                         + g/(2.*np.sqrt(hm))*(0.5*upL[0] + 0.5*upR[0])

  #smax = np.maximum(np.abs(l[0]),np.abs(l[1]))
  # flux assembly
  #F = np.zeros(np.shape(FL))  # for allocation
  F[0]    = 0.5*(FL[0]+FR[0])-0.5*dsmax*(UR[0] - UL[0]) - 0.5*smax*(upR[0] - upL[0])
  F[1]    = 0.5*(FL[1]+FR[1])-0.5*dsmax*(UR[1] - UL[1]) - 0.5*smax*(upR[1] - upL[1])
  F[2]    = 0.5*(FL[2]+FR[2])-0.5*dsmax*(UR[2] - UL[2]) - 0.5*smax*(upR[2] - upL[2])
  return F


