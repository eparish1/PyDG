import numpy as np
#from DG_functions import reconstructEdgesGeneral,faceIntegrate


def getGsNSX(u,main):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  ashape = np.array(np.shape(u[0]))
  ashape = np.insert(ashape,0,nvars)
  ashape = np.insert(ashape,0,nvars)
  G11 = np.zeros(ashape)
  G21 = np.zeros(ashape)
  G31 = np.zeros(ashape)

  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  E = u[4]/u[0]
  vsqr = v1**2 + v2**2 + v3**2
  G11[1,0] = -4./3.*v1
  G11[1,1] = 4./3.
  G11[2,0] = -v2
  G11[2,2] = 1.
  G11[3,0] = -v3
  G11[3,3] = 1.
  G11[4,0] = -(4./3.*v1**2 + v2**2 + v3**2 + gamma/Pr*(E - vsqr) )
  G11[4,1] = (4./3. - gamma/Pr)*v1
  G11[4,2] = (1. - gamma/Pr)*v2
  G11[4,3] = (1. - gamma/Pr)*v3
  G11[4,4] = gamma/Pr

  G21[1,0] = -v2
  G21[1,2] = 1.
  G21[2,0] = 2./3.*v1
  G21[2,1] = -2./3.
  G21[4,0] = -1./3.*v1*v2
  G21[4,1] = -2./3.*v2
  G21[4,2] = v1

  G31[1,0] = -v3
  G31[1,3] = 1.
  G31[3,0] = 2./3.*v1
  G31[3,1] = -2./3.
  G31[4,0] = -1./3.*v1*v3
  G31[4,1] = -2./3.*v3
  G31[4,3] = v1

  G11 = G11*main.mu/u[0]
  G21 = G21*main.mu/u[0]
  G31 = G31*main.mu/u[0]
  return G11,G21,G31

def getGsNSY(u,main):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  ashape = np.array(np.shape(u[0]))
  ashape = np.insert(ashape,0,nvars)
  ashape = np.insert(ashape,0,nvars)
  G12 = np.zeros(ashape)
  G22 = np.zeros(ashape)
  G32 = np.zeros(ashape)

  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  E = u[4]/u[0]
  vsqr = v1**2 + v2**2 + v3**2

  G12[1,0] = 2./3.*v2
  G12[1,2] = -2./3.
  G12[2,0] = -v1
  G12[2,1] = 1.
  G12[4,0] = -1./3.*v1*v2
  G12[4,1] = v2
  G12[4,2] = -2./3.*v1

  G22[1,0] = -v1
  G22[1,1] = 1.
  G22[2,0] = -4./3.*v2
  G22[2,2] = 4./3.
  G22[3,0] = -v3
  G22[3,3] = 1.
  G22[4,0] = -(v1**2 + 4./3.*v2**2 + v3**2 + gamma/Pr*(E - vsqr) )
  G22[4,1] = (1. - gamma/Pr)*v1
  G22[4,2] = (4./3. - gamma/Pr)*v2
  G22[4,3] = (1. - gamm/Pr)*v3
  G22[4,4] = gamma/Pr

  G32[2,0] = -v3
  G32[2,3] = 1.
  G32[3,0] = 2./3.*v2
  G32[3,2] = -2./3.
  G32[4,0] = -1./3.*v2*v3
  G32[4,2] = -2./3.*v3
  G33[4,3] = v2
  G12 = G12*main.mu/u[0]
  G22 = G22*main.mu/u[0]
  G32 = G32*main.mu/u[0]

  return G12,G22,G32

def getGsNSZ(u,main):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  ashape = np.array(np.shape(u[0]))
  ashape = np.insert(ashape,0,nvars)
  ashape = np.insert(ashape,0,nvars)
  G13 = np.zeros(ashape)
  G23 = np.zeros(ashape)
  G33 = np.zeros(ashape)

  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  E = u[4]/u[0]
  vsqr = v1**2 + v2**2 + v3**2

  G13[1,0] = 2./3.*v3
  G13[1,3] = -2./3.
  G13[3,0] = -v1
  G13[3,1] = 1.
  G13[4,0] = -1./3.*v1*v3
  G13[4,1] = v3
  G13[4,3] = -2./3.*v1


  G23[2,0] = 2./3.*v3
  G23[2,3] = -2./3.
  G23[3,0] = -v2
  G23[3,2] = 1.
  G23[4,0] = -1./3.*v2*v3
  G23[4,2] = v3
  G23[4,3] = -2./3.*v2


  G33[1,0] = -v1
  G33[1,1] = 1.
  G33[2,0] = -v2
  G33[2,2] = 1.
  G33[3,0] = -4./3.*v3
  G33[3,3] = 4./3.
  G33[4,0] = -(v1**2 + v2**2 + 4./3.*v3**2 + gamma/Pr*(E - vsqr) )
  G33[4,1] = (1. - gamma/Pr)*v1
  G33[4,2] = (1. - gamma/Pr)*v2
  G33[4,3] = (4./3. - gamma/Pr)*v3
  G33[4,4] = gamma/Pr


  G13 = G13*main.mu/u[0]
  G23 = G23*main.mu/u[0]
  G33 = G33*main.mu/u[0]

  return G13,G23,G33





def evalViscousFluxXNS_IP(main,u,Ux,Uy):
  gamma = 1.4
  Pr = 0.72
  ux = 1./u[0]*Ux[1] - u[1]/u[0]**2*Ux[0]
  ## ->  v_x = 1/rho d/dx(rho v) - rho v /rho^2 rho_x
  vx = 1./u[0]*Ux[2] - u[2]/u[0]**2*Ux[0]
  ## ->  u_y = 1/rho d/dy(rho u) - rho u /rho^2 rho_y
  uy = 1./u[0]*Uy[1] - u[1]/u[0]**2*Uy[0]
  ## ->  v_y = 1/rho d/dy(rho v) - rho v /rho^2 rho_y
  vy = 1./u[0]*Uy[2] - u[2]/u[0]**2*Uy[0]
  ## -> (kT)_x =d/dx[ (mu gamma)/Pr*(E - 1/2 v^2 ) ]
  ## ->        =mu gamma/Pr *[ dE/dx - 0.5 d/dx(v1^2 + v2^2) ]
  ## ->        =mu gamma/Pr *[ dE/dx - (v1 v1_x + v2 v2_x) ]
  ## ->  E_x = 1/rho d/x(rho E) - rho E /rho^2 rho_x
  kTx =( 1./u[0]*Ux[3] - u[3]/u[0]**2*Ux[0] - 1./u[0]*(u[1]*ux + u[2]*vx)  )*main.mu*gamma/Pr
  ## -> (kT)_y =d/dy[ (mu gamma)/Pr*(E - 1/2 v^2 ) ]
  ## ->        =mu gamma/Pr *[ dE/dy - 0.5 d/dy(v1^2 + v2^2) ]
  ## ->        =mu gamma/Pr *[ dE/dy - (v1 v1_y + v2 v2_y) ]
  ## ->  E_x = 1/rho d/y(rho E) - rho E /rho^2 rho_y
  kTy =( 1./u[0]*Uy[3] - u[3]/u[0]**2*Uy[0] - 1./u[0]*(u[1]*uy + u[2]*vy)  )*main.mu*gamma/Pr

  fx = np.zeros(np.shape(u))
  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  fx[1] = main.mu*(4./3.*ux - 2./3.*vy)
  fx[2] = main.mu*(vx + uy)
  fx[3] = fx[1]*v1 + fx[2]*v2 + kTx
  return fx


def evalViscousFluxYNS_IP(main,u,Ux,Uy):
  gamma = 1.4
  Pr = 0.72
  ux = 1./u[0]*Ux[1] - u[1]/u[0]**2*Ux[0]
  ## ->  v_x = 1/rho d/dx(rho v) - rho v /rho^2 rho_x
  vx = 1./u[0]*Ux[2] - u[2]/u[0]**2*Ux[0]
  ## ->  u_y = 1/rho d/dy(rho u) - rho u /rho^2 rho_y
  uy = 1./u[0]*Uy[1] - u[1]/u[0]**2*Uy[0]
  ## ->  v_y = 1/rho d/dy(rho v) - rho v /rho^2 rho_y
  vy = 1./u[0]*Uy[2] - u[2]/u[0]**2*Uy[0]
  ## -> (kT)_x =d/dx[ (mu gamma)/Pr*(E - 1/2 v^2 ) ]
  ## ->        =mu gamma/Pr *[ dE/dx - 0.5 d/dx(v1^2 + v2^2) ]
  ## ->        =mu gamma/Pr *[ dE/dx - (v1 v1_x + v2 v2_x) ]
  ## ->  E_x = 1/rho d/x(rho E) - rho E /rho^2 rho_x
  kTx =( 1./u[0]*Ux[3] - u[3]/u[0]**2*Ux[0] - 1./u[0]*(u[1]*ux + u[2]*vx)  )*main.mu*gamma/Pr
  ## -> (kT)_y =d/dy[ (mu gamma)/Pr*(E - 1/2 v^2 ) ]
  ## ->        =mu gamma/Pr *[ dE/dy - 0.5 d/dy(v1^2 + v2^2) ]
  ## ->        =mu gamma/Pr *[ dE/dy - (v1 v1_y + v2 v2_y) ]
  ## ->  E_x = 1/rho d/y(rho E) - rho E /rho^2 rho_y
  kTy =( 1./u[0]*Uy[3] - u[3]/u[0]**2*Uy[0] - 1./u[0]*(u[1]*uy + u[2]*vy)  )*main.mu*gamma/Pr

  fy = np.zeros(np.shape(u))
  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  fy[1] = main.mu*(vx + uy)
  fy[2] = main.mu*(4./3.*vy - 2./3.*ux)
  fy[3] = fy[1]*v1 + fy[2]*v2 + kTy
  return fy


### Linear Advection

def getGsLA(u,main):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  ashape = np.array(np.shape(u[0]))
  ashape = np.insert(ashape,0,nvars)
  ashape = np.insert(ashape,0,nvars)
  G11 = np.zeros(ashape)
  G12 = np.zeros(ashape)
  G21 = np.zeros(ashape)
  G22 = np.zeros(ashape)
  G11[0] = 1.
  G22[0] = 1.
  G11 = G11*main.mu
  G12 = G12*main.mu
  G21 = G21*main.mu
  G22 = G22*main.mu
  return G11,G12,G21,G22

def evalViscousFluxXLA_IP(main,u,Ux,Uy):
  fx = np.zeros(np.shape(u))
  fx[0] = main.mu*Ux[0]
  return fx

def evalViscousFluxYLA_IP(main,u,Ux,Uy):
  fy = np.zeros(np.shape(u))
  fy[0] = main.mu*Uy[0]
  return fy

