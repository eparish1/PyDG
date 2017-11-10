import numpy as np
from eos_functions import *
import cantera as ct
def getNetProductionRates(main,U,mol_weight):
  ## Reaction is
  # CH4 + 1.5 O2 => CO + 2*H2O
  # CO  + 0.5 O2 <=> CO2 
  gas = ct.Solution('2s_ch4_bfer.xml')
  p,T = computePressure_and_Temperature_CPG(main,U)
  pm = np.mean(p)
  Tm = np.mean(T)
  Ym = main.a.u[5::,0,0,0,0,0,0,0,0]/main.a.u[0,0,0,0,0,0,0,0,0] 
  Y_last = 1. - np.sum(U[5::]/U[None,0],axis=0)
  Ym = np.append(Ym,Y_last[0,0,0,0,0,0,0,0])
  gas.TPY =Tm,pm,Ym
  Winv =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],U[5::]/U[None,0]) + 1./main.W[-1]*Y_last
  X = 1./Winv[None]*1./main.W[0:-1,None,None,None,None,None,None,None,None] * U[5::]/U[None,0]
  X_last = 1./Winv*1./main.W[-1] * Y_last
  conc_den = np.sum(X*main.W[0:-1,None,None,None,None,None,None,None,None],axis=0) + X_last*main.W[-1]
  k1 = get_kf1(T,X,main.W,U[0],conc_den)
  k2 = get_kf2(T,X,main.W,U[0],conc_den)
  rates = np.zeros(np.shape(X))
  #print(np.shape(X),np.shape(U),np.shape(k1),np.shape(T),np.shape(conc_den),np.shape(main.W))
  rates[0] -= 1.5*k1 #O2 reaction 1 
  rates[0] -= 0.5*k2 #O2 reaction 2
  rates[1] += 2.*k1  #H2O reaction 1
  rates[2] -= k1     #CH4 reaction 1
  rates[3] += k1
  rates[3] -= k2
  rates[4] += k2

  rates2 = gas.net_production_rates
  print(rates[:,0,0,0,0,0,0,0,0] ,rates2[0:-1])
  print(np.linalg.norm(rates[:,0,0,0,0,0,0,0,0] - rates2[0:-1]))
  for i in range(0,5):
    rates[i,:] = rates2[i]
  #print(np.linalg.norm(rates - rates2))
  return rates

def get_kf1(T,X,mol_weight,rho,conc_den):
  #X_conc = X*rho[None]/np.sum(X*mol_weight,axis=0)
  X_conc = X*rho[None]/conc_den
  Ea = 35500 #cal/mol
  Ea = Ea*4.184 #J/mol
  n_ch4 = 0.5
  n_o2 = 0.65
  A = 4.9E+09 #1/s
  beta1 = 0
  X_CH4 = X_conc[2]/100**3 #mol/cm^3
  X_O2 = X_conc[0]/100**3 #mol/cm^3
  R = 8.314 #J/mol
  phi01 = 1.1
  sigma01 = 0.09
  B1 = 0.37
  phi11 = 1.13
  sigma11 = 0.03
  C1 = 6.7
  phi21 = 1.6
  sigma21 = 0.22
  phi = 2.
  f1 = 2./( (1 + np.tanh( (phi01 - phi)/sigma01) ) + B1*(1. + np.tanh( (phi - phi11)/(sigma11) ) ) + C1*(1. + np.tanh( (phi - phi21)/sigma21) ) )
  kf1 = A*np.exp(-Ea/(R*T))*X_CH4**n_ch4 * X_O2**n_o2
  # 1/s * exp( J/ mol * (mol * K / J) / K
  # 1/s * (mol/m^3) * (mol/m^3)
  return kf1/1000.*100**3



def get_kf2(T,X,mol_weight,rho,conc_den):
  #X_conc = X*rho[None,:]/np.sum(X*mol_weight,axis=0)
  X_conc = X*rho[None,:]/conc_den
  Ea = 1.2E+04 #cal/mol
  Ea = Ea*4.184 #J/mol
  n_co = 1.
  n_o2 = 0.5
  A = 2E+08 #1/s
  beta1 = 0.7
  X_CO = X_conc[3]/100**3 #mol/cm^3
  X_O2 = X_conc[0]/100**3 #mol/cm^3
  R = 8.314 #J/mol
  kf2 = A*T**beta1*np.exp(-Ea/(R*T))*X_CO**n_co * X_O2**n_o2
  # 1/s * exp( J/ mol * (mol * K / J) / K
  # 1/s * (mol/m^3) * (mol/m^3)
  return kf2/1000.*100**3

