import numpy as np
import time

def computePressure_and_Temperature_euler(main,u):
  gamma = 1.4
  T = 1./u[0]**2(u[4]*u[0] - 0.5*( u[1]**2 + u[2]**2 + u[3]**2 ) )
  p = (gamma - 1.)*(u[4] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0] - 0.5*u[3]**2/u[0]) 
  return p,T


def update_state(main):
    main.a.p[:],main.a.T[:] = computePressure_and_Temperature(main,main.a.u)
    main.a.pR[:],main.a.TR[:] = computePressure_and_Temperature(main,main.a.uR)
    main.a.pL[:],main.a.TL[:] = computePressure_and_Temperature(main,main.a.uL)
    main.a.pU[:],main.a.TU[:] = computePressure_and_Temperature(main,main.a.uU)
    main.a.pD[:],main.a.TD[:] = computePressure_and_Temperature(main,main.a.uD)
    main.a.pF[:],main.a.TF[:] = computePressure_and_Temperature(main,main.a.uF)
    main.a.pB[:],main.a.TB[:] = computePressure_and_Temperature(main,main.a.uB)

    main.a.pR_edge[:],main.a.TR_edge[:] = computePressure_and_Temperature(main,main.a.uR_edge)
    main.a.pL_edge[:],main.a.TL_edge[:] = computePressure_and_Temperature(main,main.a.uL_edge)
    main.a.pU_edge[:],main.a.TU_edge[:] = computePressure_and_Temperature(main,main.a.uU_edge)
    main.a.pD_edge[:],main.a.TD_edge[:] = computePressure_and_Temperature(main,main.a.uD_edge)
    main.a.pF_edge[:],main.a.TF_edge[:] = computePressure_and_Temperature(main,main.a.uF_edge)
    main.a.pB_edge[:],main.a.TB_edge[:] = computePressure_and_Temperature(main,main.a.uB_edge)

    fa = np.zeros((np.size(main.a.u[0]),np.shape(main.a.u)[0]-5 + 1))
    for i in range(0,np.shape(main.a.u)[0]-5):
      fa[:,i] = ( main.a.u[5+i]/main.a.u[0] ).flatten()
    fa[:,-1] = 1. - np.sum(fa[:,0:-1],axis=1)
    if (main.fsource):
     main.cgas_field.TPY = main.a.T.flatten(),main.a.p.flatten(),fa 

def update_state_cantera(main):
    rhoi = 1./main.a.u[0]
    rhoiL = 1./main.a.uL[0]
    rhoiR = 1./main.a.uR[0]
    rhoiU = 1./main.a.uU[0]
    rhoiD = 1./main.a.uD[0]
    rhoiF = 1./main.a.uF[0]
    rhoiB = 1./main.a.uB[0]
    rhoiL_edge = 1./main.a.uL_edge[0]
    rhoiR_edge = 1./main.a.uR_edge[0]
    rhoiU_edge = 1./main.a.uU_edge[0]
    rhoiD_edge = 1./main.a.uD_edge[0]
    rhoiF_edge = 1./main.a.uF_edge[0]
    rhoiB_edge = 1./main.a.uB_edge[0]
#
    fa = np.zeros((np.size(main.a.u[0]),np.shape(main.a.u)[0]-5 + 1))
    faR = np.zeros((np.size(main.a.uL[0]),np.shape(main.a.uR)[0]-5+1))
    faL = np.zeros((np.size(main.a.uR[0]),np.shape(main.a.uL)[0]-5+1))
    faU = np.zeros((np.size(main.a.uU[0]),np.shape(main.a.uU)[0]-5+1))
    faD = np.zeros((np.size(main.a.uD[0]),np.shape(main.a.uD)[0]-5+1))
    faF = np.zeros((np.size(main.a.uF[0]),np.shape(main.a.uF)[0]-5+1))
    faB = np.zeros((np.size(main.a.uB[0]),np.shape(main.a.uB)[0]-5+1))
    faR_edge = np.zeros((np.size(main.a.uL_edge[0]),np.shape(main.a.uR_edge)[0]-5+1))
    faL_edge = np.zeros((np.size(main.a.uR_edge[0]),np.shape(main.a.uL_edge)[0]-5+1))
    faU_edge = np.zeros((np.size(main.a.uU_edge[0]),np.shape(main.a.uU_edge)[0]-5+1))
    faD_edge = np.zeros((np.size(main.a.uD_edge[0]),np.shape(main.a.uD_edge)[0]-5+1))
    faF_edge = np.zeros((np.size(main.a.uF_edge[0]),np.shape(main.a.uF_edge)[0]-5+1))
    faB_edge = np.zeros((np.size(main.a.uB_edge[0]),np.shape(main.a.uB_edge)[0]-5+1))

    for i in range(0,np.shape(main.a.u)[0]-5):
      fa[:,i] = ( main.a.u[5+i]*rhoi ).flatten()
      faR[:,i] = ( main.a.uR[5+i]*rhoiR ).flatten()
      faL[:,i] = ( main.a.uL[5+i]*rhoiL ).flatten()
      faU[:,i] = ( main.a.uU[5+i]*rhoiU ).flatten()
      faD[:,i] = ( main.a.uD[5+i]*rhoiD ).flatten()
      faF[:,i] = ( main.a.uF[5+i]*rhoiF ).flatten()
      faB[:,i] = ( main.a.uB[5+i]*rhoiB ).flatten()
      faR_edge[:,i] = ( main.a.uR_edge[5+i]*rhoiR_edge ).flatten()
      faL_edge[:,i] = ( main.a.uL_edge[5+i]*rhoiL_edge ).flatten()
      faU_edge[:,i] = ( main.a.uU_edge[5+i]*rhoiU_edge ).flatten()
      faD_edge[:,i] = ( main.a.uD_edge[5+i]*rhoiD_edge ).flatten()
      faF_edge[:,i] = ( main.a.uF_edge[5+i]*rhoiF_edge ).flatten()
      faB_edge[:,i] = ( main.a.uB_edge[5+i]*rhoiB_edge ).flatten()
    fa[:,-1] = 1. - np.sum(fa[:,0:-1],axis=1)
    faR[:,-1] = 1. - np.sum(faR[:,0:-1],axis=1)
    faL[:,-1] = 1. - np.sum(faL[:,0:-1],axis=1)
    faU[:,-1] = 1. - np.sum(faU[:,0:-1],axis=1)
    faD[:,-1] = 1. - np.sum(faD[:,0:-1],axis=1)
    faF[:,-1] = 1. - np.sum(faF[:,0:-1],axis=1)
    faB[:,-1] = 1. - np.sum(faB[:,0:-1],axis=1)
    faR_edge[:,-1] = 1. - np.sum(faR_edge[:,0:-1],axis=1)
    faL_edge[:,-1] = 1. - np.sum(faL_edge[:,0:-1],axis=1)
    faU_edge[:,-1] = 1. - np.sum(faU_edge[:,0:-1],axis=1)
    faD_edge[:,-1] = 1. - np.sum(faD_edge[:,0:-1],axis=1)
    faF_edge[:,-1] = 1. - np.sum(faF_edge[:,0:-1],axis=1)
    faB_edge[:,-1] = 1. - np.sum(faB_edge[:,0:-1],axis=1)



    e = main.a.u[4]*rhoi - 0.5*rhoi**2*(main.a.u[1]**2 + main.a.u[2]**2 + main.a.u[3]**2)
    eR = main.a.uR[4]*rhoiR - 0.5*rhoiR**2*(main.a.uR[1]**2 + main.a.uR[2]**2 + main.a.uR[3]**2)
    eL = main.a.uL[4]*rhoiL - 0.5*rhoiL**2*(main.a.uL[1]**2 + main.a.uL[2]**2 + main.a.uL[3]**2)
    eU = main.a.uU[4]*rhoiU - 0.5*rhoiU**2*(main.a.uU[1]**2 + main.a.uU[2]**2 + main.a.uU[3]**2)
    eD = main.a.uD[4]*rhoiD - 0.5*rhoiD**2*(main.a.uD[1]**2 + main.a.uD[2]**2 + main.a.uD[3]**2)
    eF = main.a.uF[4]*rhoiF - 0.5*rhoiF**2*(main.a.uF[1]**2 + main.a.uF[2]**2 + main.a.uF[3]**2)
    eB = main.a.uB[4]*rhoiB - 0.5*rhoiB**2*(main.a.uB[1]**2 + main.a.uB[2]**2 + main.a.uB[3]**2)
    eR_edge = main.a.uR_edge[4]*rhoiR_edge - 0.5*rhoiR_edge**2*(main.a.uR_edge[1]**2 + main.a.uR_edge[2]**2 + main.a.uR_edge[3]**2)
    eL_edge = main.a.uL_edge[4]*rhoiL_edge - 0.5*rhoiL_edge**2*(main.a.uL_edge[1]**2 + main.a.uL_edge[2]**2 + main.a.uL_edge[3]**2)
    eU_edge = main.a.uU_edge[4]*rhoiU_edge - 0.5*rhoiU_edge**2*(main.a.uU_edge[1]**2 + main.a.uU_edge[2]**2 + main.a.uU_edge[3]**2)
    eD_edge = main.a.uD_edge[4]*rhoiD_edge - 0.5*rhoiD_edge**2*(main.a.uD_edge[1]**2 + main.a.uD_edge[2]**2 + main.a.uD_edge[3]**2)
    eF_edge = main.a.uF_edge[4]*rhoiF_edge - 0.5*rhoiF_edge**2*(main.a.uF_edge[1]**2 + main.a.uF_edge[2]**2 + main.a.uF_edge[3]**2)
    eB_edge = main.a.uB_edge[4]*rhoiB_edge - 0.5*rhoiB_edge**2*(main.a.uB_edge[1]**2 + main.a.uB_edge[2]**2 + main.a.uB_edge[3]**2)
    main.cgas_field.UVY = e.flatten(),rhoi.flatten(),fa
    main.cgas_field_R.UVY = eR.flatten(),rhoiR.flatten(),faR
    main.cgas_field_L.UVY = eL.flatten(),rhoiL.flatten(),faL
    main.cgas_field_U.UVY = eU.flatten(),rhoiU.flatten(),faU
    main.cgas_field_D.UVY = eD.flatten(),rhoiD.flatten(),faD
    main.cgas_field_F.UVY = eF.flatten(),rhoiF.flatten(),faF
    main.cgas_field_B.UVY = eB.flatten(),rhoiB.flatten(),faB
    main.cgas_field_R_edge.UVY = eR_edge.flatten(),rhoiR_edge.flatten(),faR_edge
    main.cgas_field_L_edge.UVY = eL_edge.flatten(),rhoiL_edge.flatten(),faL_edge
    main.cgas_field_U_edge.UVY = eU_edge.flatten(),rhoiU_edge.flatten(),faU_edge
    main.cgas_field_D_edge.UVY = eD_edge.flatten(),rhoiD_edge.flatten(),faD_edge
    main.cgas_field_F_edge.UVY = eF_edge.flatten(),rhoiF_edge.flatten(),faF_edge
    main.cgas_field_B_edge.UVY = eB_edge.flatten(),rhoiB_edge.flatten(),faB_edge
 
    main.a.p[:] = np.reshape(main.cgas_field.P,np.shape(main.a.u[0]) )
    main.a.pR[:] = np.reshape(main.cgas_field_R.P,np.shape(main.a.uR[0]) )
    main.a.pL[:] = np.reshape(main.cgas_field_L.P,np.shape(main.a.uL[0]) )
    main.a.pU[:] = np.reshape(main.cgas_field_U.P,np.shape(main.a.uU[0]) )
    main.a.pD[:] = np.reshape(main.cgas_field_D.P,np.shape(main.a.uD[0]) )
    main.a.pF[:] = np.reshape(main.cgas_field_F.P,np.shape(main.a.uF[0]) )
    main.a.pB[:] = np.reshape(main.cgas_field_B.P,np.shape(main.a.uB[0]) )
    main.a.pR_edge[:] = np.reshape(main.cgas_field_R_edge.P,np.shape(main.a.uR_edge[0]) )
    main.a.pL_edge[:] = np.reshape(main.cgas_field_L_edge.P,np.shape(main.a.uL_edge[0]) )
    main.a.pU_edge[:] = np.reshape(main.cgas_field_U_edge.P,np.shape(main.a.uU_edge[0]) )
    main.a.pD_edge[:] = np.reshape(main.cgas_field_D_edge.P,np.shape(main.a.uD_edge[0]) )
    main.a.pF_edge[:] = np.reshape(main.cgas_field_F_edge.P,np.shape(main.a.uF_edge[0]) )
    main.a.pB_edge[:] = np.reshape(main.cgas_field_B_edge.P,np.shape(main.a.uB_edge[0]) )


#   main.a.T[:] = np.reshape(main.cgas_field.T,np.shape(main.a.u[0]) )
    main.a.T[:] = np.reshape(main.cgas_field.T,np.shape(main.a.u[0]) )
    main.a.TR[:] = np.reshape(main.cgas_field_R.T,np.shape(main.a.uR[0]) )
    main.a.TL[:] = np.reshape(main.cgas_field_L.T,np.shape(main.a.uL[0]) )
    main.a.TU[:] = np.reshape(main.cgas_field_U.T,np.shape(main.a.uU[0]) )
    main.a.TD[:] = np.reshape(main.cgas_field_D.T,np.shape(main.a.uD[0]) )
    main.a.TF[:] = np.reshape(main.cgas_field_F.T,np.shape(main.a.uF[0]) )
    main.a.TB[:] = np.reshape(main.cgas_field_B.T,np.shape(main.a.uB[0]) )
    main.a.TR_edge[:] = np.reshape(main.cgas_field_R_edge.T,np.shape(main.a.uR_edge[0]) )
    main.a.TL_edge[:] = np.reshape(main.cgas_field_L_edge.T,np.shape(main.a.uL_edge[0]) )
    main.a.TU_edge[:] = np.reshape(main.cgas_field_U_edge.T,np.shape(main.a.uU_edge[0]) )
    main.a.TD_edge[:] = np.reshape(main.cgas_field_D_edge.T,np.shape(main.a.uD_edge[0]) )
    main.a.TF_edge[:] = np.reshape(main.cgas_field_F_edge.T,np.shape(main.a.uF_edge[0]) )
    main.a.TB_edge[:] = np.reshape(main.cgas_field_B_edge.T,np.shape(main.a.uB_edge[0]) )
 
 
def computePressure_and_Temperature_Cantera(main,U,cgas_field):
  t0 = time.time()
  ## need to go from total energy to internal energy
  e = np.zeros(np.shape(U[4]))
  e[:] = U[4]                                            #this is rhoE
  rhoi = 1./U[0]                                         #cmopute 1./rho 
  e*= rhoi                                               # now get e
  u,v,w = U[1]*rhoi, U[2]*rhoi, U[3]*rhoi                # get total kinetic energy
  e -= 0.5*(u**2 + v**2 + w**2)                          # subtract total KE
   
  fa = np.zeros((np.size(U[0]),np.shape(U)[0]-5 + 1))    # now compute mass fractions and reshape for cantera
  for i in range(0,np.shape(U)[0]-5):
    fa[:,i] = ( U[5+i]*rhoi ).flatten()
  fa[:,-1] = 1. - np.sum(fa[:,0:-1],axis=1)
  t1 = time.time()

  cgas_field.UVY = e.flatten(),rhoi.flatten(),fa
  t2 = time.time()
#  if (main.mpi_rank == 0):
#    print('My calc times = ' + str(t1 - t0) + '  Cantera calc times = ' + str(t2 - t1) )
  return np.reshape(cgas_field.P,np.shape(U[0])),np.reshape(cgas_field.T,np.shape(U[0]))

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


def computePressure_CPG(main,u):
  R = 8314.4621/1000.
  Y_last = 1. - np.sum(u[5::]/u[None,0],axis=0)
  Winv =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],u[5::]/u[None,0]) + 1./main.W[-1]*Y_last
  Cp = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],u[5::]/u[None,0]) + main.Cp[-1]*Y_last
  Cv = Cp - R*Winv
  gamma = Cp/Cv
  p = (gamma - 1.)*(u[4] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0] - 0.5*u[3]**2/u[0])
  return p

def computePressure_and_Temperature(main,u):
  R = 8314.4621/1000.
  T0 = 298.15*0.
  n_reacting = np.size(main.delta_h0)
  Y_last = 1. - np.sum(u[5::]/u[None,0],axis=0)
  Winv =  np.einsum('i...,ijk...->jk...',1./main.W[0:-1],u[5::]/u[None,0]) + 1./main.W[-1]*Y_last
  Cp = np.einsum('i...,ijk...->jk...',main.Cp[0:-1],u[5::]/u[None,0]) + main.Cp[-1]*Y_last
  Cv = Cp - R*Winv
  gamma = Cp/Cv
  # sensible + chemical
  T = u[4]/u[0] - 0.5/u[0]**2*( u[1]**2 + u[2]**2 + u[3]**2 )
  T += R * T0 * Winv 
  T /= Cv
  T += T0
  p = (gamma - 1.)*(u[4] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0] - 0.5*u[3]**2/u[0])
  return p,T


def nasa_get_cps(main,T):
  T0 = 298.
  a = main.nasa_coeffs[:,0:7]
  R = 8314.4621/1000./main.W
  cp = R*(a[:,0]*T + a[:,1]*T**2/2. + a[:,2]*T**3/3. + a[:,3]*T**4/4. + a[:,4]*T**5/5.)
  cp /= T
  #cp -= R*(a[:,0]*T0 + a[:,1]*T0**2/2. + a[:,2]*T0**3/3. + a[:,3]*T0**4/4. + a[:,4]*T0**5/5.)
  return cp
