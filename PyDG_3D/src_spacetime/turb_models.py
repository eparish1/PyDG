import matplotlib.pyplot as plt
import numpy as np
from MPI_functions import gatherSolSpectral
from equations_class import *
from tensor_products import *
def orthogonalDynamics(main,MZ,eqns,schemes):
    ### EVAL RESIDUAL AND DO MZ STUFF
    MZ.a.a[:,:,:,:,:,:,:] = main.a.a[:,:,:,:,:,:,:]
    MZ.getRHS(MZ,eqns,schemes)
    RHS1 = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.rorder,0:main.rorder,0:main.rorder] = main.a.a[:,0:main.rorder,0:main.rorder,0:main.rorder]
    MZ.getRHS(MZ,eqns,schemes)
    RHS2 = np.zeros(np.shape(MZ.RHS))
    RHS2[:] = MZ.RHS[:]
    return RHS1 - RHS2,0


def DNS(main,MZ,eqns):
  eqns.getRHS(main,MZ,eqns)


## Evaluate the tau model through the FD approximation. This is expensive AF

def tauModelFD(main,MZ,eqns):
    ### EVAL RESIDUAL AND DO MZ STUFF
    filtarray = np.zeros(np.shape(MZ.a.a))
    filtarray[:,0:main.order[0],0:main.order[1],0:main.order[2]] = 1.
    eps = 1.e-5
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    eqns.getRHS(main,main,eqns)
    print('here')

    eqns.getRHS(MZ,MZ,eqns)
    RHS1 = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*RHS1[:]
    eqns.getRHS(MZ,MZ,eqns)
    RHS2 = np.zeros(np.shape(MZ.RHS))
    RHS2[:] = MZ.RHS[:]
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*RHS1[:]*filtarray
    eqns.getRHS(MZ,MZ,eqns)
    RHS3 = np.zeros(np.shape(MZ.RHS))
    RHS3[:] = MZ.RHS[:]
    PLQLU = (RHS2[:,0:main.order[0],0:main.order[1],0:main.order[2]] - RHS3[:,0:main.order[0],0:main.order[1],0:main.order[2]])/eps
    main.RHS[:] =  RHS1[:,0:main.order[0],0:main.order[1],0:main.order[2]] + main.dx/MZ.order[0]**2*PLQLU


def FM1Linearized(main,MZ,eqns):
   filtarray = np.zeros(np.shape(MZ.a.a))
   filtarray[:,0:main.order[0],0:main.order[1],0:main.order[2]] = 1.
   MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
   MZ.a.a[0:5,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[0:5]
   eqns.getRHS(MZ,MZ,eqns) #compute the residual in an enriched space
   tau = 0.1
   RHS1 = np.zeros(np.shape(MZ.RHS))
   RHS1f = np.zeros(np.shape(MZ.RHS))
   RHS1[:] = MZ.RHS[:]
   RHS1f[:] = RHS1[:] - RHS1[:]*filtarray
   eqnsLin = equations('Linearized Navier-Stokes',('central','Inviscid') )
   # now we need to compute the linearized RHS
   MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
   MZ.a.a[0:5,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[0:5]
   eqnsLin.getRHS(MZ,main,eqnsLin,[RHS1f]) ## this is PLQLu
   PLQLU = MZ.RHS[:,0:main.order[0],0:main.order[1],0:main.order[2]]
   main.RHS[0:5] =  RHS1[0:5,0:main.order[0],0:main.order[1],0:main.order[2]] + main.a.a[5::] 
   main.RHS[5::] = -2.*main.a.a[5::]/tau + 2.*PLQLU
   main.comm.Barrier()




def tauModelLinearized(main,MZ,eqns):
   filtarray = np.zeros(np.shape(MZ.a.a))
   filtarray[:,0:main.order[0],0:main.order[1],0:main.order[2],:,:,:] = 1.
   MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
   MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
   eqns.getRHS(MZ,MZ,eqns) #compute the residual in an enriched space
   RHS1 = np.zeros(np.shape(MZ.RHS))
   RHS1f = np.zeros(np.shape(MZ.RHS))
   RHS1[:] = MZ.RHS[:]
   RHS1f[:] = RHS1[:] - RHS1[:]*filtarray
   RHS1f_phys = main.basis.reconstructUGeneral(MZ,RHS1f)
   eqnsLin = equations('Linearized Navier-Stokes',('central','Inviscid'),'DNS' )
   # now we need to compute the linearized RHS
   MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
   MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
   eqnsLin.getRHS(MZ,MZ,eqnsLin,[RHS1f],[RHS1f_phys]) ## this is PLQLu
   PLQLU = MZ.RHS[:,0:main.order[0],0:main.order[1],0:main.order[2]]
   main.RHS[:] =  RHS1[:,0:main.order[0],0:main.order[1],0:main.order[2]] + main.dx/MZ.order[0]**2*PLQLU
   main.comm.Barrier()




def validateLinearized(main,MZ,eqns):
    # validation of the linearized equations
    # should have R(a + eps f) - R(a) / eps = Rlin(a)
    filtarray = np.zeros(np.shape(MZ.a.a))
    filtarray[:,0:main.order[0],0:main.order[1],0:main.order[2],:,:,:] = 1.
    MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    eqnsLin = equations('Linearized Navier-Stokes',('central','Inviscid') )
    eqnsLin.getRHS(MZ,MZ,eqns) #compute the residual in an enriched space
    RHSLin = np.zeros(np.shape(MZ.RHS))
    RHSLin[:] = MZ.RHS[:]
    # now we need to compute the linearized RHS via finite difference
    eps = 1.e-6
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    eqnsLin.getRHS(MZ,eqns,eqns)
    RHS1 = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*MZ.a.a[:]
    eqns.getRHS(MZ,eqns,eqns)
    RHS2 = np.zeros(np.shape(MZ.RHS))
    RHS2[:] = MZ.RHS[:]
    RHSLinFD = (RHS2 - RHS1)/eps
    print(np.linalg.norm(RHSLin),np.linalg.norm(RHSLinFD) )
    print(np.linalg.norm(RHSLin[:] -  RHSLinFD[:]) )
    main.RHS[:] =  RHS1[:,0:main.order[0],0:main.order[1],0:main.order[2]] #+ 0.0001*MZ.RHS[:,0:main.order[0],0:main.order[1],0:main.order[2]]


def tauModelValidateLinearized(main,MZ,eqns):
    filtarray = np.zeros(np.shape(MZ.a.a))
    filtarray[:,0:main.order[0],0:main.order[1],0:main.order[2],:,:,:] = 1.
    MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    eqns.getRHS(MZ,MZ,eqns) #compute the residual in an enriched space
    RHS1 = np.zeros(np.shape(MZ.RHS))
    RHS1f = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    RHS1f[:] = RHS1[:] - RHS1[:]*filtarray
    Z = reconstructUGeneral(main,main.a.a)
    eqnsLin = equations('Linearized Navier-Stokes',('central','Inviscid') )
    # now we need to compute the linearized RHS
    MZ.a.a[:] = 0.  #zero out state variable in MZ class and assign it to be that of the main class
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    eqnsLin.getRHS(MZ,main,eqnsLin,[RHS1f]) ## this is PLQLu
    PLQLULin = np.zeros(np.shape(MZ.RHS[:,0:main.order[0],0:main.order[1],0:main.order[2]]))
    PLQLULin[:] = MZ.RHS[:,0:main.order[0],0:main.order[1],0:main.order[2]]

#    print(np.linalg.norm(PLQLU[0]))

    eps = 1.e-5
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    eqns.getRHS(MZ,MZ,eqns)
    RHS1 = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*RHS1[:]
    eqns.getRHS(MZ,MZ,eqns)
    RHS2 = np.zeros(np.shape(MZ.RHS))
    RHS2[:] = MZ.RHS[:]
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order[0],0:main.order[1],0:main.order[2]] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*RHS1[:]*filtarray
    eqns.getRHS(MZ,MZ,eqns)
    RHS3 = np.zeros(np.shape(MZ.RHS))
    RHS3[:] = MZ.RHS[:]
    PLQLUFD = (RHS2[:,0:main.order[0],0:main.order[1],0:main.order[2]] - RHS3[:,0:main.order[0],0:main.order[1],0:main.order[2]])/eps

    print(np.linalg.norm(PLQLULin),np.linalg.norm(PLQLUFD) )
    print(np.linalg.norm(PLQLULin[4] - PLQLUFD[4] ))
    main.RHS[:] =  RHS1[:,0:main.order[0],0:main.order[1],0:main.order[2]] #+ 0.0001*MZ.RHS[:,0:main.order[0],0:main.order[1],0:main.order[2]]

    plt.clf()
    plt.plot(PLQLULin[4,1,0,0,:,0,0])
    plt.plot( PLQLUFD[4,1,0,0,:,0,0],'o')
    plt.pause(0.001)
    main.comm.Barrier()



def DtauModel(main,MZ,eqns,schemes):

  def sendScalar(scalar,main):
    if (main.mpi_rank == 0):
      for i in range(1,main.num_processes):
        loc_rank = i
        main.comm.Send(np.ones(1)*scalar,dest=loc_rank,tag=loc_rank)
      return scalar
    else:
      test = np.ones(1)*scalar
      main.comm.Recv(test,source=0,tag=main.mpi_rank)
      return test[0]

  ### EVAL RESIDUAL AND DO MZ STUFF
  filtarray = np.zeros(np.shape(MZ.a.a))
  filtarray[:,0:main.order,0:main.order,:,:] = 1.
  eps = 1.e-5
  MZ.a.a[:] = 0.
  MZ.a.a[:,0:main.order,0:main.order] = main.a.a[:]
  MZ.getRHS(MZ,eqns,schemes)
  RHS1 = np.zeros(np.shape(MZ.RHS))
  RHS1[:] = MZ.RHS[:]
  MZ.a.a[:] = 0.
  MZ.a.a[:,0:main.order,0:main.order] = main.a.a[:]
  MZ.a.a[:] = MZ.a.a[:] + eps*RHS1[:]
  MZ.getRHS(MZ,eqns,schemes)
  RHS2 = np.zeros(np.shape(MZ.RHS))
  RHS2[:] = MZ.RHS[:]
  MZ.a.a[:] = 0.
  MZ.a.a[:,0:main.order,0:main.order] = main.a.a[:]
  MZ.a.a[:] = MZ.a.a[:] + eps*RHS1[:]*filtarray
  MZ.getRHS(MZ,eqns,schemes)
  RHS3 = np.zeros(np.shape(MZ.RHS))
  RHS3[:] = MZ.RHS[:]
  PLQLU = (RHS2[:,0:main.order,0:main.order] - RHS3[:,0:main.order,0:main.order])/eps

  if (main.rkstage == 0):
    ### Now do dynamic procedure to get tau
    filtarray2 = np.zeros(np.shape(MZ.a.a))
    filtarray2[:,0:MZ.forder,0:MZ.forder,:,:] = 1.
    eps = 1.e-5
    ## Get RHS
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:MZ.forder,0:MZ.forder] = main.a.a[:,0:MZ.forder,0:MZ.forder]
    MZ.getRHS(MZ,eqns,schemes)
    RHS4 = np.zeros(np.shape(MZ.RHS))
    RHS4[:] = MZ.RHS[:]
    ## Now get RHS(a + eps*RHS)
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:MZ.forder,0:MZ.forder] = main.a.a[:,0:MZ.forder,0:MZ.forder]
    MZ.a.a[:] = MZ.a.a[:]*filtarray2 + eps*RHS4[:]
    MZ.getRHS(MZ,eqns,schemes)
    RHS5 = np.zeros(np.shape(MZ.RHS))
    RHS5[:] = MZ.RHS[:]
    ## Now get RHS(a + eps*RHSf)
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:MZ.forder,0:MZ.forder] = main.a.a[:,0:MZ.forder,0:MZ.forder]
    MZ.a.a[:] = MZ.a.a[:]*filtarray2 + eps*RHS4[:]*filtarray2
    MZ.getRHS(MZ,eqns,schemes)
    RHS6 = np.zeros(np.shape(MZ.RHS))
    RHS6[:] = MZ.RHS[:]
  
    ## Now compute PLQLUf
    PLQLUf = (RHS5[:,0:main.order,0:main.order] - RHS6[:,0:main.order,0:main.order])/eps
  
    PLQLUG = gatherSolSpectral(PLQLU,main)
    MZ.PLQLUG = PLQLUG
    PLQLUfG = gatherSolSpectral(PLQLUf[:,0:main.order,0:main.order],main)
    RHS1G = gatherSolSpectral(RHS1[:,0:main.order,0:main.order],main)
    RHS4G = gatherSolSpectral(RHS4[:,0:main.order,0:main.order],main)

    afG = gatherSolSpectral(main.a.a[:,0:main.order,0:main.order],main)

    if (main.mpi_rank == 0):
      num = 2.*np.mean(np.sum(afG[1:3,0:MZ.forder,0:MZ.forder]*(RHS4G[1:3,0:MZ.forder,0:MZ.forder] - RHS1G[1:3,0:MZ.forder,0:MZ.forder]),axis=(0,1,2)) ,axis=(0,1))
      den =  np.mean ( np.sum(afG[1:3,0:MZ.forder,0:MZ.forder]*(PLQLUG[1:3,0:MZ.forder,0:MZ.forder] - \
                                         (main.order/MZ.forder)*PLQLUfG[1:3,0:MZ.forder,0:MZ.forder]),axis=(0,1,2)) ,axis=(0,1))
      tau = num/(den + 1.e-1)
      print(tau)
    else:
      tau = 0.
    MZ.tau = np.maximum(0.,sendScalar(tau,main))
    #MZ.tau = sendScalar(tau,main)
  return 0.0002*PLQLU



def tauModel2(main,MZ,eqns,schemes):
    ### EVAL RESIDUAL AND DO MZ STUFF
    filtarray = np.ones(np.shape(main.a.a))
    filtarray[:,main.rorder::,main.rorder::,:,:] = 0.
    eps = 1.e-5
    MZ.a.a[:] = main.a.a[:]*filtarray
    main.getRHS(MZ,eqns,schemes)
    RHS1 = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    MZ.a.a[:] = main.a.a[:]*filtarray + eps*RHS1[:]
    main.getRHS(MZ,eqns,schemes)
    RHS2 = np.zeros(np.shape(MZ.RHS))
    RHS2[:] = MZ.RHS[:]
    MZ.a.a[:] = main.a.a[:] + eps*RHS1[:]*filtarray
    main.getRHS(MZ,eqns,schemes)
    RHS3 = np.zeros(np.shape(MZ.RHS))
    RHS3[:] = MZ.RHS[:]
    PLQLU = (RHS2 - RHS3)/eps
    return MZ.tau*PLQLU



def shockCapturingSetViscosity(main):
  ### Shock capturing
  filta = np.zeros(np.shape(main.a.a))
  filta[:,0:main.order[0]-1,main.order[1]-1,main.order[2]-1] = 1.
  af = main.a.a*filta    #make filtered state
  uf = reconstructUGeneral(main,af)
  udff = (main.a.u - uf)**2
  # now compute switch
  Se_num = volIntegrate(main.weights0,main.weights1,main.weights2,udff) 
  Se_den = volIntegrate(main.weights0,main.weights1,main.weights2,main.a.u**2)
  Se = (Se_num + 1e-10)/(Se_den + 1.e-30)
  eps0 = 1.*main.dx/main.order[0]
  s0 =1./main.order[0]**4
  kap = 5.
  se = np.log10(Se)
  #print(np.amax(udff))
  epse = eps0/2.*(1. + np.sin(np.pi/(2.*kap)*(se - s0) ) )
  epse[se<s0-kap] = 0.
  epse[se>s0  + kap] = eps0
  #plt.clf()
  #print(np.amax(epse),np.amin(epse))
  #plt.plot(epse[0,:,0,0])
  #plt.ylim([1e-9,0.005])
  #plt.pause(0.001)
  #print(np.shape(main.mu),np.shape(epse) )
  main.mu = main.mu0 + epse[0]
  main.muR = main.mu0R + epse[0]
  main.muL = main.mu0L + epse[0]
  main.muU = main.mu0F + epse[0]
  main.muD = main.mu0D + epse[0]
  main.muF = main.mu0U + epse[0]
  main.muB = main.mu0B + epse[0]





