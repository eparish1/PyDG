import numpy as np
from MPI_functions import gatherSolSpectral

def tauModel(main,MZ,eqns,schemes):
    ### EVAL RESIDUAL AND DO MZ STUFF
    filtarray = np.zeros(np.shape(MZ.a.a))
    filtarray[:,0:main.order,0:main.order,0:main.order,:,:,:] = 1.
    eps = 1.e-5
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order,0:main.order,0:main.order] = main.a.a[:]
    MZ.getRHS(MZ,eqns,schemes)
    RHS1 = np.zeros(np.shape(MZ.RHS))
    RHS1[:] = MZ.RHS[:]
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order,0:main.order,0:main.order] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*RHS1[:]
    MZ.getRHS(MZ,eqns,schemes)
    RHS2 = np.zeros(np.shape(MZ.RHS))
    RHS2[:] = MZ.RHS[:]
    MZ.a.a[:] = 0.
    MZ.a.a[:,0:main.order,0:main.order,0:main.order] = main.a.a[:]
    MZ.a.a[:] = MZ.a.a[:] + eps*RHS1[:]*filtarray
    MZ.getRHS(MZ,eqns,schemes)
    RHS3 = np.zeros(np.shape(MZ.RHS))
    RHS3[:] = MZ.RHS[:]
    PLQLU = (RHS2[:,0:main.order,0:main.order,0:main.order] - RHS3[:,0:main.order,0:main.order,0:main.order])/eps
    return MZ.tau*PLQLU


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

def DNS(main,MZ,eqns,schemes):
  return 0.
